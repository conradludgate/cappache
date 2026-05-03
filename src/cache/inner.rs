use std::any::Any;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hashbrown::{HashMap, HashSet};
use tokio::sync::watch;

use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

/// Settled state of a miss flight broadcast through [`MissFlight::settle`].
///
/// `Cancelled` is published by [`LeaderGuard::drop`] when the leader's
/// `init` future was dropped or panicked before producing a value; followers
/// translate it into a probe retry rather than reporting the leader's
/// never-arrived result. Without this sentinel a stale `MapSlot::Miss` would
/// outlive the leader and deadlock every subsequent caller for that key.
#[derive(Clone)]
pub(crate) enum FlightMsg<V> {
    Ok(V),
    Err(Arc<dyn Any + Send + Sync>),
    Cancelled,
}

impl<V: Clone> FlightMsg<V> {
    pub(crate) fn from_result<E: Send + Sync + 'static>(r: Result<V, E>) -> Self {
        match r {
            Ok(v) => FlightMsg::Ok(v),
            Err(e) => FlightMsg::Err(Arc::new(e)),
        }
    }

    /// Returns `None` when the leader was cancelled; the follower must retry
    /// the probe loop and may itself be promoted to leader.
    pub(crate) fn into_settled<E: Clone + Send + Sync + 'static>(
        self,
    ) -> Option<Result<V, E>> {
        match self {
            FlightMsg::Ok(v) => Some(Ok(v)),
            FlightMsg::Err(arc) => Some(Err(Arc::downcast::<E>(arc)
                .expect("miss flight error type mismatch: E must match the stored miss error type")
                .as_ref()
                .clone())),
            FlightMsg::Cancelled => None,
        }
    }
}

pub(crate) struct MissFlight<V> {
    pub(crate) settle: watch::Sender<Option<FlightMsg<V>>>,
    pub(crate) _hold: watch::Receiver<Option<FlightMsg<V>>>,
}

pub(crate) struct Inner<K: 'static, V: 'static> {
    pub(crate) config: CacheConfig<K, V>,
    /// All `Ready` / `Miss` map state. **`std::sync::Mutex` only** — never `tokio::sync::Mutex`; the
    /// lock is released before awaiting user `init` futures in [`Self::try_get_with`].
    pub(crate) state: Mutex<State<K, V>>,
}

/// Stable handle into [`Slab`]. `u32` caps slab population at 2^32-1, which
/// far exceeds any practical cache bounded by RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SlotId(u32);

pub(crate) struct State<K, V> {
    /// Lookup map: key → `Ready` slab id or `Miss` flight handle.
    pub(crate) entries: HashMap<K, MapSlot<V>>,
    /// Backing storage for `Ready` entries. The slab owns `CacheEntry`,
    /// the back-reference key, the cached weight, and the intrusive LRU links.
    slab: Slab<K, V>,
    /// Intrusive LRU over slab slots. `head` is MRU, `tail` is LRU.
    evict_list: EvictList,
    /// Min-heap of (deadline, slot, generation) tuples used to find past-deadline
    /// entries in `O(log n)`. Validated lazily: a popped tuple is honoured
    /// only if the slab still holds an `Occupied` slot at `slot` whose `generation`
    /// matches and whose deadline is exactly `Some(deadline)`. Stale tuples
    /// (slot reused, deadline shifted by a read hook, or slot evicted) are
    /// dropped on pop. Read-deadline updates push a new tuple instead of
    /// removing the old one.
    expiry: ExpiryHeap,
    pub(crate) refresh_inflight: HashSet<K>,
    /// Sum of [`Policy::weight`](crate::policy::Policy::weight) over `Ready` entries only.
    pub(crate) total_weight: u64,
}

pub(crate) enum MapSlot<V> {
    Ready(SlotId),
    Miss(Arc<MissFlight<V>>),
}

pub(crate) struct CacheEntry<V> {
    pub(crate) value: V,
    pub(crate) written_at: Instant,
    pub(crate) deadline: Option<Instant>,
}

/// Owned cache entry held inside [`Slab`]. The `key` back-reference lets eviction
/// find and remove the corresponding `entries` map slot in O(1).
struct EntrySlot<K, V> {
    key: K,
    entry: CacheEntry<V>,
    /// Cached `Policy::weight` so eviction does not need to re-call the user policy.
    weight: u64,
    /// Bumped on every slot reuse. Heap tuples carry the generation at push time and
    /// fail validation on pop after the slot has been recycled.
    generation: u32,
    lru_prev: Option<SlotId>,
    lru_next: Option<SlotId>,
}

enum SlabEntry<K, V> {
    Occupied(EntrySlot<K, V>),
    /// `generation` is the generation of the **last** occupant; the next insert into
    /// this slot uses `generation.wrapping_add(1)`.
    Free { next_free: Option<u32>, generation: u32 },
}

/// Vector-backed slab with a free-list. Pure-safe (no `unsafe`).
struct Slab<K, V> {
    slots: Vec<SlabEntry<K, V>>,
    free_head: Option<u32>,
    len: usize,
}

impl<K, V> Slab<K, V> {
    fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            free_head: None,
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Install `entry` and return its slot id. Callers must set `entry.generation`
    /// to whatever this method returns from [`Self::next_generation_for`] beforehand
    /// — the slab does not mutate `entry` in place to keep the borrow simple.
    fn insert(&mut self, entry: EntrySlot<K, V>) -> SlotId {
        if let Some(idx) = self.free_head {
            let slot = &mut self.slots[idx as usize];
            let next_free = match slot {
                SlabEntry::Free { next_free, .. } => *next_free,
                SlabEntry::Occupied(_) => unreachable!("free list points at occupied slot"),
            };
            self.free_head = next_free;
            *slot = SlabEntry::Occupied(entry);
            self.len += 1;
            SlotId(idx)
        } else {
            let idx = u32::try_from(self.slots.len()).expect("slab capacity exceeds u32::MAX");
            self.slots.push(SlabEntry::Occupied(entry));
            self.len += 1;
            SlotId(idx)
        }
    }

    /// Generation an `EntrySlot` should claim if installed at `idx` next.
    /// Returns `0` for a fresh tail-extension slot, or `prev_generation.wrapping_add(1)`
    /// for an existing slot on the free list.
    fn next_generation_for(&self, idx: u32) -> u32 {
        match self.slots.get(idx as usize) {
            Some(SlabEntry::Free { generation, .. }) => generation.wrapping_add(1),
            Some(SlabEntry::Occupied(_)) | None => 0,
        }
    }

    /// Index of the next slot `insert` will populate, used to look up
    /// [`Self::next_generation_for`] before the new `EntrySlot` is constructed.
    fn next_slot_index(&self) -> u32 {
        match self.free_head {
            Some(idx) => idx,
            None => u32::try_from(self.slots.len()).expect("slab capacity exceeds u32::MAX"),
        }
    }

    fn remove(&mut self, id: SlotId) -> Option<EntrySlot<K, V>> {
        let slot = self.slots.get_mut(id.0 as usize)?;
        if !matches!(slot, SlabEntry::Occupied(_)) {
            return None;
        }
        let prev_free = self.free_head;
        let taken = std::mem::replace(
            slot,
            SlabEntry::Free {
                next_free: prev_free,
                generation: 0, // overwritten below from the taken EntrySlot.
            },
        );
        let SlabEntry::Occupied(entry) = taken else {
            unreachable!("checked Occupied above")
        };
        // Re-stamp the Free's `generation` to the removed entry's generation so the next
        // occupant computes its `generation + 1` cleanly.
        if let SlabEntry::Free { generation, .. } = slot {
            *generation = entry.generation;
        }
        self.free_head = Some(id.0);
        self.len -= 1;
        Some(entry)
    }

    fn get(&self, id: SlotId) -> Option<&EntrySlot<K, V>> {
        match self.slots.get(id.0 as usize)? {
            SlabEntry::Occupied(e) => Some(e),
            SlabEntry::Free { .. } => None,
        }
    }

    fn get_mut(&mut self, id: SlotId) -> Option<&mut EntrySlot<K, V>> {
        match self.slots.get_mut(id.0 as usize)? {
            SlabEntry::Occupied(e) => Some(e),
            SlabEntry::Free { .. } => None,
        }
    }
}

#[derive(Default)]
struct EvictList {
    head: Option<SlotId>,
    tail: Option<SlotId>,
}

impl EvictList {
    fn push_mru<K, V>(&mut self, slab: &mut Slab<K, V>, id: SlotId) {
        let new_node = slab
            .get_mut(id)
            .expect("evict_list.push_mru: slot must be occupied");
        new_node.lru_prev = None;
        new_node.lru_next = self.head;

        if let Some(old_head) = self.head {
            slab.get_mut(old_head)
                .expect("evict_list head must be occupied")
                .lru_prev = Some(id);
        } else {
            self.tail = Some(id);
        }
        self.head = Some(id);
    }

    fn unlink<K, V>(&mut self, slab: &mut Slab<K, V>, id: SlotId) {
        let (prev, next) = {
            let s = slab
                .get(id)
                .expect("evict_list.unlink: slot must be occupied");
            (s.lru_prev, s.lru_next)
        };

        match prev {
            Some(p) => {
                slab.get_mut(p)
                    .expect("prev LRU node must be occupied")
                    .lru_next = next;
            }
            None => self.head = next,
        }
        match next {
            Some(n) => {
                slab.get_mut(n)
                    .expect("next LRU node must be occupied")
                    .lru_prev = prev;
            }
            None => self.tail = prev,
        }

        let s = slab
            .get_mut(id)
            .expect("evict_list.unlink: slot must be occupied");
        s.lru_prev = None;
        s.lru_next = None;
    }

    fn promote_to_mru<K, V>(&mut self, slab: &mut Slab<K, V>, id: SlotId) {
        if self.head == Some(id) {
            return;
        }
        self.unlink(slab, id);
        self.push_mru(slab, id);
    }
}

/// `(deadline, slot, generation)` tuple stored in [`ExpiryHeap`]. Ordered by
/// `deadline` so a `BinaryHeap<Reverse<HeapKey>>` yields earliest deadlines
/// first. `slot` and `generation` are tiebreakers for total ordering; only `deadline`
/// matters semantically.
#[derive(Clone, Copy, PartialEq, Eq)]
struct HeapKey {
    deadline: Instant,
    slot: SlotId,
    generation: u32,
}

impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deadline
            .cmp(&other.deadline)
            .then_with(|| self.slot.0.cmp(&other.slot.0))
            .then_with(|| self.generation.cmp(&other.generation))
    }
}

impl PartialOrd for HeapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct ExpiryHeap(BinaryHeap<Reverse<HeapKey>>);

impl ExpiryHeap {
    fn push(&mut self, deadline: Instant, slot: SlotId, generation: u32) {
        self.0.push(Reverse(HeapKey {
            deadline,
            slot,
            generation,
        }));
    }

    fn peek(&self) -> Option<HeapKey> {
        self.0.peek().map(|Reverse(k)| *k)
    }

    fn pop(&mut self) -> Option<HeapKey> {
        self.0.pop().map(|Reverse(k)| k)
    }
}

/// Drain entries whose materialised deadline is at or before `now`. Each
/// validated tuple yields an `(K, V)` pair via [`evict_slot`]; stale tuples
/// (slot recycled / deadline shifted by a read hook / already evicted) are
/// silently dropped. Drains in `O(d log n)` where `d` is the number of
/// past-deadline entries observed during this call.
///
/// `skip_key`, when set, suppresses eviction of the matching entry: the heap
/// tuple is dropped (the caller is about to replace this slot anyway and
/// will push a fresh tuple), but the existing slot is left in place so the
/// caller can read its old value or fire a different lifecycle reason.
fn drain_expired<K, V>(st: &mut State<K, V>, now: Instant, skip_key: Option<&K>) -> Vec<(K, V)>
where
    K: Eq + Hash,
{
    let mut out = Vec::new();
    while let Some(top) = st.expiry.peek() {
        if top.deadline > now {
            break;
        }
        st.expiry.pop();
        let Some(es) = st.slab.get(top.slot) else {
            continue;
        };
        if es.generation != top.generation || es.entry.deadline != Some(top.deadline) {
            continue;
        }
        if let Some(skip) = skip_key
            && &es.key == skip
        {
            continue;
        }
        if let Some((k, v, _)) = evict_slot(st, top.slot) {
            out.push((k, v));
        }
    }
    out
}

/// Result of a single [`Inner::reclaim`] pass. Split per lifecycle reason so
/// the caller fires `TimeExpired` for the deadline drain and `EvictedSize`
/// for the LRU drain that brought weight back under capacity.
struct Reclaimed<K, V> {
    expired: Vec<(K, V)>,
    evicted: Vec<(K, V)>,
}

pub(crate) enum MissStep<V> {
    Hit(V),
    Wait(Arc<MissFlight<V>>),
    Retry,
    Leader { miss: Arc<MissFlight<V>> },
}

/// Returns `None` when the leader was cancelled (sender published
/// `FlightMsg::Cancelled` or was dropped without sending); callers are
/// expected to re-enter the probe loop in that case.
async fn await_miss_settled<V, E>(flight: &MissFlight<V>) -> Option<Result<V, E>>
where
    V: Clone,
    E: Clone + Send + Sync + 'static,
{
    let mut rx = flight.settle.subscribe();
    loop {
        {
            let cur = rx.borrow();
            if let Some(m) = cur.as_ref() {
                return m.clone().into_settled::<E>();
            }
        }
        if rx.changed().await.is_err() {
            // Sender dropped without sending — treat as cancellation.
            return None;
        }
    }
}

/// RAII cleanup for the `try_get_with` leader. On drop (cancel or panic),
/// removes the in-flight `MapSlot::Miss` slot iff it still points at the same
/// flight and broadcasts `FlightMsg::Cancelled` so followers retry instead of
/// blocking forever on a sender that will never publish.
///
/// `Arc::ptr_eq` is the safety hinge: if some other path already replaced the
/// slot (e.g. a prior leader was cancelled and a new leader took over) the
/// guard leaves the new slot alone.
struct LeaderGuard<'a, K: Eq + Hash + 'static, V: 'static> {
    inner: &'a Inner<K, V>,
    key: &'a K,
    flight: Arc<MissFlight<V>>,
    armed: bool,
}

impl<K, V> Drop for LeaderGuard<'_, K, V>
where
    K: Eq + Hash,
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Tolerate poisoning: the leader might already be unwinding from a
        // panic in `init`, and double-panicking from Drop aborts the process.
        let mut st = match self.inner.state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let same = matches!(
            st.entries.get(self.key),
            Some(MapSlot::Miss(arc)) if Arc::ptr_eq(arc, &self.flight)
        );
        if same {
            st.entries.remove(self.key);
        }
        drop(st);
        let _ = self.flight.settle.send(Some(FlightMsg::Cancelled));
    }
}

/// Removes the slab slot at `id` and the matching `entries` map row, drops it
/// from the LRU list, and returns the owned `(K, V)` plus the entry's cached
/// weight (already subtracted from `total_weight`). Returns `None` if the
/// slot was already free (defensive — should not happen under our invariants).
/// Stale tuples in `state.expiry` are not removed here; lazy validation on
/// pop discards them once the slot is recycled.
fn evict_slot<K, V>(st: &mut State<K, V>, id: SlotId) -> Option<(K, V, u64)>
where
    K: Eq + Hash,
{
    let State {
        entries,
        slab,
        evict_list,
        total_weight,
        ..
    } = st;
    evict_list.unlink(slab, id);
    let entry_slot = slab.remove(id)?;
    *total_weight = total_weight.saturating_sub(entry_slot.weight);
    let _ = entries.remove(&entry_slot.key);
    Some((entry_slot.key, entry_slot.entry.value, entry_slot.weight))
}

/// Allocate a slab slot for `key`/`entry`/`weight`, push it to MRU, register
/// it in `state.entries`, bump `total_weight`, and (if applicable) push the
/// deadline into `state.expiry`. Centralises every install site so the heap
/// invariant cannot be silently dropped.
fn install_ready<K, V>(
    st: &mut State<K, V>,
    key: K,
    entry: CacheEntry<V>,
    weight: u64,
) -> SlotId
where
    K: Eq + Hash + Clone,
{
    let State {
        entries,
        slab,
        evict_list,
        expiry,
        total_weight,
        ..
    } = st;
    let idx = slab.next_slot_index();
    let generation = slab.next_generation_for(idx);
    let deadline = entry.deadline;
    let id = slab.insert(EntrySlot {
        key: key.clone(),
        entry,
        weight,
        generation,
        lru_prev: None,
        lru_next: None,
    });
    debug_assert_eq!(
        id.0, idx,
        "slab.insert is expected to use the slot index it advertised"
    );
    evict_list.push_mru(slab, id);
    entries.insert(key, MapSlot::Ready(id));
    *total_weight = total_weight.saturating_add(weight);
    if let Some(dl) = deadline {
        expiry.push(dl, id, generation);
    }
    id
}

impl<K, V> Inner<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    pub(crate) fn new(config: CacheConfig<K, V>) -> Self {
        // Pre-size both the lookup map and the slab to the cache's weight budget so
        // steady-state inserts never trigger a one-shot O(n) rehash. `weight ≥ 1`
        // is the working assumption that bounds entry count by `max_weight`.
        let initial_capacity = capacity_hint_from_max_weight(config.max_weight.get());
        Self {
            config,
            state: Mutex::new(State {
                entries: HashMap::with_capacity(initial_capacity),
                slab: Slab::with_capacity(initial_capacity),
                evict_list: EvictList::default(),
                expiry: ExpiryHeap::default(),
                refresh_inflight: HashSet::new(),
                total_weight: 0,
            }),
        }
    }

    pub(crate) fn get_served(&self, key: &K) -> Option<V>
    where
        K: 'static,
        V: 'static,
    {
        let now = Instant::now();
        let mut st = self.state.lock().expect("cache mutex poisoned");

        let Some(&MapSlot::Ready(id)) = st.entries.get(key) else {
            return None;
        };

        // Fast-path read: borrow the slot to inspect the deadline; if expired,
        // drop the borrow and fall into the eviction path below.
        let expired = st
            .slab
            .get(id)
            .is_some_and(|es| !entry_is_served(&es.entry, now));
        if expired {
            let (_, v, _) = evict_slot(&mut st, id)?;
            drop(st);
            self.notify_lifecycle(LifecycleReason::TimeExpired, key.clone(), Some(v));
            return None;
        }

        let State {
            slab,
            evict_list,
            expiry,
            ..
        } = &mut *st;
        evict_list.promote_to_mru(slab, id);
        let es = slab.get_mut(id)?;
        let value = es.entry.value.clone();
        let generation = es.generation;
        let new_dl = update_read_deadline(key, &mut es.entry, self.config.policy.as_ref(), now);
        if let Some(dl) = new_dl {
            expiry.push(dl, id, generation);
        }
        Some(value)
    }

    pub(crate) fn insert(&self, key: K, value: V) -> Option<V>
    where
        K: 'static,
        V: 'static,
    {
        let now = Instant::now();
        let new_w = self.config.policy.weight(&key, &value).get();
        let key_for_updated = key.clone();
        let mut st = self.state.lock().expect("cache mutex poisoned");

        // Existing value (cloned for return + `Updated` lifecycle) before any
        // structural change. `reclaim` skips `key`, so the existing slot is
        // still present after reclaim returns.
        let prev = match st.entries.get(&key) {
            Some(&MapSlot::Ready(id)) => st.slab.get(id).map(|es| es.entry.value.clone()),
            _ => None,
        };

        let reclaimed = self.reclaim(&mut st, new_w, &key, now);

        if let Some(&MapSlot::Ready(id)) = st.entries.get(&key) {
            let _ = evict_slot(&mut st, id);
        }

        let deadline = deadline_after_write(self.config.policy.as_ref(), &key, &value, now);
        install_ready(
            &mut st,
            key,
            CacheEntry {
                value,
                written_at: now,
                deadline,
            },
            new_w,
        );

        drop(st);
        if let Some(ref v) = prev {
            self.notify_lifecycle(LifecycleReason::Updated, key_for_updated, Some(v.clone()));
        }
        self.fire_reclaim(reclaimed);
        prev
    }

    pub(crate) fn remove(&self, key: &K) -> Option<V>
    where
        K: 'static,
        V: 'static,
    {
        let mut st = self.state.lock().expect("cache mutex poisoned");
        let Some(&MapSlot::Ready(id)) = st.entries.get(key) else {
            return None;
        };
        let (_, v, _) = evict_slot(&mut st, id)?;
        drop(st);
        let v_for_listener = v.clone();
        self.notify_lifecycle(LifecycleReason::Removed, key.clone(), Some(v_for_listener));
        Some(v)
    }

    /// Two-phase capacity reclaim, called before installing a new entry of
    /// weight `add_w` for `skip_key`:
    ///
    /// 1. **Expiry drain** — pop heap tuples with `deadline ≤ now` and evict
    ///    the matching slots. Lazy validation discards stale tuples in O(1).
    ///    Cost: `O(d log n)` for `d` validated victims; `O(s log n)` for
    ///    `s` stale tuples observed.
    /// 2. **Size drain** — while the projected post-install weight exceeds
    ///    `max_weight`, drop the LRU tail. Each victim is O(1): tail lookup,
    ///    `evict_slot`, repeat. The skip-key short-circuit (promote the
    ///    skip slot to MRU when it surfaces at the tail) prevents evicting
    ///    the entry the caller is about to install or replace.
    ///
    /// Returns the two victim lists separately so the caller can fire the
    /// correct lifecycle reason for each phase.
    fn reclaim(
        &self,
        st: &mut State<K, V>,
        add_w: u64,
        skip_key: &K,
        now: Instant,
    ) -> Reclaimed<K, V>
    where
        K: 'static,
        V: 'static,
    {
        let expired = drain_expired(st, now, Some(skip_key));

        let max = self.config.max_weight.get();
        let mut evicted = Vec::new();
        loop {
            if !Self::projected_weight_over_capacity(st, skip_key, add_w, max) {
                break;
            }
            let Some(tail_id) = st.evict_list.tail else {
                break;
            };
            let Some(tail_slot) = st.slab.get(tail_id) else {
                break;
            };
            if &tail_slot.key == skip_key {
                if st.slab.len() <= 1 {
                    break;
                }
                let State {
                    slab, evict_list, ..
                } = &mut *st;
                evict_list.promote_to_mru(slab, tail_id);
                continue;
            }
            let Some((k, v, _)) = evict_slot(st, tail_id) else {
                break;
            };
            evicted.push((k, v));
        }
        Reclaimed { expired, evicted }
    }

    fn fire_reclaim(&self, r: Reclaimed<K, V>)
    where
        K: 'static,
        V: 'static,
    {
        for (k, v) in r.expired {
            self.notify_lifecycle(LifecycleReason::TimeExpired, k, Some(v));
        }
        for (k, v) in r.evicted {
            self.notify_lifecycle(LifecycleReason::EvictedSize, k, Some(v));
        }
    }

    fn projected_weight_over_capacity(
        st: &State<K, V>,
        insert_key: &K,
        new_entry_weight: u64,
        max: u64,
    ) -> bool {
        let projected = match st.entries.get(insert_key) {
            Some(&MapSlot::Ready(id)) => {
                let ow = st.slab.get(id).map_or(0, |es| es.weight);
                st.total_weight
                    .saturating_sub(ow)
                    .saturating_add(new_entry_weight)
            }
            Some(MapSlot::Miss(_)) | None => st.total_weight.saturating_add(new_entry_weight),
        };
        projected > max
    }

    fn notify_lifecycle(&self, reason: LifecycleReason, key: K, value: Option<V>)
    where
        K: 'static,
        V: 'static,
    {
        if let Some(l) = self.config.lifecycle_listener.as_ref() {
            l.on_event(reason, key, value);
        }
    }

    fn finish_miss_ok(
        &self,
        st: &mut State<K, V>,
        key: &K,
        v: &V,
        now: Instant,
    ) -> Reclaimed<K, V>
    where
        K: 'static,
        V: 'static,
    {
        let deadline = deadline_after_write(self.config.policy.as_ref(), key, v, now);
        let nw = self.config.policy.weight(key, v).get();
        let reclaimed = self.reclaim(st, nw, key, now);
        install_ready(
            st,
            key.clone(),
            CacheEntry {
                value: v.clone(),
                written_at: now,
                deadline,
            },
            nw,
        );
        reclaimed
    }

    /// Single critical section for one `try_get_with` iteration: served hit, in-flight miss wait, or
    /// install miss leader. Avoids TOCTOU between separate `get_served` / miss checks / leader insert.
    fn miss_probe(&self, key: &K) -> MissStep<V>
    where
        K: 'static,
        V: 'static,
    {
        let now = Instant::now();
        let mut st = self.state.lock().expect("cache mutex poisoned");

        if let Some(&MapSlot::Ready(id)) = st.entries.get(key) {
            let expired = st
                .slab
                .get(id)
                .is_some_and(|es| !entry_is_served(&es.entry, now));
            if expired
                && let Some((_, v, _)) = evict_slot(&mut st, id)
            {
                drop(st);
                self.notify_lifecycle(LifecycleReason::TimeExpired, key.clone(), Some(v));
                st = self.state.lock().expect("cache mutex poisoned");
            }
        }

        let now = Instant::now();
        if let Some(&MapSlot::Ready(id)) = st.entries.get(key) {
            let still_served = st
                .slab
                .get(id)
                .is_some_and(|es| entry_is_served(&es.entry, now));
            if still_served {
                let State {
                    slab,
                    evict_list,
                    expiry,
                    ..
                } = &mut *st;
                evict_list.promote_to_mru(slab, id);
                let es = slab
                    .get_mut(id)
                    .expect("just confirmed Occupied above under our mutex");
                let value = es.entry.value.clone();
                let generation = es.generation;
                let new_dl =
                    update_read_deadline(key, &mut es.entry, self.config.policy.as_ref(), now);
                if let Some(dl) = new_dl {
                    expiry.push(dl, id, generation);
                }
                return MissStep::Hit(value);
            }
        }

        match st.entries.get(key) {
            Some(MapSlot::Miss(f)) => MissStep::Wait(Arc::clone(f)),
            Some(MapSlot::Ready(_)) => MissStep::Retry,
            None => {
                let (settle_tx, settle_rx) = watch::channel::<Option<FlightMsg<V>>>(None);
                let miss = Arc::new(MissFlight {
                    settle: settle_tx,
                    _hold: settle_rx,
                });
                st.entries
                    .insert(key.clone(), MapSlot::Miss(Arc::clone(&miss)));
                MissStep::Leader {
                    miss: Arc::clone(&miss),
                }
            }
        }
    }

    pub(crate) async fn try_get_with<F, Fut, E>(self: &Arc<Self>, key: K, init: F) -> Result<V, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<V, E>>,
        E: Clone + Send + Sync + 'static,
        K: Eq + Hash + Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let mut init = Some(init);
        loop {
            match self.miss_probe(&key) {
                MissStep::Hit(v) => return Ok(v),
                MissStep::Wait(f) => {
                    if let Some(res) = await_miss_settled(&f).await {
                        return res;
                    }
                    // Leader was cancelled; the cancelled slot has already
                    // been cleared by `LeaderGuard::drop`, so the next probe
                    // either hits a fresh value or installs us as leader.
                }
                MissStep::Retry => {}
                MissStep::Leader { miss } => {
                    let init = init
                        .take()
                        .expect("init consumed: only the leader arm consumes it");
                    let mut guard = LeaderGuard {
                        inner: self.as_ref(),
                        key: &key,
                        flight: miss,
                        armed: true,
                    };
                    let fut = init();
                    let res = fut.await;
                    let mut st = guard.inner.state.lock().expect("cache mutex poisoned");
                    st.entries.remove(&key);
                    let _ = guard
                        .flight
                        .settle
                        .send(Some(FlightMsg::from_result(res.clone())));
                    guard.armed = false;
                    drop(guard);

                    if let Ok(v) = &res {
                        let now = Instant::now();
                        let reclaimed = self.finish_miss_ok(&mut st, &key, v, now);
                        drop(st);
                        self.fire_reclaim(reclaimed);
                        return res;
                    }
                    drop(st);
                    return res;
                }
            }
        }
    }

    pub(crate) fn refresh_inflight_remove(&self, key: &K)
    where
        K: Eq + Hash,
    {
        self.state
            .lock()
            .expect("cache mutex poisoned")
            .refresh_inflight
            .remove(key);
    }

    pub(crate) fn apply_refresh_ok(&self, key: &K, v: V)
    where
        K: 'static,
        V: 'static,
    {
        let mut st = self.state.lock().expect("cache mutex poisoned");
        let now = Instant::now();
        let nw = self.config.policy.weight(key, &v).get();
        let reclaimed = self.reclaim(&mut st, nw, key, now);
        let displaced = match st.entries.get(key) {
            Some(&MapSlot::Ready(id)) => evict_slot(&mut st, id).map(|(_, v, _)| v),
            _ => None,
        };
        let deadline = deadline_after_write(self.config.policy.as_ref(), key, &v, now);
        install_ready(
            &mut st,
            key.clone(),
            CacheEntry {
                value: v,
                written_at: now,
                deadline,
            },
            nw,
        );

        drop(st);
        drop(displaced);
        self.fire_reclaim(reclaimed);
    }
}

fn deadline_after_write<K, V>(
    policy: &(dyn Policy<K, V> + Send + Sync),
    key: &K,
    value: &V,
    now: Instant,
) -> Option<Instant>
where
    K: 'static,
    V: 'static,
{
    let d = policy.expire_after_write(key, value, now);
    if d >= crate::policy::TIME_UNBOUNDED / 2 {
        None
    } else {
        now.checked_add(d)
    }
}

/// Whether the entry may be returned for a read at `now`, using only materialized [`CacheEntry::deadline`].
///
/// [`Policy::expire_after_read`](crate::policy::Policy::expire_after_read) runs only after a live hit
/// in [`update_read_deadline`] to compute the next deadline.
fn entry_is_served<V>(e: &CacheEntry<V>, now: Instant) -> bool {
    !matches!(e.deadline, Some(dl) if now >= dl)
}

/// Pre-sizing hint for `State::entries` and `State::slab`. Bounds the
/// rare-resize tail without allocating absurdly large tables for callers who
/// set huge weight budgets: once the per-entry `Policy::weight` is `≥ 1`,
/// total entry count is bounded by `max_weight`, but we still cap the
/// *allocation* here because the configured budget is a weight unit, not
/// necessarily an entry count.
const CAPACITY_HINT_CAP: u64 = 1 << 20;

fn capacity_hint_from_max_weight(max_weight: u64) -> usize {
    let bounded = max_weight.min(CAPACITY_HINT_CAP);
    usize::try_from(bounded).unwrap_or(usize::MAX)
}

/// Apply the read hook to `e.deadline` in place. Returns `Some(new_deadline)`
/// only when the deadline transitions to a new finite value (different from
/// the old one); the caller is then responsible for pushing it into the
/// expiry heap. Returns `None` when the deadline either stayed the same or
/// became `None`, both of which leave the heap untouched.
fn update_read_deadline<K, V>(
    key: &K,
    e: &mut CacheEntry<V>,
    policy: &(dyn Policy<K, V> + Send + Sync),
    now: Instant,
) -> Option<Instant>
where
    K: 'static,
    V: 'static,
{
    let prev = e.deadline;
    let time_remaining = prev.map_or(crate::policy::TIME_UNBOUNDED, |dl| {
        dl.saturating_duration_since(now)
    });
    let d = policy.expire_after_read(key, &e.value, now, e.written_at, time_remaining);
    let new = if d >= crate::policy::TIME_UNBOUNDED / 2 {
        None
    } else {
        now.checked_add(d)
    };
    e.deadline = new;
    match new {
        Some(dl) if Some(dl) != prev => Some(dl),
        _ => None,
    }
}
