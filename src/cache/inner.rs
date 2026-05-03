use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hashbrown::{HashMap, HashSet};
use tokio::sync::watch;

use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

/// Outcome published on a miss flight's settle channel. `Cancelled` is sent by
/// [`LeaderGuard::drop`] when the leader's `init` future was dropped or
/// panicked before producing a value; followers translate it into a probe
/// retry. Each lane is monomorphic on `(V, E)` so no `Any` downcast is needed
/// to read the result.
#[derive(Clone)]
pub(crate) enum LaneOutcome<V, E> {
    Settled(Result<V, E>),
    Cancelled,
}

/// Per-(key, error type) coalescing point for `try_get_with`. Followers with
/// the same `E` subscribe to the same flight; followers with a different `E`
/// install their own lane and run their own `init`. Mirrors moka's typed miss
/// lanes — see `State::inflight`.
pub(crate) struct MissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    settle: watch::Sender<Option<LaneOutcome<V, E>>>,
}

impl<V, E> MissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn new() -> Self {
        let (tx, _) = watch::channel(None);
        Self { settle: tx }
    }

    fn publish(&self, r: Result<V, E>) {
        let _ = self.settle.send(Some(LaneOutcome::Settled(r)));
    }

    fn cancel(&self) {
        let _ = self.settle.send(Some(LaneOutcome::Cancelled));
    }

    /// `None` when the leader cancelled (sender published `Cancelled`, or
    /// dropped before publishing); callers re-enter the probe loop.
    async fn await_settled(&self) -> Option<Result<V, E>> {
        let mut rx = self.settle.subscribe();
        loop {
            {
                let cur = rx.borrow();
                if let Some(o) = cur.as_ref() {
                    return match o.clone() {
                        LaneOutcome::Settled(r) => Some(r),
                        LaneOutcome::Cancelled => None,
                    };
                }
            }
            if rx.changed().await.is_err() {
                return None;
            }
        }
    }
}

/// One in-flight miss for a key. Lanes are matched by `TypeId<E>`; the
/// `Arc<dyn Any + Send + Sync>` carries the typed `MissFlight<V, E>` for
/// downcast on the follower side, and `cancel_fn` lets a future cancel-all
/// path (e.g. `clear()`) cancel a flight without knowing `E` statically.
///
/// `_v` keeps the `V` parameter at the type level (needed by `downcast::<E>`
/// to reconstruct the typed `MissFlight<V, E>`) without restricting `State`'s
/// auto traits — `fn() -> V` is unconditionally `Send + Sync`.
struct InflightLane<V> {
    type_id: TypeId,
    flight: Arc<dyn Any + Send + Sync>,
    // Wired by `cancel()` once a non-leader cancellation path lands.
    #[allow(dead_code)]
    cancel_fn: fn(&Arc<dyn Any + Send + Sync>),
    _v: PhantomData<fn() -> V>,
}

fn cancel_typed<V, E>(a: &Arc<dyn Any + Send + Sync>)
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    if let Some(f) = (**a).downcast_ref::<MissFlight<V, E>>() {
        f.cancel();
    }
}

impl<V> InflightLane<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn new<E>(flight: Arc<MissFlight<V, E>>) -> Self
    where
        E: Clone + Send + Sync + 'static,
    {
        Self {
            type_id: TypeId::of::<E>(),
            flight: flight as Arc<dyn Any + Send + Sync>,
            cancel_fn: cancel_typed::<V, E>,
            _v: PhantomData,
        }
    }

    fn downcast<E>(&self) -> Option<Arc<MissFlight<V, E>>>
    where
        E: Clone + Send + Sync + 'static,
    {
        if self.type_id != TypeId::of::<E>() {
            return None;
        }
        Arc::clone(&self.flight)
            .downcast::<MissFlight<V, E>>()
            .ok()
    }

    /// Identity-only comparison: two lanes match if they wrap the same
    /// `Arc<MissFlight<_, _>>` allocation. Used by `LeaderGuard` and the
    /// leader's manual cleanup to avoid removing a fresh lane that replaced
    /// the original.
    fn ptr_eq_any(&self, other: &Arc<dyn Any + Send + Sync>) -> bool {
        Arc::as_ptr(&self.flight).cast::<()>() == Arc::as_ptr(other).cast::<()>()
    }

    /// Will be wired by `Cache::clear` (and any other cancel-all paths) once
    /// they land; kept now so downcast and identity comparison live on one
    /// type.
    #[allow(dead_code)]
    fn cancel(&self) {
        (self.cancel_fn)(&self.flight);
    }
}

pub(crate) struct Inner<K: 'static, V: 'static, P: 'static> {
    pub(crate) config: CacheConfig<K, V, P>,
    /// All `Ready` / `Miss` map state. **`std::sync::Mutex` only** — never `tokio::sync::Mutex`; the
    /// lock is released before awaiting user `init` futures in [`Self::try_get_with`].
    pub(crate) state: Mutex<State<K, V>>,
}

/// Stable handle into [`Slab`]. `u32` caps slab population at 2^32-1, which
/// far exceeds any practical cache bounded by RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SlotId(u32);

pub(crate) struct State<K, V> {
    /// Lookup map for `Ready` entries only. In-flight miss leaders live in
    /// [`Self::inflight`] under their own `(K, TypeId<E>)` lane so a
    /// `try_get_with::<…, E1>` call cannot deadlock waiting on a leader that
    /// will publish an unrelated `E2` via type-erased downcast.
    pub(crate) entries: HashMap<K, SlotId>,
    /// Per-key vector of in-flight miss lanes, deduped by the leader's `E`
    /// type. Single lane is the common case; multiple lanes appear only when
    /// concurrent `try_get_with` calls disagree on `E`.
    inflight: HashMap<K, Vec<InflightLane<V>>>,
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

    fn clear(&mut self) {
        self.0.clear();
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

pub(crate) enum MissStep<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    Hit(V),
    Wait(Arc<MissFlight<V, E>>),
    Leader { miss: Arc<MissFlight<V, E>> },
}

/// Identity-only lane removal. Compares Arc pointer addresses; ignores any
/// fresh lane installed for the same key by an unrelated leader (cancelled
/// then re-entered). If the per-key vec becomes empty the entry is dropped to
/// keep `inflight` from accumulating dead keys.
fn remove_lane<K, V>(
    st: &mut State<K, V>,
    key: &K,
    flight: &Arc<dyn Any + Send + Sync>,
) where
    K: Eq + Hash,
    V: Clone + Send + Sync + 'static,
{
    if let Some(lanes) = st.inflight.get_mut(key) {
        lanes.retain(|lane| !lane.ptr_eq_any(flight));
        if lanes.is_empty() {
            st.inflight.remove(key);
        }
    }
}

/// RAII cleanup for the `try_get_with` leader. On drop (cancel or panic),
/// removes the lane (identified by Arc identity, so we won't disturb a
/// follower-installed replacement) and broadcasts `LaneOutcome::Cancelled` so
/// any follower waiting on this exact flight retries instead of blocking
/// forever on a sender that will never publish.
struct LeaderGuard<'a, K, V, E, P>
where
    K: Eq + Hash + 'static,
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
    P: 'static,
{
    inner: &'a Inner<K, V, P>,
    key: &'a K,
    flight: Arc<MissFlight<V, E>>,
    /// Type-erased view of `flight`, kept alongside so identity comparison
    /// against stored lanes does not need to know `E` at the call site.
    flight_any: Arc<dyn Any + Send + Sync>,
    armed: bool,
}

impl<K, V, E, P> Drop for LeaderGuard<'_, K, V, E, P>
where
    K: Eq + Hash + 'static,
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
    P: 'static,
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
        remove_lane(&mut st, self.key, &self.flight_any);
        drop(st);
        self.flight.cancel();
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
    entries.insert(key, id);
    *total_weight = total_weight.saturating_add(weight);
    if let Some(dl) = deadline {
        expiry.push(dl, id, generation);
    }
    id
}

impl<K, V, P> Inner<K, V, P>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
    P: Policy<K, V>,
{
    pub(crate) fn new(config: CacheConfig<K, V, P>) -> Self {
        // Pre-size both the lookup map and the slab to the cache's weight budget so
        // steady-state inserts never trigger a one-shot O(n) rehash. `weight ≥ 1`
        // is the working assumption that bounds entry count by `max_weight`.
        let initial_capacity = capacity_hint_from_max_weight(config.max_weight.get());
        Self {
            config,
            state: Mutex::new(State {
                entries: HashMap::with_capacity(initial_capacity),
                inflight: HashMap::new(),
                slab: Slab::with_capacity(initial_capacity),
                evict_list: EvictList::default(),
                expiry: ExpiryHeap::default(),
                refresh_inflight: HashSet::new(),
                total_weight: 0,
            }),
        }
    }

    pub(crate) fn get_served<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
        V: 'static,
    {
        let now = Instant::now();
        let mut st = self.state.lock().expect("cache mutex poisoned");

        let &id = st.entries.get(key)?;

        // Fast-path read: borrow the slot to inspect the deadline; if expired,
        // drop the borrow and fall into the eviction path below.
        let expired = st
            .slab
            .get(id)
            .is_some_and(|es| !entry_is_served(&es.entry, now));
        if expired {
            // Use the slot's owned key for the lifecycle event so callers
            // querying with a borrowed `Q` need not own a `K`.
            let (k, v, _) = evict_slot(&mut st, id)?;
            drop(st);
            self.notify_lifecycle(LifecycleReason::TimeExpired, k, Some(v));
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
        // Disjoint-field borrow: `&es.key` (immutable) and `&mut es.entry`
        // (mutable) don't overlap, so the borrow checker accepts both.
        let new_dl =
            update_read_deadline(&es.key, &mut es.entry, &self.config.policy, now);
        if let Some(dl) = new_dl {
            expiry.push(dl, id, generation);
        }
        Some(value)
    }

    pub(crate) fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let st = self.state.lock().expect("cache mutex poisoned");
        let Some(&id) = st.entries.get(key) else {
            return false;
        };
        // Read-only check: do not evict expired entries here; let the next
        // `get`/`insert` reclaim path do it so contains_key has no side
        // effects beyond taking the lock.
        let now = Instant::now();
        st.slab
            .get(id)
            .is_some_and(|es| entry_is_served(&es.entry, now))
    }

    pub(crate) fn len(&self) -> usize {
        self.state
            .lock()
            .expect("cache mutex poisoned")
            .slab
            .len()
    }

    pub(crate) fn weight(&self) -> u64 {
        self.state
            .lock()
            .expect("cache mutex poisoned")
            .total_weight
    }

    /// Drains every `Ready` slot, fires `Removed` for each one outside the
    /// lock, and cancels every in-flight miss lane so any waiter sees a
    /// retry signal rather than blocking forever on an abandoned flight.
    pub(crate) fn clear(&self)
    where
        K: 'static,
        V: 'static,
    {
        let mut st = self.state.lock().expect("cache mutex poisoned");
        let State {
            entries,
            inflight,
            slab,
            evict_list,
            expiry,
            total_weight,
            ..
        } = &mut *st;
        let mut drained: Vec<(K, V)> = Vec::with_capacity(slab.len());
        while let Some(tail) = evict_list.tail {
            evict_list.unlink(slab, tail);
            if let Some(entry_slot) = slab.remove(tail) {
                drained.push((entry_slot.key, entry_slot.entry.value));
            }
        }
        entries.clear();
        expiry.clear();
        *total_weight = 0;

        // Cancel pending miss lanes before releasing the lock so followers
        // see Cancelled instead of waiting on a sender we are about to drop.
        let cancelled: Vec<Vec<InflightLane<V>>> = inflight.drain().map(|(_, v)| v).collect();
        drop(st);

        for lanes in &cancelled {
            for lane in lanes {
                lane.cancel();
            }
        }

        for (k, v) in drained {
            self.notify_lifecycle(LifecycleReason::Removed, k, Some(v));
        }
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
            Some(&id) => st.slab.get(id).map(|es| es.entry.value.clone()),
            None => None,
        };

        let reclaimed = self.reclaim(&mut st, new_w, &key, now);

        if let Some(&id) = st.entries.get(&key) {
            let _ = evict_slot(&mut st, id);
        }

        let deadline = deadline_after_write(&self.config.policy, &key, &value, now);
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

    pub(crate) fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
        V: 'static,
    {
        let mut st = self.state.lock().expect("cache mutex poisoned");
        let &id = st.entries.get(key)?;
        let (k, v, _) = evict_slot(&mut st, id)?;
        drop(st);
        let v_for_listener = v.clone();
        self.notify_lifecycle(LifecycleReason::Removed, k, Some(v_for_listener));
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

        // Hoist the existing-slot weight lookup out of the loop: the
        // skip_key short-circuit means its slot is never reclaimed in this
        // pass, so its weight is invariant across iterations. Only
        // `state.total_weight` changes per iteration; the projection then
        // becomes a couple of arithmetic ops instead of a HashMap probe.
        let displaced_weight = st
            .entries
            .get(skip_key)
            .and_then(|&id| st.slab.get(id))
            .map_or(0, |es| es.weight);

        let max = self.config.max_weight.get();
        let mut evicted = Vec::new();
        loop {
            if !Self::projected_weight_over_capacity(st, displaced_weight, add_w, max) {
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
        displaced_weight: u64,
        new_entry_weight: u64,
        max: u64,
    ) -> bool {
        let projected = st
            .total_weight
            .saturating_sub(displaced_weight)
            .saturating_add(new_entry_weight);
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
        let deadline = deadline_after_write(&self.config.policy, key, v, now);
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
    fn miss_probe<E>(&self, key: &K) -> MissStep<V, E>
    where
        K: 'static,
        V: 'static,
        E: Clone + Send + Sync + 'static,
    {
        let now = Instant::now();
        let mut st = self.state.lock().expect("cache mutex poisoned");

        if let Some(&id) = st.entries.get(key) {
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
        if let Some(&id) = st.entries.get(key) {
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
                    update_read_deadline(key, &mut es.entry, &self.config.policy, now);
                if let Some(dl) = new_dl {
                    expiry.push(dl, id, generation);
                }
                return MissStep::Hit(value);
            }
        }

        // No live Ready. Look for an in-flight lane with the same E and join
        // it as a follower; otherwise install a fresh lane and become leader.
        if let Some(lanes) = st.inflight.get(key)
            && let Some(lane) = lanes.iter().find_map(InflightLane::downcast::<E>)
        {
            return MissStep::Wait(lane);
        }

        let miss = Arc::new(MissFlight::<V, E>::new());
        st.inflight
            .entry(key.clone())
            .or_default()
            .push(InflightLane::new(Arc::clone(&miss)));
        MissStep::Leader { miss }
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
            match self.miss_probe::<E>(&key) {
                MissStep::Hit(v) => return Ok(v),
                MissStep::Wait(f) => {
                    if let Some(res) = f.await_settled().await {
                        return res;
                    }
                    // Leader cancelled — its lane has been cleared. Loop and
                    // either hit a fresh value or take leadership ourselves.
                }
                MissStep::Leader { miss } => {
                    let init = init
                        .take()
                        .expect("init consumed: only the leader arm consumes it");
                    let flight_any: Arc<dyn Any + Send + Sync> =
                        Arc::clone(&miss) as Arc<dyn Any + Send + Sync>;
                    let mut guard = LeaderGuard {
                        inner: self.as_ref(),
                        key: &key,
                        flight: miss,
                        flight_any,
                        armed: true,
                    };
                    let fut = init();
                    let res = fut.await;
                    let mut st = guard.inner.state.lock().expect("cache mutex poisoned");
                    remove_lane(&mut st, &key, &guard.flight_any);

                    let reclaimed = if let Ok(v) = &res {
                        let now = Instant::now();
                        Some(self.finish_miss_ok(&mut st, &key, v, now))
                    } else {
                        None
                    };

                    // Publish under the lock so a follower that wakes and
                    // immediately calls `get(key)` cannot observe a state
                    // where the value was reported settled but is not yet
                    // installed in `entries`.
                    guard.flight.publish(res.clone());
                    guard.armed = false;
                    drop(guard);
                    drop(st);

                    if let Some(rec) = reclaimed {
                        self.fire_reclaim(rec);
                    }
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
            Some(&id) => evict_slot(&mut st, id).map(|(_, v, _)| v),
            None => None,
        };
        let deadline = deadline_after_write(&self.config.policy, key, &v, now);
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
        // Mirror `Inner::insert`: when an existing Ready value is replaced,
        // emit Updated outside the lock so listeners see the same lifecycle
        // for refresh-driven and caller-driven replacements.
        if let Some(prev) = displaced {
            self.notify_lifecycle(LifecycleReason::Updated, key.clone(), Some(prev));
        }
        self.fire_reclaim(reclaimed);
    }
}

fn deadline_after_write<K, V, P>(policy: &P, key: &K, value: &V, now: Instant) -> Option<Instant>
where
    K: 'static,
    V: 'static,
    P: Policy<K, V> + ?Sized,
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
fn update_read_deadline<K, V, P>(
    key: &K,
    e: &mut CacheEntry<V>,
    policy: &P,
    now: Instant,
) -> Option<Instant>
where
    K: 'static,
    V: 'static,
    P: Policy<K, V> + ?Sized,
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
