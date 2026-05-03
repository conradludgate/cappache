use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::watch;

use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

#[derive(Clone)]
pub(crate) enum FlightMsg<V> {
    Ok(V),
    Err(Arc<dyn Any + Send + Sync>),
}

impl<V: Clone> FlightMsg<V> {
    pub(crate) fn from_result<E: Send + Sync + 'static>(r: Result<V, E>) -> Self {
        match r {
            Ok(v) => FlightMsg::Ok(v),
            Err(e) => FlightMsg::Err(Arc::new(e)),
        }
    }

    pub(crate) fn into_result<E: Clone + Send + Sync + 'static>(self) -> Result<V, E> {
        match self {
            FlightMsg::Ok(v) => Ok(v),
            FlightMsg::Err(arc) => Err(Arc::downcast::<E>(arc)
                .expect("miss flight error type mismatch: E must match the stored miss error type")
                .as_ref()
                .clone()),
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
    lru_prev: Option<SlotId>,
    lru_next: Option<SlotId>,
}

enum SlabEntry<K, V> {
    Occupied(EntrySlot<K, V>),
    Free { next_free: Option<u32> },
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

    fn insert(&mut self, entry: EntrySlot<K, V>) -> SlotId {
        if let Some(idx) = self.free_head {
            let slot = &mut self.slots[idx as usize];
            let next_free = match slot {
                SlabEntry::Free { next_free } => *next_free,
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
            },
        );
        let SlabEntry::Occupied(entry) = taken else {
            unreachable!("checked Occupied above")
        };
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

pub(crate) enum MissStep<V> {
    Hit(V),
    Wait(Arc<MissFlight<V>>),
    Retry,
    Leader { miss: Arc<MissFlight<V>> },
}

async fn await_miss_settled<V, E>(flight: &MissFlight<V>) -> Result<V, E>
where
    V: Clone,
    E: Clone + Send + Sync + 'static,
{
    let mut rx = flight.settle.subscribe();
    loop {
        {
            let cur = rx.borrow();
            if let Some(m) = cur.as_ref() {
                return m.clone().into_result::<E>();
            }
        }
        rx.changed().await.expect(
            "miss settle watch: sender dropped before flight completed; this is a bug in clache",
        );
    }
}

/// Removes the slab slot at `id` and the matching `entries` map row, drops it
/// from the LRU list, and returns the owned `(K, V)` plus the entry's cached
/// weight (already subtracted from `total_weight`). Returns `None` if the
/// slot was already free (defensive — should not happen under our invariants).
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
            slab, evict_list, ..
        } = &mut *st;
        evict_list.promote_to_mru(slab, id);
        let es = slab.get_mut(id)?;
        update_read_deadline(key, &mut es.entry, self.config.policy.as_ref(), now);
        Some(es.entry.value.clone())
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
        // structural change. `reclaim_size` skips `key`, so the existing slot is
        // still present after reclaim.
        let prev = match st.entries.get(&key) {
            Some(&MapSlot::Ready(id)) => st.slab.get(id).map(|es| es.entry.value.clone()),
            _ => None,
        };

        let pending = self.reclaim_size(&mut st, new_w, &key);

        if let Some(&MapSlot::Ready(id)) = st.entries.get(&key) {
            // Drop the previous slot for this key; we install a fresh one below.
            let _ = evict_slot(&mut st, id);
        }

        let deadline = deadline_after_write(self.config.policy.as_ref(), &key, &value, now);
        let entry_slot = EntrySlot {
            key: key.clone(),
            entry: CacheEntry {
                value,
                written_at: now,
                deadline,
            },
            weight: new_w,
            lru_prev: None,
            lru_next: None,
        };
        let State {
            entries,
            slab,
            evict_list,
            total_weight,
            ..
        } = &mut *st;
        let id = slab.insert(entry_slot);
        evict_list.push_mru(slab, id);
        entries.insert(key, MapSlot::Ready(id));
        *total_weight = total_weight.saturating_add(new_w);

        drop(st);
        if let Some(ref v) = prev {
            self.notify_lifecycle(LifecycleReason::Updated, key_for_updated, Some(v.clone()));
        }
        for (ek, ev) in pending {
            self.notify_lifecycle(LifecycleReason::EvictedSize, ek, Some(ev));
        }
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

    /// Drain LRU-tail victims until the projected weight (current
    /// `total_weight`, minus any existing entry for `skip_key`, plus
    /// `add_weight`) is within `max_weight`. Each drained slot is **O(1)**:
    /// the LRU tail gives the next victim, eviction unlinks + frees + drops
    /// from the map. The skip-key short-circuit prevents evicting the entry
    /// the caller is about to install/replace.
    fn reclaim_size(&self, st: &mut State<K, V>, add_w: u64, skip_key: &K) -> Vec<(K, V)>
    where
        K: 'static,
        V: 'static,
    {
        let max = self.config.max_weight.get();
        let mut pending = Vec::new();
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
            pending.push((k, v));
        }
        pending
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

    fn finish_miss_ok(&self, st: &mut State<K, V>, key: &K, v: &V, now: Instant) -> Vec<(K, V)>
    where
        K: 'static,
        V: 'static,
    {
        let deadline = deadline_after_write(self.config.policy.as_ref(), key, v, now);
        let nw = self.config.policy.weight(key, v).get();
        let pending = self.reclaim_size(st, nw, key);
        let entry_slot = EntrySlot {
            key: key.clone(),
            entry: CacheEntry {
                value: v.clone(),
                written_at: now,
                deadline,
            },
            weight: nw,
            lru_prev: None,
            lru_next: None,
        };
        let State {
            entries,
            slab,
            evict_list,
            total_weight,
            ..
        } = st;
        let id = slab.insert(entry_slot);
        evict_list.push_mru(slab, id);
        entries.insert(key.clone(), MapSlot::Ready(id));
        *total_weight = total_weight.saturating_add(nw);
        pending
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
                    slab, evict_list, ..
                } = &mut *st;
                evict_list.promote_to_mru(slab, id);
                let es = slab
                    .get_mut(id)
                    .expect("just confirmed Occupied above under our mutex");
                update_read_deadline(key, &mut es.entry, self.config.policy.as_ref(), now);
                return MissStep::Hit(es.entry.value.clone());
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
        loop {
            match self.miss_probe(&key) {
                MissStep::Hit(v) => return Ok(v),
                MissStep::Wait(f) => return await_miss_settled(&f).await,
                MissStep::Retry => {}
                MissStep::Leader { miss } => {
                    let fut = init();
                    let res = fut.await;
                    let mut st = self.state.lock().expect("cache mutex poisoned");
                    st.entries.remove(&key);
                    let _ = miss.settle.send(Some(FlightMsg::from_result(res.clone())));

                    if let Ok(v) = &res {
                        let now = Instant::now();
                        let pending = self.finish_miss_ok(&mut st, &key, v, now);
                        drop(st);
                        for (ek, ev) in pending {
                            self.notify_lifecycle(LifecycleReason::EvictedSize, ek, Some(ev));
                        }
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
        let displaced = match st.entries.get(key) {
            Some(&MapSlot::Ready(id)) => evict_slot(&mut st, id).map(|(_, v, _)| v),
            _ => None,
        };
        let now = Instant::now();
        let deadline = deadline_after_write(self.config.policy.as_ref(), key, &v, now);
        let nw = self.config.policy.weight(key, &v).get();
        let pending = self.reclaim_size(&mut st, nw, key);

        let entry_slot = EntrySlot {
            key: key.clone(),
            entry: CacheEntry {
                value: v,
                written_at: now,
                deadline,
            },
            weight: nw,
            lru_prev: None,
            lru_next: None,
        };
        let State {
            entries,
            slab,
            evict_list,
            total_weight,
            ..
        } = &mut *st;
        let id = slab.insert(entry_slot);
        evict_list.push_mru(slab, id);
        entries.insert(key.clone(), MapSlot::Ready(id));
        *total_weight = total_weight.saturating_add(nw);

        drop(st);
        drop(displaced);
        for (ek, ev) in pending {
            self.notify_lifecycle(LifecycleReason::EvictedSize, ek, Some(ev));
        }
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

fn update_read_deadline<K, V>(
    key: &K,
    e: &mut CacheEntry<V>,
    policy: &(dyn Policy<K, V> + Send + Sync),
    now: Instant,
) where
    K: 'static,
    V: 'static,
{
    let time_remaining = e.deadline.map_or(crate::policy::TIME_UNBOUNDED, |dl| {
        dl.saturating_duration_since(now)
    });
    let d = policy.expire_after_read(key, &e.value, now, e.written_at, time_remaining);
    if d >= crate::policy::TIME_UNBOUNDED / 2 {
        e.deadline = None;
    } else {
        e.deadline = now.checked_add(d);
    }
}
