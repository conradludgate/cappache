use std::any::TypeId;
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
use std::time::Instant;

use clashcore::ClashCollection;
use clashcore::lock::RwLock;
use hashbrown::{DefaultHashBuilder, HashMap, HashTable};

use super::clock;
use super::expiry::{ExpiryHeap, expiry_push_extend};
use super::flight::InflightLane;
use super::reclaim::{Reclaimed, reclaim};
use super::slab::{CacheEntry, EntrySlab, EntrySlot, Fifo, GhostSet, Queue, SlotKey};
use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

pub(super) type Shard<K, V> = clashcore::CachePadded<RwLock<State<K, V>>>;

pub(crate) enum RefreshGate {
    Submit,
    SkipInFlight,
    AbsentKey,
}

pub(crate) struct Inner<K: 'static, V: 'static, P: 'static, S: 'static = DefaultHashBuilder> {
    pub(crate) config: CacheConfig<K, V, P, S>,
    /// Sharded per-key state. [`ClashCollection`] does the
    /// `(h << 7) >> shift` indexing — left-shift parks the top 7 bits
    /// for hashbrown's SIMD tag, right-shift picks the shard. Releasing
    /// the shard lock before awaiting user `init` futures is a hard
    /// requirement (see [`super::miss`]).
    pub(crate) shards: ClashCollection<State<K, V>>,
    /// Used both for shard selection and per-shard table lookup —
    /// stored once at `Inner` level so a key's bucket stays stable
    /// across clones and cross-shard iteration.
    pub(crate) hasher: S,
    /// Per-shard weight cap: `ceil(max_weight / shard_count)`. Total
    /// observed weight may overshoot `max_weight` by up to
    /// `shard_count - 1` units of rounding slack.
    pub(crate) shard_max_weight: u64,
    /// Monotonic anchor for `super::clock` ticks.
    pub(crate) start: Instant,
}

/// `small_max_weight = shard_max_weight / SMALL_DENOMINATOR`. Yang et
/// al. (SOSP'23) §4 default to S = 10%; we use ~3% (matching
/// `quick_cache`'s `hot_allocation = 0.97`) which won across nearly
/// every operating point in our mokabench validation pass.
pub(super) const SMALL_DENOMINATOR: u64 = 33;

pub(super) const FREQ_CAP: u8 = 3;

/// Lookup keyed by `K` against the [`SlotKey`]-only `entries` table.
/// Equality goes through the slab, so a recycled slot (slotmap version
/// mismatch) self-cleans here as `slab.get` returning `None`.
pub(super) fn entries_find<K, V, Q, H>(
    entries: &HashTable<SlotKey>,
    hasher: &H,
    slab: &EntrySlab<K, V>,
    key: &Q,
) -> Option<SlotKey>
where
    K: Hash + Eq + Borrow<Q>,
    Q: Hash + Eq + ?Sized,
    H: BuildHasher,
{
    let h = hasher.hash_one(key);
    entries
        .find(h, |&id| {
            slab.get(id).is_some_and(|es| es.key.borrow() == key)
        })
        .copied()
}

/// Caller must have pushed `id` into the slab already; the resize
/// closure rehashes against the live slab.
fn entries_insert<K, V, H>(
    entries: &mut HashTable<SlotKey>,
    hasher: &H,
    slab: &EntrySlab<K, V>,
    id: SlotKey,
    key_hash: u64,
) where
    K: Hash + Eq,
    H: BuildHasher,
{
    entries.insert_unique(key_hash, id, |&other| {
        hasher.hash_one(
            &slab
                .get(other)
                .expect("entries refer only to occupied slots")
                .key,
        )
    });
}

pub(crate) struct State<K, V> {
    /// Key-free lookup table; equality goes through the slab so we
    /// don't double-store `K` (one copy lives on `EntrySlot::key`).
    pub(crate) entries: HashTable<SlotKey>,
    /// In-flight miss lanes keyed by `(K, TypeId<F>)`. Distinct `E`
    /// (or sync vs async) install separate rows so a leader cannot
    /// deadlock followers expecting a different result type. Borrowed
    /// lookups use `LaneKeyRef` to avoid cloning the key.
    pub(super) inflight: HashMap<(K, TypeId), InflightLane>,
    pub(super) slab: EntrySlab<K, V>,
    pub(super) small: Fifo,
    pub(super) main: Fifo,
    /// Fingerprints of slots recently evicted from Small with
    /// `freq ≤ 1`. The install path consumes membership to route a
    /// returning key straight to Main.
    pub(super) ghost: GhostSet,
    /// Cached `sum(weight)` over Small. Maintained eagerly on
    /// install/evict so reclaim's drain decision is O(1).
    pub(super) small_weight: u64,
    /// Min-heap of `(deadline, slot)` tuples; lazily validated on pop
    /// (slotmap version + deadline equality).
    pub(super) expiry: ExpiryHeap,
    /// Heap pushes that left a stale tuple behind. Triggers a rebuild
    /// when it exceeds `slab.len()` so memory stays bounded under
    /// read-extending policies.
    pub(super) stale_pushes: u64,
    pub(crate) total_weight: u64,
}

impl<K, V> State<K, V> {
    pub(super) fn queue_unlink(&mut self, queue: Queue, id: SlotKey) {
        match queue {
            Queue::Small => self.small.unlink(&mut self.slab, id),
            Queue::Main => self.main.unlink(&mut self.slab, id),
        }
    }

    pub(super) fn queue_push_head(&mut self, queue: Queue, id: SlotKey) {
        match queue {
            Queue::Small => self.small.push_head(&mut self.slab, id),
            Queue::Main => self.main.push_head(&mut self.slab, id),
        }
    }

    pub(super) fn queue_promote_to_head(&mut self, queue: Queue, id: SlotKey) {
        match queue {
            Queue::Small => self.small.promote_to_head(&mut self.slab, id),
            Queue::Main => self.main.promote_to_head(&mut self.slab, id),
        }
    }

    pub(super) fn queue_pop_tail(&mut self, queue: Queue) -> Option<SlotKey> {
        match queue {
            Queue::Small => self.small.pop_tail(&mut self.slab),
            Queue::Main => self.main.pop_tail(&mut self.slab),
        }
    }
}

/// Removes `id` from the slab, its FIFO, and `entries`; debits weight.
/// Returns the owned `(K, V, weight)`. Stale heap tuples and stale
/// `entries` rows for this slot are left to self-clean via slotmap
/// version mismatch (see [`entries_find`]).
pub(super) fn evict_slot<K, V, H>(
    st: &mut State<K, V>,
    hasher: &H,
    id: SlotKey,
) -> Option<(K, V, u64)>
where
    K: Eq + Hash,
    H: BuildHasher,
{
    let queue = st.slab.get(id)?.queue;
    st.queue_unlink(queue, id);
    // Take the slot out first; the `entries` probe predicate uses
    // `slab.get`, so removing the slab entry up front means we can
    // match the table row by `SlotKey` directly.
    let entry_slot = st.slab.remove(id)?;
    st.total_weight = st.total_weight.saturating_sub(entry_slot.weight);
    if matches!(queue, Queue::Small) {
        st.small_weight = st.small_weight.saturating_sub(entry_slot.weight);
    }
    let h = hasher.hash_one(&entry_slot.key);
    if let Ok(occ) = st.entries.find_entry(h, |&other| other == id) {
        occ.remove();
    }
    Some((entry_slot.key, entry_slot.entry.value, entry_slot.weight))
}

/// Slab + queue + `entries` + `expiry` + weight: the single install
/// site for `Ready` slots. Keeps the heap and ghost invariants in
/// lockstep with the slab.
pub(super) fn install_ready<K, V, H>(
    st: &mut State<K, V>,
    hasher: &H,
    key: K,
    entry: CacheEntry<V>,
    weight: u64,
) -> SlotKey
where
    K: Eq + Hash,
    H: BuildHasher,
{
    let deadline = entry.deadline;
    let key_hash = hasher.hash_one(&key);
    // Consume-on-hit: a present fingerprint routes the entry to Main.
    let to_main = st.ghost.remove(key_hash);
    let queue = if to_main { Queue::Main } else { Queue::Small };
    let id = st.slab.insert(EntrySlot {
        key,
        entry,
        weight,
        queue,
        freq: 0,
        fifo_prev: None,
        fifo_next: None,
        refresh_inflight: false,
    });
    st.queue_push_head(queue, id);
    if !to_main {
        st.small_weight = st.small_weight.saturating_add(weight);
    }
    entries_insert(&mut st.entries, hasher, &st.slab, id, key_hash);
    st.total_weight = st.total_weight.saturating_add(weight);
    if deadline != crate::cache::clock::NO_DEADLINE {
        st.expiry.push(deadline, id);
    }
    id
}

impl<K, V, P, S> Inner<K, V, P, S>
where
    S: BuildHasher,
{
    pub(super) fn shard_for<Q>(&self, key: &Q) -> &Shard<K, V>
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        // 32-bit truncation is fine: the default hasher distributes
        // well enough that dropping high bits doesn't bias selection.
        #[allow(clippy::cast_possible_truncation)]
        let h = self.hasher.hash_one(key) as usize;
        let idx = self.shards.determine_shard(h);
        &self.shards.shards()[idx]
    }

    /// Best-effort clear of `refresh_inflight`. On the bound-free impl
    /// so `RefreshGuard::drop` can call it without `Policy<K, V>`.
    pub(crate) fn refresh_inflight_remove(&self, key: &K)
    where
        K: Eq + Hash,
    {
        let mut st = self.shard_for(key).write();
        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key)
            && let Some(es) = st.slab.get_mut(id)
        {
            es.refresh_inflight = false;
        }
    }

    pub(super) fn fire_reclaim(&self, r: Reclaimed<K, V>)
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

    pub(super) fn notify_lifecycle(&self, reason: LifecycleReason, key: K, value: Option<V>)
    where
        K: 'static,
        V: 'static,
    {
        if let Some(l) = self.config.lifecycle_listener.as_ref() {
            l.on_event(reason, key, value);
        }
    }
}

impl<K, V, P, S> Inner<K, V, P, S>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
    P: Policy<K, V>,
    S: BuildHasher + Clone,
{
    pub(crate) fn new(config: CacheConfig<K, V, P, S>) -> Self {
        // `ClashCollection::with_shard_amount` requires a pow2 ≥ 2;
        // `CacheBuilder::shards` enforces that up front.
        let shard_count = config.shards;
        let max_weight = config.max_weight.get();
        let shard_max_weight = max_weight.div_ceil(shard_count as u64).max(1);
        // Don't pre-size from max_weight: `Policy::weight != 1`
        // over-allocates immediately. Eat a few rehashes instead.
        let shards = ClashCollection::with_shard_amount(shard_count, || State {
            entries: HashTable::new(),
            inflight: HashMap::new(),
            slab: EntrySlab::with_key(),
            small: Fifo::default(),
            main: Fifo::default(),
            ghost: GhostSet::new(DEFAULT_GHOST_CAP_PER_SHARD),
            small_weight: 0,
            expiry: ExpiryHeap::default(),
            stale_pushes: 0,
            total_weight: 0,
        });
        let hasher = config.hasher.clone();
        Self {
            config,
            shards,
            hasher,
            shard_max_weight,
            start: Instant::now(),
        }
    }

    pub(crate) fn get_served<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
        V: 'static,
    {
        let now = Instant::now();
        let now_tick = clock::floor_tick(self.start, now);
        let mut st = self.shard_for(key).write();

        let id = entries_find(&st.entries, &self.hasher, &st.slab, key)?;

        let expired = st
            .slab
            .get(id)
            .is_some_and(|es| !entry_is_served(&es.entry, now_tick));
        if expired {
            // Use the slot's owned key so a `Q`-keyed caller doesn't
            // have to own a `K` for the lifecycle event.
            let (k, v, _) = evict_slot(&mut st, &self.hasher, id)?;
            drop(st);
            self.notify_lifecycle(LifecycleReason::TimeExpired, k, Some(v));
            return None;
        }

        let (value, update) = {
            let es = st.slab.get_mut(id)?;
            es.freq = es.freq.saturating_add(1).min(FREQ_CAP);
            let value = es.entry.value.clone();
            // Disjoint-field borrow: `&es.key` and `&mut es.entry`.
            let update =
                update_read_deadline(&es.key, &mut es.entry, &self.config.policy, self.start, now);
            (value, update)
        };
        if let Some(dl) = update.new_deadline {
            expiry_push_extend(&mut st, dl, id, update.superseded_finite);
        }
        Some(value)
    }

    pub(crate) fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let st = self.shard_for(key).read();
        let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) else {
            return false;
        };
        // No eviction here: keep `contains_key` side-effect-free; the
        // next `get`/`insert` will reclaim.
        let now_tick = clock::floor_tick(self.start, Instant::now());
        st.slab
            .get(id)
            .is_some_and(|es| entry_is_served(&es.entry, now_tick))
    }

    // Per-shard locks taken in turn; not snapshot-consistent but each
    // shard's contribution is taken atomically.
    pub(crate) fn len(&self) -> usize {
        self.shards
            .shards()
            .iter()
            .map(|s| s.read().slab.len())
            .sum()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.shards
            .shards()
            .iter()
            .all(|s| s.read().slab.len() == 0)
    }

    pub(crate) fn weight(&self) -> u64 {
        self.shards
            .shards()
            .iter()
            .map(|s| s.read().total_weight)
            .sum()
    }

    pub(crate) fn clear(&self)
    where
        K: 'static,
        V: 'static,
    {
        for shard in self.shards.shards() {
            let mut st = shard.write();
            let mut drained: Vec<(K, V)> = Vec::with_capacity(st.slab.len());
            // Main then Small, tail-to-head, for a stable cross-run
            // `Removed` sequence (S3-FIFO has no global recency axis).
            for queue in [Queue::Main, Queue::Small] {
                while let Some(tail) = st.queue_pop_tail(queue) {
                    if let Some(entry_slot) = st.slab.remove(tail) {
                        drained.push((entry_slot.key, entry_slot.entry.value));
                    }
                }
            }
            st.ghost.clear();
            st.entries.clear();
            st.expiry.clear();
            st.stale_pushes = 0;
            st.total_weight = 0;
            st.small_weight = 0;

            // Cancel before releasing the lock so followers see
            // Cancelled instead of blocking on a sender we just dropped.
            let cancelled: Vec<InflightLane> = st.inflight.drain().map(|(_, v)| v).collect();
            drop(st);

            for lane in &cancelled {
                lane.cancel();
            }

            for (k, v) in drained {
                self.notify_lifecycle(LifecycleReason::Removed, k, Some(v));
            }
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
        let mut st = self.shard_for(&key).write();

        // Cache `existing_id` once: derive both `prev` and the displaced
        // eviction from it instead of probing twice.
        let existing_id = entries_find(&st.entries, &self.hasher, &st.slab, &key);
        let prev = existing_id.and_then(|id| st.slab.get(id).map(|es| es.entry.value.clone()));

        let reclaimed = reclaim(self, &mut st, new_w, &key, now);

        // `reclaim` skips `skip_key`, so `existing_id` is still valid.
        if let Some(id) = existing_id {
            let _ = evict_slot(&mut st, &self.hasher, id);
        }

        let deadline = deadline_after_write(&self.config.policy, &key, &value, self.start, now);
        install_ready(
            &mut st,
            &self.hasher,
            key,
            CacheEntry {
                value,
                written_at: clock::floor_tick(self.start, now),
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
        let mut st = self.shard_for(key).write();
        let id = entries_find(&st.entries, &self.hasher, &st.slab, key)?;
        let (k, v, _) = evict_slot(&mut st, &self.hasher, id)?;
        drop(st);
        let v_for_listener = v.clone();
        self.notify_lifecycle(LifecycleReason::Removed, k, Some(v_for_listener));
        Some(v)
    }

    /// On `Submit`, flips `refresh_inflight = true`; subsequent calls
    /// return `SkipInFlight` until `refresh_inflight_remove`.
    pub(crate) fn try_mark_refresh_inflight(&self, key: &K) -> RefreshGate
    where
        K: Eq + Hash,
    {
        let mut st = self.shard_for(key).write();
        let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) else {
            return RefreshGate::AbsentKey;
        };
        let Some(es) = st.slab.get_mut(id) else {
            return RefreshGate::AbsentKey;
        };
        if es.refresh_inflight {
            RefreshGate::SkipInFlight
        } else {
            es.refresh_inflight = true;
            RefreshGate::Submit
        }
    }

    pub(crate) fn apply_refresh_ok(&self, key: &K, v: V)
    where
        K: 'static,
        V: 'static,
    {
        let mut st = self.shard_for(key).write();
        let now = Instant::now();
        let nw = self.config.policy.weight(key, &v).get();
        let reclaimed = reclaim(self, &mut st, nw, key, now);
        let displaced = match entries_find(&st.entries, &self.hasher, &st.slab, key) {
            Some(id) => evict_slot(&mut st, &self.hasher, id).map(|(_, v, _)| v),
            None => None,
        };
        let deadline = deadline_after_write(&self.config.policy, key, &v, self.start, now);
        install_ready(
            &mut st,
            &self.hasher,
            key.clone(),
            CacheEntry {
                value: v,
                written_at: clock::floor_tick(self.start, now),
                deadline,
            },
            nw,
        );

        drop(st);
        // Mirror `Inner::insert`: emit Updated outside the lock.
        if let Some(prev) = displaced {
            self.notify_lifecycle(LifecycleReason::Updated, key.clone(), Some(prev));
        }
        self.fire_reclaim(reclaimed);
    }
}

pub(super) fn deadline_after_write<K, V, P>(
    policy: &P,
    key: &K,
    value: &V,
    start: Instant,
    now: Instant,
) -> clock::Tick
where
    K: 'static,
    V: 'static,
    P: Policy<K, V> + ?Sized,
{
    let d = policy.expire_after_write(key, value, now);
    if d >= crate::policy::TIME_UNBOUNDED {
        clock::NO_DEADLINE
    } else {
        match now.checked_add(d) {
            Some(t) => clock::ceil_tick(start, t),
            None => clock::NO_DEADLINE,
        }
    }
}

/// `NO_DEADLINE` (= `Tick::MAX`) folds into "always served" because
/// `now_tick < Tick::MAX` for any realistic clock.
pub(super) fn entry_is_served<V>(e: &CacheEntry<V>, now_tick: clock::Tick) -> bool {
    now_tick < e.deadline
}

// Sized as a fixed constant: deriving from `max_weight` is wrong as
// soon as `Policy::weight != 1`. Pick "big enough for typical working
// sets, small enough not to bloat per-shard memory".
const DEFAULT_GHOST_CAP_PER_SHARD: usize = 1024;

pub(super) struct ReadDeadlineUpdate {
    /// `Some` only when the read hook transitioned to a fresh finite
    /// deadline the caller still has to push into the heap.
    pub(super) new_deadline: Option<clock::Tick>,
    /// `true` when the prior deadline was finite — the heap now holds
    /// a stale tuple that contributes to the rebuild trigger.
    pub(super) superseded_finite: bool,
}

pub(super) fn update_read_deadline<K, V, P>(
    key: &K,
    e: &mut CacheEntry<V>,
    policy: &P,
    start: Instant,
    now: Instant,
) -> ReadDeadlineUpdate
where
    K: 'static,
    V: 'static,
    P: Policy<K, V> + ?Sized,
{
    let prev_tick = e.deadline;
    let time_remaining = if prev_tick == clock::NO_DEADLINE {
        crate::policy::TIME_UNBOUNDED
    } else {
        clock::tick_to_instant(start, prev_tick).saturating_duration_since(now)
    };
    let written_at = clock::tick_to_instant(start, e.written_at);
    let d = policy.expire_after_read(key, &e.value, now, written_at, time_remaining);
    let new_tick = if d >= crate::policy::TIME_UNBOUNDED {
        clock::NO_DEADLINE
    } else {
        match now.checked_add(d) {
            Some(t) => clock::ceil_tick(start, t),
            None => clock::NO_DEADLINE,
        }
    };
    e.deadline = new_tick;
    let new_deadline = if new_tick != clock::NO_DEADLINE && new_tick != prev_tick {
        Some(new_tick)
    } else {
        None
    };
    ReadDeadlineUpdate {
        new_deadline,
        superseded_finite: new_deadline.is_some() && prev_tick != clock::NO_DEADLINE,
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "typesize")]
mod typesize_impls {
    //! `extra_size()` on `Inner` walks every shard's slab under a read
    //! lock — fine for diagnostics, not a hot path.

    use core::mem;
    use std::any::TypeId;
    use std::sync::Arc;

    use hashbrown::{HashMap, HashTable};
    use typesize::TypeSize;

    use super::super::slab::slot_storage_bytes;
    use super::{CacheConfig, EntrySlab, EntrySlot, Inner, SlotKey, State};
    use crate::cache::flight::InflightLane;

    fn entries_extra(entries: &HashTable<SlotKey>) -> usize {
        entries.allocation_size()
    }

    fn slab_extra<K, V>(slab: &EntrySlab<K, V>) -> usize
    where
        K: TypeSize,
        V: TypeSize,
    {
        let storage = slot_storage_bytes::<EntrySlot<K, V>>(slab.capacity());
        let per_entry: usize = slab.values().map(EntrySlot::extra_size).sum();
        storage + per_entry
    }

    fn inflight_extra<K>(inflight: &HashMap<(K, TypeId), InflightLane>) -> usize
    where
        K: TypeSize + Eq + std::hash::Hash,
    {
        let buckets = inflight.allocation_size();
        let per_entry: usize = inflight
            .iter()
            .map(|((k, _tid), lane)| k.extra_size() + lane.extra_size())
            .sum();
        buckets + per_entry
    }

    impl<K, V> TypeSize for State<K, V>
    where
        K: TypeSize + Eq + std::hash::Hash,
        V: TypeSize,
    {
        fn extra_size(&self) -> usize {
            // Exhaustive destructure: a new `State` field becomes a
            // compile error until it's accounted for.
            let Self {
                entries,
                inflight,
                slab,
                small,
                main,
                ghost,
                expiry,
                small_weight,
                stale_pushes,
                total_weight,
            } = self;
            entries_extra(entries)
                + inflight_extra(inflight)
                + slab_extra(slab)
                + small.extra_size()
                + main.extra_size()
                + ghost.extra_size()
                + expiry.extra_size()
                + small_weight.extra_size()
                + stale_pushes.extra_size()
                + total_weight.extra_size()
        }
    }

    impl<K, V, P, S> TypeSize for CacheConfig<K, V, P, S>
    where
        P: TypeSize,
        S: TypeSize,
    {
        fn extra_size(&self) -> usize {
            // `lifecycle_listener: Arc<dyn …>` is type-erased beyond
            // the orphan rule — falls through `_`.
            let Self {
                max_weight,
                policy,
                hasher,
                shards,
                lifecycle_listener: _,
                _marker,
            } = self;
            max_weight.extra_size()
                + policy.extra_size()
                + hasher.extra_size()
                + shards.extra_size()
        }
    }

    impl<K, V, P, S> TypeSize for Inner<K, V, P, S>
    where
        K: TypeSize + Eq + std::hash::Hash,
        V: TypeSize,
        P: TypeSize,
        S: TypeSize,
    {
        fn extra_size(&self) -> usize {
            // `start: Instant` is std-owned (orphan rule) and stack-only.
            let Self {
                config,
                shards,
                hasher,
                shard_max_weight,
                start: _,
            } = self;
            shards.extra_size()
                + config.extra_size()
                + hasher.extra_size()
                + shard_max_weight.extra_size()
        }
    }

    pub(crate) fn arc_inner_size<T: TypeSize>(arc: &Arc<T>) -> usize {
        // 2 atomic counters (strong/weak) + size_of::<T>() + extras.
        2 * mem::size_of::<usize>() + mem::size_of::<T>() + arc.extra_size()
    }
}

#[cfg(feature = "typesize")]
pub(crate) use typesize_impls::arc_inner_size;
