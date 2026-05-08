use std::any::TypeId;
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
use std::time::Instant;

use clashcore::CachePadded;
use clashcore::lock::RwLock;
use hashbrown::{DefaultHashBuilder, HashMap, HashTable};

use super::expiry::{ExpiryHeap, expiry_push_extend};
use super::flight::InflightLane;
use super::reclaim::{Reclaimed, reclaim};
use super::slab::{CacheEntry, EntrySlab, EntrySlot, Fifo, GhostSet, Queue, SlotKey};
use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

/// One cache-padded shard. Aliased so the verbose `CachePadded<RwLock<...>>`
/// stack does not need to spell itself out at every use site (and so
/// clippy stops flagging `Inner::shards` for `type_complexity`).
pub(super) type Shard<K, V> = CachePadded<RwLock<State<K, V>>>;

/// Decision returned by [`Inner::try_mark_refresh_inflight`]. Mirrors the
/// public [`crate::rt::RevalidateScheduled`] variants so the call site only
/// has to translate, not branch on slot state directly.
pub(crate) enum RefreshGate {
    /// Caller should poll `init` and submit to the executor.
    Submit,
    /// A refresh for this key is already in flight; caller returns
    /// `RevalidateScheduled::SkippedInFlight`.
    SkipInFlight,
    /// No `Ready` slot exists for this key; caller returns
    /// `RevalidateScheduled::AbsentKey`.
    AbsentKey,
}

pub(crate) struct Inner<K: 'static, V: 'static, P: 'static, S: 'static = DefaultHashBuilder> {
    pub(crate) config: CacheConfig<K, V, P, S>,
    /// Per-shard state, cache-padded to keep adjacent shards out of each
    /// other's cache lines. Single-key paths take exactly one shard's
    /// `RwLock` (mostly `.write()` because reads mutate the slot's
    /// S3-FIFO `freq` counter and read-deadline state; pure read-only
    /// views — `contains_key`, `len`, `is_empty`, `weight` — take
    /// `.read()`). Multi-key paths fold across all shards.
    ///
    /// `clashcore::lock::RwLock` is `lock_api::RwLock<RawRwLock, T>`: a
    /// single-CAS uncontended fast path with no poisoning. Releasing
    /// before awaiting user `init` futures in [`Self::try_get_with`]
    /// remains a hard requirement.
    pub(crate) shards: Box<[Shard<K, V>]>,
    /// `usize::BITS - log2(shards.len())` for the shard count, used by
    /// the `(h << 7) >> shift` indexing borrowed from `clashcore`'s
    /// `ClashCollection`. The left-shift hides the high 7 bits of the
    /// hash from shard selection so `hashbrown`'s SIMD tag does not
    /// collide with our shard partition. Unused (and set to `0`) when
    /// `shards.len() == 1`; that case short-circuits to index 0 in
    /// [`Self::shard_for`] to avoid an `usize::BITS` shift (UB).
    pub(crate) shard_shift: u32,
    /// Hasher used both to pick a shard and (per shard) to hash table
    /// entries. Stored at the `Inner` level so all shards share the same
    /// seed — that keeps a key's bucket position stable across `Cache`
    /// clones and across shard boundaries during `clear`/iteration.
    ///
    /// Type-parameterised so callers can opt into `ahash`, `foldhash`,
    /// `xxh3`, or any other [`BuildHasher`] without us paying for a
    /// vtable on the hot path. Defaults to [`DefaultHashBuilder`]
    /// (foldhash via `hashbrown` 0.17).
    pub(crate) hasher: S,
    /// Per-shard weight cap: `ceil(max_weight / shards.len())`. With more
    /// than one shard the observed total weight may exceed
    /// `config.max_weight` by up to `shards.len() - 1` units of rounding
    /// slack; pass `.shards(1)` for a strict global cap.
    pub(crate) shard_max_weight: u64,
}

/// Inverse-fraction sizing for the S3-FIFO Small queue: per shard,
/// `small_max_weight = shard_max_weight / SMALL_DENOMINATOR`. Yang et
/// al. (SOSP'23) §4 default to S = 10%; the V1 mokabench validation
/// pass on the same trace set we baseline against (DS1, S2/S3, P-series,
/// OLTP) preferred a smaller Small (~3%) — closer to `quick_cache`'s
/// `hot_allocation = 0.97` — across nearly every operating point. Held
/// as a module-level constant rather than a builder knob so the tuning
/// surface stays narrow for V1; revisit if a workload demands per-cache
/// override.
pub(super) const SMALL_DENOMINATOR: u64 = 33;

/// Per-slot frequency-counter saturation cap. The cutoff at 3 mirrors
/// Yang et al.'s reference: enough resolution for `evict_from_small` to
/// distinguish "single-touch" from "warm" without growing the counter
/// past a single byte.
pub(super) const FREQ_CAP: u8 = 3;

/// `entries` lookup that compares keys through the slab — the table only
/// holds [`SlotKey`] so equality lives on `EntrySlot::key`. Returns
/// `None` when no slot matches (or when the matching bucket points at a
/// slot whose `slotmap` version no longer matches, which is the
/// self-cleaning behaviour we get for free from `slab.get`).
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

/// `entries` insert. Caller has already pushed `id` into the slab and is
/// responsible for the resize closure rehashing existing entries against
/// the same `slab`. The slot must be occupied at `id`.
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
    /// Lookup table for `Ready` entries. Stores only [`SlotKey`]; the key
    /// itself lives once on the slab as `EntrySlot::key`. Migrating from
    /// `HashMap<K, SlotKey>` to `HashTable<SlotKey>` removes one `K`
    /// allocation per ready entry (notable for `String`-keyed caches) and
    /// collapses the previous "find then remove" two-probe pattern into a
    /// single probe via `find_entry`.
    ///
    /// Equality during probes goes through [`SlotMap::get`] so the table
    /// can stay key-free; the `hasher` field below is the matching
    /// `BuildHasher` used both at insert time and at probe time so the
    /// stored hash matches what `find` recomputes.
    ///
    /// In-flight miss leaders live in [`Self::inflight`] under their own
    /// `(K, TypeId<F>)` lane so a `try_get_with::<…, E1>` call cannot
    /// deadlock waiting on a leader that will publish an unrelated `E2`.
    pub(crate) entries: HashTable<SlotKey>,
    /// In-flight miss lanes keyed by `(K, TypeId<F>)`. At most one
    /// leader per `(key, F)` pair: concurrent `try_get_with` calls that
    /// agree on both `K` and `F = MissFlight<V, E>` follow the same
    /// leader, while a disagreeing `E` (or the sync
    /// `SyncMissFlight<V>` lane for the same key) installs its own row.
    /// Borrowed lookups go through [`LaneKeyRef`] so `&K` paths do not
    /// clone the key.
    pub(super) inflight: HashMap<(K, TypeId), InflightLane>,
    /// Backing storage for `Ready` entries. The slab owns `CacheEntry`,
    /// the back-reference key, the cached weight, and the intrusive
    /// FIFO links. `slotmap` carries a per-slot version on the key;
    /// recycled slots produce a fresh [`SlotKey`] that no stale heap
    /// tuple or `entries` row will validate against.
    pub(super) slab: EntrySlab<K, V>,
    /// S3-FIFO Small queue. New inserts land here at the head and age
    /// toward the tail. The reclaim path's `evict_from_small` decides
    /// per slot whether to promote into `main` (warm) or evict +
    /// fingerprint into `ghost` (cold).
    pub(super) small: Fifo,
    /// S3-FIFO Main queue. Holds slots that survived a Small-tail
    /// promotion (`freq > 1` at the Small tail) or were re-inserted
    /// while their fingerprint sat in `ghost`. Drains via
    /// [`reclaim::evict_from_main`](super::reclaim) only when Small
    /// is at or under its share of the budget.
    pub(super) main: Fifo,
    /// Bounded set of fingerprints recently evicted from `small` with
    /// `freq <= 1`. The install path consumes membership
    /// (`ghost.remove(fp)`) to route a returning key straight to
    /// `main` with `freq = 0`.
    pub(super) ghost: GhostSet,
    /// Sum of [`Policy::weight`](crate::policy::Policy::weight) over
    /// slots currently in `small`. The reclaim outer loop drains Small
    /// first when `small_weight > shard_max_weight / SMALL_DENOMINATOR`,
    /// otherwise Main. Maintained eagerly on install / evict so the
    /// reclaim path does not have to recompute it.
    pub(super) small_weight: u64,
    /// Min-heap of `(deadline, slot)` tuples used to find past-deadline
    /// entries in `O(log n)`. Validated lazily on pop: a tuple is
    /// honoured only if `slab.get(slot)` returns `Some` (slotmap version
    /// still matches) **and** the slot's deadline equals the tuple's. The
    /// version check covers slot recycling and outright eviction; the
    /// deadline check covers read hooks that pushed a fresh tuple for the
    /// same live slot. Read-deadline updates always push a new tuple
    /// rather than mutating the old one.
    pub(super) expiry: ExpiryHeap,
    /// Number of heap pushes that superseded a still-finite prior deadline
    /// for the same slot (i.e. left a stale tuple behind). Read-extending
    /// policies — `IdleTimeout`, custom `expire_after_read` hooks — can
    /// pile these up arbitrarily under read-heavy workloads. When the
    /// stale count exceeds the live slab size we rebuild the heap from
    /// scratch so memory stays bounded by `slab.len()`.
    pub(super) stale_pushes: u64,
    /// Sum of [`Policy::weight`](crate::policy::Policy::weight) over `Ready` entries only.
    pub(crate) total_weight: u64,
}

impl<K, V> State<K, V> {
    /// Routes `Fifo::unlink` through the queue picked by `EntrySlot::queue`,
    /// hiding the disjoint `&mut self.{small|main}` + `&mut self.slab`
    /// borrow under one method call.
    pub(super) fn queue_unlink(&mut self, queue: Queue, id: SlotKey) {
        match queue {
            Queue::Small => self.small.unlink(&mut self.slab, id),
            Queue::Main => self.main.unlink(&mut self.slab, id),
        }
    }

    /// Push `id` at the head of the queue picked by `queue`.
    pub(super) fn queue_push_head(&mut self, queue: Queue, id: SlotKey) {
        match queue {
            Queue::Small => self.small.push_head(&mut self.slab, id),
            Queue::Main => self.main.push_head(&mut self.slab, id),
        }
    }

    /// Promote `id` to the head of its current queue (used by reclaim's
    /// skip-key short-circuit when the skip slot surfaces at the tail).
    pub(super) fn queue_promote_to_head(&mut self, queue: Queue, id: SlotKey) {
        match queue {
            Queue::Small => self.small.promote_to_head(&mut self.slab, id),
            Queue::Main => self.main.promote_to_head(&mut self.slab, id),
        }
    }

    /// Pop the tail of the queue picked by `queue`. Used by `clear` to
    /// drain Main and Small in turn.
    pub(super) fn queue_pop_tail(&mut self, queue: Queue) -> Option<SlotKey> {
        match queue {
            Queue::Small => self.small.pop_tail(&mut self.slab),
            Queue::Main => self.main.pop_tail(&mut self.slab),
        }
    }
}

/// Removes the slab slot at `id` and the matching `entries` map row, drops it
/// from whichever S3-FIFO queue it currently belongs to (consulted via
/// `EntrySlot::queue`), and returns the owned `(K, V)` plus the entry's
/// cached weight (already subtracted from `total_weight`, and from
/// `small_weight` when the slot lived in Small). Returns `None` if the
/// slot was already free (defensive — should not happen under our invariants).
/// Stale tuples in `state.expiry` are not removed here; lazy validation on
/// pop discards them once the slot is recycled. Stale `entries` rows are
/// likewise self-cleaned by the `slab.get(id).is_some_and(...)` filter
/// inside [`entries_find`] — so even if a bucket walk misses the row
/// to remove, no probe will ever return a value for a freed slot.
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
    // Take the slot out first so the entries removal can compare against
    // a slab that no longer holds `id` — the probe predicate uses
    // `slab.get(id)`, which now returns `None`, so we hash by the taken
    // entry's key and match buckets by `SlotKey` directly.
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

/// Allocate a slab slot for `key`/`entry`/`weight`, route it to the
/// right S3-FIFO queue (Main when its fingerprint sits in `ghost`,
/// otherwise Small), register it in `state.entries`, bump
/// `total_weight` (and `small_weight` when the entry lands in Small),
/// and (if applicable) push the deadline into `state.expiry`.
/// Centralises every install site so the heap invariant cannot be
/// silently dropped.
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
    // Ghost membership is consume-on-hit: a true return removes the
    // fingerprint and routes the entry into Main with freq=0.
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
    if let Some(dl) = deadline {
        st.expiry.push(dl, id);
    }
    id
}

impl<K, V, P, S> Inner<K, V, P, S>
where
    S: BuildHasher,
{
    /// Returns the shard responsible for `key`. Single-key paths funnel
    /// through here so the shard mapping is centralised; cross-shard
    /// work (`len`, `weight`, `clear`) iterates `self.shards` directly.
    ///
    /// Kept on a bound-free impl block so that the `LeaderGuard::drop`
    /// path can call it without re-asserting the full `Inner` bounds.
    /// Borrowed verbatim from `clashcore::ClashCollection::determine_shard`:
    /// `(hash << 7) >> shift` keeps the high 7 bits of the hash for
    /// hashbrown's SIMD tag while picking a shard from the next bits.
    pub(super) fn shard_for<Q>(&self, key: &Q) -> &Shard<K, V>
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let idx = if self.shards.len() == 1 {
            0
        } else {
            // Mirrors `clashcore::ClashCollection::determine_shard`: cast
            // the 64-bit hash down to `usize` and pick the upper
            // `log2(shard_count)` bits after the 7-bit SIMD-tag shift.
            // High bits get dropped on 32-bit targets, which is fine — the
            // default hasher distributes well enough that the truncation
            // does not bias shard selection.
            #[allow(clippy::cast_possible_truncation)]
            let h = self.hasher.hash_one(key) as usize;
            (h << 7) >> self.shard_shift
        };
        &self.shards[idx]
    }

    /// Clears the per-slot `refresh_inflight` flag for `key`. Best-effort:
    /// if the slot has been evicted in the meantime there is nothing to
    /// clear. Lives on the bound-free impl so cleanup paths
    /// (`RefreshGuard`, `RefreshWork::Drop`) can invoke it without the
    /// fuller `Policy<K, V>` bound the method-bearing impl carries.
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
        let shard_count = config.shards.max(1).next_power_of_two();
        let max_weight = config.max_weight.get();
        // Round up so the per-shard sum is at least `max_weight`. With more
        // than one shard total observed weight may exceed `max_weight` by
        // up to `shard_count - 1` (rounding); callers wanting a strict
        // global cap pass `.shards(1)`.
        let shard_max_weight = max_weight.div_ceil(shard_count as u64).max(1);
        // Pre-size each shard for its share of the budget. `weight >= 1`
        // is the working assumption that bounds per-shard entry count by
        // `shard_max_weight`.
        let initial_capacity = capacity_hint_from_max_weight(shard_max_weight);
        // Ghost capacity is an entry-count proxy, not a weight figure;
        // the same conservative `min(2^20, shard_max_weight)` estimate
        // that pre-sizes `entries` and `slab` keeps the ghost table's
        // memory bounded to one `u64` fingerprint per slot.
        let ghost_cap = initial_capacity;
        let shards = (0..shard_count)
            .map(|_| {
                CachePadded::new(RwLock::new(State {
                    entries: HashTable::with_capacity(initial_capacity),
                    inflight: HashMap::new(),
                    slab: EntrySlab::with_capacity_and_key(initial_capacity),
                    small: Fifo::default(),
                    main: Fifo::default(),
                    ghost: GhostSet::new(ghost_cap),
                    small_weight: 0,
                    expiry: ExpiryHeap::default(),
                    stale_pushes: 0,
                    total_weight: 0,
                }))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        // Match clashcore::ClashCollection's shard indexing: shift leaves
        // the high 7 bits of the hash undisturbed for hashbrown's SIMD
        // tag. The single-shard branch in `shard_for` bypasses the shift
        // entirely so `usize::BITS` here would be UB — set to 0 instead.
        let shard_shift = if shard_count == 1 {
            0
        } else {
            usize::BITS - shard_count.trailing_zeros()
        };
        let hasher = config.hasher.clone();
        Self {
            config,
            shards,
            shard_shift,
            hasher,
            shard_max_weight,
        }
    }

    pub(crate) fn get_served<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q> + 'static,
        Q: Hash + Eq + ?Sized,
        V: 'static,
    {
        let now = Instant::now();
        let mut st = self.shard_for(key).write();

        let id = entries_find(&st.entries, &self.hasher, &st.slab, key)?;

        // Fast-path read: borrow the slot to inspect the deadline; if expired,
        // drop the borrow and fall into the eviction path below.
        let expired = st
            .slab
            .get(id)
            .is_some_and(|es| !entry_is_served(&es.entry, now));
        if expired {
            // Use the slot's owned key for the lifecycle event so callers
            // querying with a borrowed `Q` need not own a `K`.
            let (k, v, _) = evict_slot(&mut st, &self.hasher, id)?;
            drop(st);
            self.notify_lifecycle(LifecycleReason::TimeExpired, k, Some(v));
            return None;
        }

        let (value, update) = {
            let es = st.slab.get_mut(id)?;
            es.freq = es.freq.saturating_add(1).min(FREQ_CAP);
            let value = es.entry.value.clone();
            // Disjoint-field borrow: `&es.key` (immutable) and `&mut es.entry`
            // (mutable) don't overlap, so the borrow checker accepts both.
            let update = update_read_deadline(&es.key, &mut es.entry, &self.config.policy, now);
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
        // Read-only check: do not evict expired entries here; let the next
        // `get`/`insert` reclaim path do it so contains_key has no side
        // effects beyond taking the lock.
        let now = Instant::now();
        st.slab
            .get(id)
            .is_some_and(|es| entry_is_served(&es.entry, now))
    }

    /// Sums per-shard `slab.len()`. Briefly takes each shard's read lock
    /// in turn; not snapshot-consistent under concurrent mutation, but
    /// each shard's contribution is taken atomically.
    pub(crate) fn len(&self) -> usize {
        self.shards.iter().map(|s| s.read().slab.len()).sum()
    }

    /// Short-circuiting form of `len() == 0`: bails out on the first
    /// non-empty shard so a busy cache only pays for a single lock+probe.
    /// Inherits the non-snapshot caveat from [`Self::len`].
    pub(crate) fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.read().slab.len() == 0)
    }

    /// Sums per-shard `total_weight`. Same consistency caveat as
    /// [`Self::len`].
    pub(crate) fn weight(&self) -> u64 {
        self.shards.iter().map(|s| s.read().total_weight).sum()
    }

    /// Drains every `Ready` slot across every shard, fires `Removed` for
    /// each one outside the locks, and cancels every in-flight miss lane
    /// so any waiter sees a retry signal rather than blocking forever on
    /// an abandoned flight. Walks shards in deterministic index order so
    /// `Removed` callbacks observe a stable cross-shard sequence.
    pub(crate) fn clear(&self)
    where
        K: 'static,
        V: 'static,
    {
        for shard in &self.shards {
            let mut st = shard.write();
            let mut drained: Vec<(K, V)> = Vec::with_capacity(st.slab.len());
            // Deterministic drain order: Main first (warm tail-to-head),
            // then Small (cold tail-to-head). The order does not carry
            // an "oldest-first" meaning (S3-FIFO has no global recency
            // axis); it is documented purely so observers see a stable
            // sequence of `Removed` lifecycle events across runs.
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

            // Cancel pending miss lanes before releasing the lock so
            // followers see Cancelled instead of waiting on a sender we
            // are about to drop.
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

        // Single lookup: cache the existing slot id, then derive both the
        // returned `prev` value and the displaced-slot eviction from it.
        // Avoids two passes through `entries.find` per insert.
        let existing_id = entries_find(&st.entries, &self.hasher, &st.slab, &key);
        let prev = existing_id.and_then(|id| st.slab.get(id).map(|es| es.entry.value.clone()));

        let reclaimed = reclaim(self, &mut st, new_w, &key, now);

        // `reclaim` never evicts `skip_key`, so `existing_id` is still valid
        // here if it was valid before the call.
        if let Some(id) = existing_id {
            let _ = evict_slot(&mut st, &self.hasher, id);
        }

        let deadline = deadline_after_write(&self.config.policy, &key, &value, now);
        install_ready(
            &mut st,
            &self.hasher,
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
        let mut st = self.shard_for(key).write();
        let id = entries_find(&st.entries, &self.hasher, &st.slab, key)?;
        let (k, v, _) = evict_slot(&mut st, &self.hasher, id)?;
        drop(st);
        let v_for_listener = v.clone();
        self.notify_lifecycle(LifecycleReason::Removed, k, Some(v_for_listener));
        Some(v)
    }

    /// Atomically inspects the slot for `key` and decides whether
    /// `try_revalidate` should submit, skip (already inflight), or refuse
    /// (no `Ready` slot). On `Submit`, sets the slot's
    /// `refresh_inflight = true` so the next call returns `SkipInFlight`
    /// until the matching `refresh_inflight_remove`.
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
        let deadline = deadline_after_write(&self.config.policy, key, &v, now);
        install_ready(
            &mut st,
            &self.hasher,
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

pub(super) fn deadline_after_write<K, V, P>(
    policy: &P,
    key: &K,
    value: &V,
    now: Instant,
) -> Option<Instant>
where
    K: 'static,
    V: 'static,
    P: Policy<K, V> + ?Sized,
{
    let d = policy.expire_after_write(key, value, now);
    if d >= crate::policy::TIME_UNBOUNDED {
        None
    } else {
        now.checked_add(d)
    }
}

/// Whether the entry may be returned for a read at `now`, using only materialized [`CacheEntry::deadline`].
///
/// [`Policy::expire_after_read`](crate::policy::Policy::expire_after_read) runs only after a live hit
/// in [`update_read_deadline`] to compute the next deadline.
pub(super) fn entry_is_served<V>(e: &CacheEntry<V>, now: Instant) -> bool {
    !matches!(e.deadline, Some(dl) if now >= dl)
}

/// Pre-sizing hint for `State::entries` and `State::slab`. Bounds the
/// rare-resize tail without allocating absurdly large tables for callers who
/// set huge weight budgets: once the per-entry `Policy::weight` is `≥ 1`,
/// total entry count is bounded by `max_weight`, but we still cap the
/// *allocation* here because the configured budget is a weight unit, not
/// necessarily an entry count. Pure pre-allocation hint — both the
/// hashtable and slab grow on demand, so the exact value only affects
/// the cost of the first few inserts.
const CAPACITY_HINT_CAP: u64 = 1 << 20;

fn capacity_hint_from_max_weight(max_weight: u64) -> usize {
    let bounded = max_weight.min(CAPACITY_HINT_CAP);
    usize::try_from(bounded).unwrap_or(usize::MAX)
}

/// Outcome of [`update_read_deadline`]. `new_deadline` is `Some` only when
/// the deadline transitioned to a fresh finite value the caller still has
/// to push into the expiry heap. `superseded_finite` is `true` when the
/// previous deadline was finite (so any tuple already in the heap for
/// this slot is now stale) — read-extending policies pile these up over
/// time and are what trigger the periodic rebuild.
pub(super) struct ReadDeadlineUpdate {
    pub(super) new_deadline: Option<Instant>,
    pub(super) superseded_finite: bool,
}

/// Apply the read hook to `e.deadline` in place. The returned
/// [`ReadDeadlineUpdate`] tells the caller whether to push a fresh tuple
/// and whether that push leaves a stale tuple behind.
pub(super) fn update_read_deadline<K, V, P>(
    key: &K,
    e: &mut CacheEntry<V>,
    policy: &P,
    now: Instant,
) -> ReadDeadlineUpdate
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
    let new = if d >= crate::policy::TIME_UNBOUNDED {
        None
    } else {
        now.checked_add(d)
    };
    e.deadline = new;
    let new_deadline = match new {
        Some(dl) if Some(dl) != prev => Some(dl),
        _ => None,
    };
    ReadDeadlineUpdate {
        new_deadline,
        superseded_finite: new_deadline.is_some() && prev.is_some(),
    }
}

#[cfg(test)]
mod tests;
