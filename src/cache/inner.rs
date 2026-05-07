use std::any::TypeId;
use std::borrow::Borrow;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;
use std::time::Instant;

use clashcore::CachePadded;
use clashcore::lock::RwLock;
use hashbrown::{DefaultHashBuilder, HashMap, HashTable};
use slotmap::Key;

use super::flight::{
    Flight, InflightLane, LeaderGuard, MissFlight, MissStep, NewFlight, SyncMissFlight,
    install_or_join_lane, remove_lane,
};
use super::slab::{CacheEntry, EntrySlab, EntrySlot, Fifo, GhostSet, Queue, SlotKey};
use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

/// One cache-padded shard. Aliased so the verbose `CachePadded<RwLock<...>>`
/// stack does not need to spell itself out at every use site (and so
/// clippy stops flagging `Inner::shards` for `type_complexity`).
type Shard<K, V> = CachePadded<RwLock<State<K, V>>>;

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

/// Hash a key using the table's `BuildHasher`. Centralising the hash
/// derivation guarantees insert and lookup hash the same bytes the same
/// way; otherwise a `Borrow`-related mismatch would silently produce
/// missed lookups.
fn hash_key<H: BuildHasher, Q: Hash + ?Sized>(hasher: &H, key: &Q) -> u64 {
    hasher.hash_one(key)
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
const SMALL_DENOMINATOR: u64 = 33;

/// Per-slot frequency-counter saturation cap. The cutoff at 3 mirrors
/// Yang et al.'s reference: enough resolution for `evict_from_small` to
/// distinguish "single-touch" from "warm" without growing the counter
/// past a single byte.
const FREQ_CAP: u8 = 3;

/// `entries` lookup that compares keys through the slab — the table only
/// holds [`SlotKey`] so equality lives on `EntrySlot::key`. Returns
/// `None` when no slot matches (or when the matching bucket points at a
/// slot whose `slotmap` version no longer matches, which is the
/// self-cleaning behaviour we get for free from `slab.get`).
fn entries_find<K, V, Q, H>(
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
    let h = hash_key(hasher, key);
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
        hash_key(
            hasher,
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
    slab: EntrySlab<K, V>,
    /// S3-FIFO Small queue. New inserts land here at the head and age
    /// toward the tail. The reclaim path's `evict_from_small` decides
    /// per slot whether to promote into `main` (warm) or evict +
    /// fingerprint into `ghost` (cold).
    small: Fifo,
    /// S3-FIFO Main queue. Holds slots that survived a Small-tail
    /// promotion or were re-inserted while their fingerprint sat in
    /// `ghost`. Stage 2 keeps this empty; the cutover commit wires
    /// admission and the recirculation loop.
    main: Fifo,
    /// Bounded set of fingerprints recently evicted from `small`. The
    /// install path consumes membership to route a returning key
    /// straight to `main`. Stage 2 keeps this empty; the cutover commit
    /// wires population and consumption.
    ghost: GhostSet,
    /// Sum of [`Policy::weight`](crate::policy::Policy::weight) over
    /// slots currently in `small`. Cutover uses
    /// `small_weight > shard_max_weight / SMALL_DENOMINATOR` to decide
    /// whether to drain Small first vs Main. Maintained eagerly on
    /// install / evict so the cutover does not have to retrofit the
    /// invariant onto existing call sites.
    small_weight: u64,
    /// Min-heap of `(deadline, slot)` tuples used to find past-deadline
    /// entries in `O(log n)`. Validated lazily on pop: a tuple is
    /// honoured only if `slab.get(slot)` returns `Some` (slotmap version
    /// still matches) **and** the slot's deadline equals the tuple's. The
    /// version check covers slot recycling and outright eviction; the
    /// deadline check covers read hooks that pushed a fresh tuple for the
    /// same live slot. Read-deadline updates always push a new tuple
    /// rather than mutating the old one.
    expiry: ExpiryHeap,
    /// Number of heap pushes that superseded a still-finite prior deadline
    /// for the same slot (i.e. left a stale tuple behind). Read-extending
    /// policies — `IdleTimeout`, custom `expire_after_read` hooks — can
    /// pile these up arbitrarily under read-heavy workloads. When the
    /// stale count exceeds the live slab size we rebuild the heap from
    /// scratch so memory stays bounded by `slab.len()`.
    stale_pushes: u64,
    /// Sum of [`Policy::weight`](crate::policy::Policy::weight) over `Ready` entries only.
    pub(crate) total_weight: u64,
}

/// `(deadline, slot)` tuple stored in [`ExpiryHeap`]. Ordered by
/// `deadline` so a `BinaryHeap<Reverse<HeapKey>>` yields earliest
/// deadlines first; the slot's packed FFI form is a tiebreaker for
/// total ordering only — the slotmap version inside [`SlotKey`] is what
/// gates stale tuples on pop, not anything in the heap ordering.
#[derive(Clone, Copy, PartialEq, Eq)]
struct HeapKey {
    deadline: Instant,
    slot: SlotKey,
}

impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deadline
            .cmp(&other.deadline)
            .then_with(|| self.slot.data().as_ffi().cmp(&other.slot.data().as_ffi()))
    }
}

impl PartialOrd for HeapKey {
    // `BinaryHeap` calls through `Ord::cmp`, so this body is never
    // exercised from the cache; required only to satisfy the trait.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct ExpiryHeap(BinaryHeap<Reverse<HeapKey>>);

impl ExpiryHeap {
    fn push(&mut self, deadline: Instant, slot: SlotKey) {
        self.0.push(Reverse(HeapKey { deadline, slot }));
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

/// Push a deadline tuple into the expiry heap and bump `stale_pushes`
/// when the push leaves a previous (still-finite) tuple for the same slot
/// behind. When stale tuples outnumber live entries the heap is rebuilt
/// from the slab so memory stays bounded by `slab.len()`. Pure-`State`
/// helper so it's reachable from both the read and miss-probe paths.
fn expiry_push_extend<K, V>(
    st: &mut State<K, V>,
    deadline: Instant,
    slot: SlotKey,
    superseded_finite: bool,
) {
    st.expiry.push(deadline, slot);
    if superseded_finite {
        st.stale_pushes = st.stale_pushes.saturating_add(1);
        // Rebuilding under the same lock keeps the invariant simple
        // (no stale tuple ever escapes a probe) and is amortised: the
        // trigger only fires once per `slab.len()` superseded pushes.
        if st.stale_pushes > st.slab.len() as u64 {
            rebuild_expiry(st);
        }
    }
}

/// Walk the slab and rebuild the expiry heap from the live deadline of
/// each occupied slot. `stale_pushes` resets to 0; the new heap holds at
/// most one tuple per live entry.
fn rebuild_expiry<K, V>(st: &mut State<K, V>) {
    st.expiry.clear();
    for (id, es) in &st.slab {
        if let Some(dl) = es.entry.deadline {
            st.expiry.push(dl, id);
        }
    }
    st.stale_pushes = 0;
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
fn drain_expired<K, V, H>(
    st: &mut State<K, V>,
    hasher: &H,
    now: Instant,
    skip_key: Option<&K>,
) -> Vec<(K, V)>
where
    K: Eq + Hash,
    H: BuildHasher,
{
    let mut out = Vec::new();
    while let Some(top) = st.expiry.peek() {
        if top.deadline > now {
            break;
        }
        st.expiry.pop();
        // `slab.get` returns `None` if the slot was evicted or recycled;
        // the slotmap version on `top.slot` is the gate, no separate
        // generation field needed. The deadline check still discards
        // tuples that a read hook superseded for the same live slot.
        let Some(es) = st.slab.get(top.slot) else {
            continue;
        };
        if es.entry.deadline != Some(top.deadline) {
            continue;
        }
        if let Some(skip) = skip_key
            && &es.key == skip
        {
            continue;
        }
        if let Some((k, v, _)) = evict_slot(st, hasher, top.slot) {
            out.push((k, v));
        }
    }
    out
}

/// Result of a single [`Inner::reclaim`] pass. Split per lifecycle reason so
/// the caller fires `TimeExpired` for the deadline drain and `EvictedSize`
/// for the S3-FIFO size drain that brought weight back under capacity.
struct Reclaimed<K, V> {
    expired: Vec<(K, V)>,
    evicted: Vec<(K, V)>,
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
fn evict_slot<K, V, H>(
    st: &mut State<K, V>,
    hasher: &H,
    id: SlotKey,
) -> Option<(K, V, u64)>
where
    K: Eq + Hash,
    H: BuildHasher,
{
    let State {
        entries,
        slab,
        small,
        main,
        small_weight,
        total_weight,
        ..
    } = st;
    let queue = slab.get(id)?.queue;
    let fifo = match queue {
        Queue::Small => small,
        Queue::Main => main,
    };
    fifo.unlink(slab, id);
    // Take the slot out first so the entries removal can compare against
    // a slab that no longer holds `id` — the probe predicate uses
    // `slab.get(id)`, which now returns `None`, so we hash by the taken
    // entry's key and match buckets by `SlotKey` directly.
    let entry_slot = slab.remove(id)?;
    *total_weight = total_weight.saturating_sub(entry_slot.weight);
    if matches!(queue, Queue::Small) {
        *small_weight = small_weight.saturating_sub(entry_slot.weight);
    }
    let h = hash_key(hasher, &entry_slot.key);
    if let Ok(occ) = entries.find_entry(h, |&other| other == id) {
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
fn install_ready<K, V, H>(
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
    let State {
        entries,
        slab,
        small,
        main,
        ghost,
        small_weight,
        expiry,
        total_weight,
        ..
    } = st;
    let deadline = entry.deadline;
    let key_hash = hash_key(hasher, &key);
    // Ghost membership is consume-on-hit: a true return removes the
    // fingerprint and routes the entry into Main with freq=0.
    let to_main = ghost.remove(key_hash);
    let queue = if to_main { Queue::Main } else { Queue::Small };
    let id = slab.insert(EntrySlot {
        key,
        entry,
        weight,
        queue,
        freq: 0,
        fifo_prev: None,
        fifo_next: None,
        refresh_inflight: false,
    });
    if to_main {
        main.push_head(slab, id);
    } else {
        small.push_head(slab, id);
        *small_weight = small_weight.saturating_add(weight);
    }
    entries_insert(entries, hasher, slab, id, key_hash);
    *total_weight = total_weight.saturating_add(weight);
    if let Some(dl) = deadline {
        expiry.push(dl, id);
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
            let h = hash_key(&self.hasher, key) as usize;
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
            let State {
                entries,
                inflight,
                slab,
                small,
                main,
                ghost,
                small_weight,
                expiry,
                stale_pushes,
                total_weight,
                ..
            } = &mut *st;
            let mut drained: Vec<(K, V)> = Vec::with_capacity(slab.len());
            // Deterministic drain order: Main first (warm tail-to-head),
            // then Small (cold tail-to-head). The order does not carry
            // an "oldest-first" meaning (S3-FIFO has no global recency
            // axis); it is documented purely so observers see a stable
            // sequence of `Removed` lifecycle events across runs.
            while let Some(tail) = main.pop_tail(slab) {
                if let Some(entry_slot) = slab.remove(tail) {
                    drained.push((entry_slot.key, entry_slot.entry.value));
                }
            }
            while let Some(tail) = small.pop_tail(slab) {
                if let Some(entry_slot) = slab.remove(tail) {
                    drained.push((entry_slot.key, entry_slot.entry.value));
                }
            }
            ghost.clear();
            entries.clear();
            expiry.clear();
            *stale_pushes = 0;
            *total_weight = 0;
            *small_weight = 0;

            // Cancel pending miss lanes before releasing the lock so
            // followers see Cancelled instead of waiting on a sender we
            // are about to drop.
            let cancelled: Vec<InflightLane> = inflight.drain().map(|(_, v)| v).collect();
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

        let reclaimed = self.reclaim(&mut st, new_w, &key, now);

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

    /// Two-phase capacity reclaim, called before installing a new entry of
    /// weight `add_w` for `skip_key`:
    ///
    /// 1. **Expiry drain** — pop heap tuples with `deadline ≤ now` and evict
    ///    the matching slots. Lazy validation discards stale tuples in O(1).
    ///    Cost: `O(d log n)` for `d` validated victims; `O(s log n)` for
    ///    `s` stale tuples observed.
    /// 2. **S3-FIFO size drain** — while the projected post-install weight
    ///    exceeds `max_weight`, drain Small first when it sits over its
    ///    `shard_max_weight / SMALL_DENOMINATOR` budget, otherwise drain
    ///    Main. Each `evict_from_small` either promotes the tail slot to
    ///    Main (when its `freq > 1`) or evicts + records its fingerprint
    ///    in `ghost`. Each `evict_from_main` either recirculates the
    ///    tail (decrementing `freq`) or evicts. The skip-key short-circuit
    ///    is preserved on both queues: the skip slot, when surfacing at
    ///    the tail, is promoted back to its current queue's head.
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
        let expired = drain_expired(st, &self.hasher, now, Some(skip_key));

        // Hoist the existing-slot weight lookup out of the loop: the
        // skip_key short-circuit means its slot is never reclaimed in this
        // pass, so its weight is invariant across iterations. Only
        // `state.total_weight` changes per iteration; the projection then
        // becomes a couple of arithmetic ops instead of a HashTable probe.
        let displaced_weight = entries_find(&st.entries, &self.hasher, &st.slab, skip_key)
            .and_then(|id| st.slab.get(id))
            .map_or(0, |es| es.weight);

        let max = self.shard_max_weight;
        let small_max = max / SMALL_DENOMINATOR;
        let mut evicted = Vec::new();
        loop {
            if !Self::projected_weight_over_capacity(st, displaced_weight, add_w, max) {
                break;
            }
            // Try the prescribed queue first; fall back to the other if
            // the prescribed one had nothing actionable. This handles
            // the transient cases where Small drained to empty mid-pass
            // (we still need to reclaim from Main) and vice versa.
            let small_first = st.small_weight > small_max;
            let primary = if small_first {
                self.evict_from_small(st, skip_key, &mut evicted)
            } else {
                self.evict_from_main(st, skip_key, &mut evicted)
            };
            if primary {
                continue;
            }
            let secondary = if small_first {
                self.evict_from_main(st, skip_key, &mut evicted)
            } else {
                self.evict_from_small(st, skip_key, &mut evicted)
            };
            if !secondary {
                break;
            }
        }
        Reclaimed { expired, evicted }
    }

    /// One step against the Small queue's tail. Returns `true` when state
    /// changed (promotion to Main, eviction, or skip-key rotation) so the
    /// outer reclaim loop iterates; `false` when Small is empty.
    fn evict_from_small(
        &self,
        st: &mut State<K, V>,
        skip_key: &K,
        evicted: &mut Vec<(K, V)>,
    ) -> bool
    where
        K: 'static,
        V: 'static,
    {
        let Some(tail_id) = st.small.tail() else {
            return false;
        };
        let (key_eq_skip, freq, fp) = {
            let Some(tail_slot) = st.slab.get(tail_id) else {
                return false;
            };
            (
                &tail_slot.key == skip_key,
                tail_slot.freq,
                hash_key(&self.hasher, &tail_slot.key),
            )
        };

        if key_eq_skip {
            if st.slab.len() <= 1 {
                return false;
            }
            let State { slab, small, .. } = &mut *st;
            small.promote_to_head(slab, tail_id);
            return true;
        }

        if freq > 1 {
            // Warm slot: migrate Small → Main with freq reset to 0.
            let State {
                slab,
                small,
                main,
                small_weight,
                ..
            } = &mut *st;
            small.unlink(slab, tail_id);
            let es = slab
                .get_mut(tail_id)
                .expect("Small tail must be occupied");
            let w = es.weight;
            es.queue = Queue::Main;
            es.freq = 0;
            *small_weight = small_weight.saturating_sub(w);
            main.push_head(slab, tail_id);
            true
        } else {
            // Cold slot: evict and seed ghost so a quick re-insert
            // routes the returning entry straight to Main.
            let Some((k, v, _)) = evict_slot(st, &self.hasher, tail_id) else {
                return false;
            };
            st.ghost.insert(fp);
            evicted.push((k, v));
            true
        }
    }

    /// One step against the Main queue's tail. Returns `true` when state
    /// changed (recirculation, eviction, or skip-key rotation) so the
    /// outer reclaim loop iterates; `false` when Main is empty.
    fn evict_from_main(
        &self,
        st: &mut State<K, V>,
        skip_key: &K,
        evicted: &mut Vec<(K, V)>,
    ) -> bool
    where
        K: 'static,
        V: 'static,
    {
        let Some(tail_id) = st.main.tail() else {
            return false;
        };
        let (key_eq_skip, freq) = {
            let Some(tail_slot) = st.slab.get(tail_id) else {
                return false;
            };
            (&tail_slot.key == skip_key, tail_slot.freq)
        };

        if key_eq_skip {
            if st.slab.len() <= 1 {
                return false;
            }
            let State { slab, main, .. } = &mut *st;
            main.promote_to_head(slab, tail_id);
            return true;
        }

        if freq > 0 {
            // Recirculate: decrement freq and reinsert at Main head.
            // Each Main slot is touched at most FREQ_CAP+1 times before
            // being forced to freq=0 and dropped, bounding loop work.
            let State { slab, main, .. } = &mut *st;
            main.unlink(slab, tail_id);
            let es = slab.get_mut(tail_id).expect("Main tail must be occupied");
            es.freq = es.freq.saturating_sub(1);
            main.push_head(slab, tail_id);
            true
        } else {
            // No second-chance seed: only Small evictions populate ghost.
            let Some((k, v, _)) = evict_slot(st, &self.hasher, tail_id) else {
                return false;
            };
            evicted.push((k, v));
            true
        }
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

    /// Installs a fresh `Ready` slot for a leader's miss outcome.
    ///
    /// Returns the post-install reclaim victims plus the prior value (if a
    /// concurrent `Cache::insert` for the same key landed while the leader
    /// was running its `init`); callers fire `Updated` for that prior
    /// value once the shard lock is released, mirroring [`Self::insert`]
    /// and [`Self::apply_refresh_ok`].
    fn finish_miss_ok(
        &self,
        st: &mut State<K, V>,
        key: &K,
        v: &V,
        now: Instant,
    ) -> (Reclaimed<K, V>, Option<V>)
    where
        K: 'static,
        V: 'static,
    {
        let deadline = deadline_after_write(&self.config.policy, key, v, now);
        let nw = self.config.policy.weight(key, v).get();
        let reclaimed = self.reclaim(st, nw, key, now);
        let displaced = entries_find(&st.entries, &self.hasher, &st.slab, key)
            .and_then(|id| evict_slot(st, &self.hasher, id).map(|(_, prev, _)| prev));
        install_ready(
            st,
            &self.hasher,
            key.clone(),
            CacheEntry {
                value: v.clone(),
                written_at: now,
                deadline,
            },
            nw,
        );
        (reclaimed, displaced)
    }

    /// Acquires the shard write lock, drains an expired entry (if any),
    /// bumps the freq counter on a still-live entry + pushes a
    /// refreshed read-side expiry, then hands control to `then` with
    /// the held lock and the hit (if any). The closure runs in the
    /// same critical section so the install/join-lane decision sees no
    /// TOCTOU window between hit detection and lane install.
    ///
    /// Releases and retakes the shard lock around the `TimeExpired`
    /// lifecycle callback so observers may re-enter the cache. The
    /// closure pattern keeps the [`RwLockWriteGuard`] type private to
    /// this method so we do not need `lock_api` as a direct dep.
    fn with_probe_for_hit<R>(
        &self,
        shard: &Shard<K, V>,
        key: &K,
        then: impl FnOnce(&mut State<K, V>, Option<V>) -> R,
    ) -> R
    where
        K: 'static,
        V: 'static,
    {
        let mut st = shard.write();

        let now = Instant::now();
        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let expired = st
                .slab
                .get(id)
                .is_some_and(|es| !entry_is_served(&es.entry, now));
            if expired && let Some((_, v, _)) = evict_slot(&mut st, &self.hasher, id) {
                drop(st);
                self.notify_lifecycle(LifecycleReason::TimeExpired, key.clone(), Some(v));
                st = shard.write();
            }
        }

        let now = Instant::now();
        let mut hit = None;
        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let still_served = st
                .slab
                .get(id)
                .is_some_and(|es| entry_is_served(&es.entry, now));
            if still_served {
                let (value, update) = {
                    let es = st
                        .slab
                        .get_mut(id)
                        .expect("just confirmed Occupied above under our mutex");
                    es.freq = es.freq.saturating_add(1).min(FREQ_CAP);
                    let value = es.entry.value.clone();
                    let update = update_read_deadline(key, &mut es.entry, &self.config.policy, now);
                    (value, update)
                };
                if let Some(dl) = update.new_deadline {
                    expiry_push_extend(&mut st, dl, id, update.superseded_finite);
                }
                hit = Some(value);
            }
        }

        then(&mut st, hit)
    }

    /// Single critical section for one miss iteration on flight type `F`:
    /// served hit, in-flight miss wait, or install miss leader. Avoids
    /// TOCTOU between separate `get_served` / miss checks / leader insert.
    /// Used by both `try_get_with` (`F = MissFlight<V, E>`) and
    /// `get_or_insert_with` (`F = SyncMissFlight<V>`).
    fn miss_probe<F>(&self, key: &K) -> MissStep<V, F>
    where
        K: Eq + Hash + Clone + 'static,
        V: Clone + Send + Sync + 'static,
        F: Flight + NewFlight,
    {
        let shard = self.shard_for(key);
        self.with_probe_for_hit(shard, key, |st, hit| {
            if let Some(v) = hit {
                return MissStep::Hit(v);
            }
            install_or_join_lane::<K, V, F>(st, key)
        })
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
            match self.miss_probe::<MissFlight<V, E>>(&key) {
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
                    let mut guard = LeaderGuard::<MissFlight<V, E>, K, V, P, S> {
                        inner: self.as_ref(),
                        key: &key,
                        flight: miss,
                        armed: true,
                    };
                    let fut = init();
                    let res = fut.await;
                    let mut st = guard.inner.shard_for(&key).write();
                    remove_lane(
                        &mut st,
                        &key,
                        TypeId::of::<MissFlight<V, E>>(),
                        Arc::as_ptr(&guard.flight).cast::<()>(),
                    );

                    let finish = if let Ok(v) = &res {
                        let now = Instant::now();
                        Some(self.finish_miss_ok(&mut st, &key, v, now))
                    } else {
                        None
                    };

                    // Publish under the lock so a follower that wakes and
                    // immediately calls `get(key)` cannot observe a state
                    // where the value was reported settled but is not yet
                    // installed in `entries`.
                    guard.flight.publish_result(res.clone());
                    guard.armed = false;
                    drop(guard);
                    drop(st);

                    if let Some((rec, displaced)) = finish {
                        if let Some(prev) = displaced {
                            self.notify_lifecycle(
                                LifecycleReason::Updated,
                                key.clone(),
                                Some(prev),
                            );
                        }
                        self.fire_reclaim(rec);
                    }
                    return res;
                }
            }
        }
    }


    /// Sync miss-coalescing entry point. The user closure runs **without**
    /// the shard lock held, so the cache stays responsive on other keys
    /// while the leader produces a value.
    ///
    /// `key` is taken by value to match the async [`Self::try_get_with`]
    /// signature; internally it is borrowed everywhere except the owned
    /// clone that goes into the `inflight` map and the slab.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn get_or_insert_with<F>(self: &Arc<Self>, key: K, init: F) -> V
    where
        K: Eq + Hash + Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
        F: FnOnce(&K) -> V,
    {
        let mut init = Some(init);
        loop {
            match self.miss_probe::<SyncMissFlight<V>>(&key) {
                MissStep::Hit(v) => return v,
                MissStep::Wait(f) => {
                    if let Some(v) = f.wait() {
                        return v;
                    }
                    // Leader cancelled — its lane is gone; loop and either
                    // observe a fresh value or take leadership ourselves.
                }
                MissStep::Leader { miss } => {
                    let init = init
                        .take()
                        .expect("init consumed: only the leader arm consumes it");
                    let mut guard = LeaderGuard::<SyncMissFlight<V>, K, V, P, S> {
                        inner: self.as_ref(),
                        key: &key,
                        flight: miss,
                        armed: true,
                    };
                    let value = init(&key);
                    let mut st = guard.inner.shard_for(&key).write();
                    remove_lane(
                        &mut st,
                        &key,
                        TypeId::of::<SyncMissFlight<V>>(),
                        Arc::as_ptr(&guard.flight).cast::<()>(),
                    );
                    let now = Instant::now();
                    let (reclaimed, displaced) = self.finish_miss_ok(&mut st, &key, &value, now);
                    guard.flight.publish(value.clone());
                    guard.armed = false;
                    drop(guard);
                    drop(st);
                    if let Some(prev) = displaced {
                        self.notify_lifecycle(LifecycleReason::Updated, key.clone(), Some(prev));
                    }
                    self.fire_reclaim(reclaimed);
                    return value;
                }
            }
        }
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
        let reclaimed = self.reclaim(&mut st, nw, key, now);
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
struct ReadDeadlineUpdate {
    new_deadline: Option<Instant>,
    superseded_finite: bool,
}

/// Apply the read hook to `e.deadline` in place. The returned
/// [`ReadDeadlineUpdate`] tells the caller whether to push a fresh tuple
/// and whether that push leaves a stale tuple behind.
fn update_read_deadline<K, V, P>(
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
    let new = if d >= crate::policy::TIME_UNBOUNDED / 2 {
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
mod size_regressions {
    use super::*;

    /// Lock the niche-packed [`SlotKey`] size. `slotmap::KeyData` packs an
    /// index `u32` plus a non-zero version `u32`, so `SlotKey` is one
    /// machine word and `Option<SlotKey>` is the same size (the version's
    /// niche absorbs the discriminant). If this fires, either `slotmap`
    /// changed its key layout or a custom key wrapper lost the niche.
    #[test]
    fn option_slot_key_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<SlotKey>(), 8);
        assert_eq!(std::mem::size_of::<Option<SlotKey>>(), 8);
    }

    /// Lock the `EntrySlot` shape for a representative `<String, String>`
    /// cache against an upper bound rather than an exact byte count, so
    /// the test stays useful across platforms (`Instant` differs in size
    /// between macOS arm64 and `x86_64` / Linux). The load-bearing claim
    /// is "we did not accidentally double-bump a field width or lose the
    /// niche on `Option<SlotKey>`".
    #[test]
    fn entry_slot_size_within_budget() {
        // Two `String`s (48) + `CacheEntry<String>` worst-case 64 +
        // `weight` 8 + two `Option<SlotKey>` 16 (niche) +
        // `queue` 1 + `freq` 1 + `refresh_inflight` 1 (rounded to
        // alignment) gives a platform-portable upper bound. The +8 vs
        // the pre-S3-FIFO 144 is the tail padding around the three new
        // small fields.
        const BUDGET: usize = 152;
        let observed = std::mem::size_of::<EntrySlot<String, String>>();
        assert!(
            observed <= BUDGET,
            "EntrySlot grew to {observed} bytes (budget {BUDGET}); \
             check field widths and Option<SlotKey> niche packing",
        );
    }
}

#[cfg(test)]
mod shard_tests {
    use std::sync::Arc;

    use super::*;
    use crate::config::{CacheBuilder, DynPolicy};
    use crate::policy::{DefaultPolicy, Weight};

    type K = String;
    type V = String;

    fn build(shards: usize, max_weight: u64) -> Arc<Inner<K, V, DynPolicy<K, V>>> {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        let cfg = CacheBuilder::<K, V>::new()
            .max_weight(Weight::new(max_weight))
            .shards(shards)
            .policy(policy)
            .build();
        Arc::new(Inner::new(cfg))
    }

    /// `shard_for` derives the shard purely from `&self.hasher` and
    /// `&self.shards`, both of which live on `Inner` and are therefore
    /// shared by every clone. Lock identity is the simplest cross-clone
    /// stability check: same key on two `Arc<Inner>` clones must hand
    /// back the *same* `Mutex` reference.
    #[test]
    fn shard_for_is_stable_across_clones() {
        let inner = build(8, 64);
        let clone = Arc::clone(&inner);
        for k in ["alpha", "beta", "gamma", "delta", "omega"] {
            let lhs = std::ptr::from_ref(inner.shard_for(k));
            let rhs = std::ptr::from_ref(clone.shard_for(k));
            assert!(
                std::ptr::eq(lhs, rhs),
                "key {k} mapped to different shards across clones",
            );
        }
    }

    /// With `shards == 1` the local cap equals the global cap exactly,
    /// so the standard "weight never exceeds `max_weight`" invariant
    /// holds without rounding slack. Documents the boundary the test
    /// fixups above lean on.
    #[test]
    fn single_shard_strictly_caps_total_weight() {
        let inner = build(1, 4);
        for i in 0..32 {
            inner.insert(format!("k{i}"), "v".to_string());
            assert!(
                inner.weight() <= 4,
                "weight {} exceeded max=4 with shards=1 after inserting k{i}",
                inner.weight(),
            );
        }
    }

    /// The plan documents that with `shards > 1` the observed total
    /// weight may exceed `max_weight` by up to `shards - 1` units of
    /// rounding slack (each shard rounds its share up). Lock the bound
    /// with a stress insert so a future change to per-shard accounting
    /// can't quietly blow past it.
    #[test]
    fn multi_shard_total_weight_bounded_by_max_plus_slack() {
        let shards: usize = 8;
        let max_weight: u64 = 17;
        let inner = build(shards, max_weight);
        let upper = max_weight + shards as u64 - 1;
        for i in 0..1024 {
            inner.insert(format!("key-{i}"), "v".to_string());
            let w = inner.weight();
            assert!(
                w <= upper,
                "weight {w} exceeded documented upper bound {upper} after k{i}",
            );
        }
    }

    /// Read-extending policies (e.g. `IdleTimeout`) push a fresh
    /// deadline tuple per read but cannot remove the old one cheaply, so
    /// without a rebuild trigger the heap grows without bound under
    /// repeated reads. The trigger fires once stale tuples exceed live
    /// slab size, capping the heap at one tuple per live entry plus the
    /// transient batch since the last rebuild.
    #[test]
    fn heap_stays_bounded_under_read_extending_policy() {
        use std::time::Duration;

        use crate::policy::{IdleTimeout, Weight};

        let policy: DynPolicy<K, V> = Arc::new(IdleTimeout::new(
            Duration::from_mins(1),
            crate::policy::DefaultPolicy,
        ));
        let cfg = CacheBuilder::<K, V>::new()
            .max_weight(Weight::new(8))
            .shards(1)
            .policy(policy)
            .build();
        let inner = Arc::new(Inner::new(cfg));

        for i in 0..8 {
            inner.insert(format!("k{i}"), "v".to_string());
        }

        // Without the rebuild trigger, each read pushes a fresh heap
        // tuple and leaves the previous one stale; we'd see the heap
        // grow without bound. With the trigger, the heap stays at most
        // a constant multiple of the live slab size.
        for _ in 0..1024 {
            for i in 0..8 {
                let _ = inner.get_served(&format!("k{i}"));
            }
        }

        let st = inner.shards[0].write();
        let live = st.slab.len();
        let heap_len = st.expiry.0.len();
        assert!(
            heap_len <= 2 * live + 1,
            "expiry heap grew to {heap_len} for {live} live entries; rebuild trigger broken",
        );
    }

    /// Per-shard eviction must not leak across shards: filling one shard
    /// hard should not knock entries out of another shard. We exercise
    /// this without naming shards by inserting many keys that all map
    /// to a single shard (via `shard_for`-driven probing) and asserting
    /// every other shard's slab is untouched.
    #[test]
    fn eviction_stays_within_the_owning_shard() {
        let inner = build(4, 16);
        let target_shard = std::ptr::from_ref(inner.shard_for(&"anchor".to_string()));

        // Seed every other shard with a sentinel so we can confirm they
        // survive the storm aimed at `target_shard`.
        let mut sentinels: Vec<String> = Vec::new();
        let mut tries = 0u32;
        while sentinels.len() < 3 && tries < 10_000 {
            let candidate = format!("sentinel-{tries}");
            if !std::ptr::eq(
                std::ptr::from_ref(inner.shard_for(&candidate)),
                target_shard,
            ) {
                inner.insert(candidate.clone(), "x".into());
                sentinels.push(candidate);
            }
            tries += 1;
        }
        assert_eq!(
            sentinels.len(),
            3,
            "could not find one sentinel per non-target shard; check hasher distribution"
        );

        let mut storm = 0u32;
        let mut placed = 0u32;
        while placed < 64 && storm < 100_000 {
            let candidate = format!("hot-{storm}");
            if std::ptr::eq(
                std::ptr::from_ref(inner.shard_for(&candidate)),
                target_shard,
            ) {
                inner.insert(candidate, "y".into());
                placed += 1;
            }
            storm += 1;
        }

        for s in &sentinels {
            assert!(
                inner.contains_key(s),
                "sentinel {s} was evicted by storm in another shard",
            );
        }
    }
}

#[cfg(test)]
mod s3_fifo_tests {
    //! White-box S3-FIFO invariants that need to peek at internal
    //! state (`EntrySlot::queue`, `State::small_weight`,
    //! `State::ghost`). The black-box behaviours observable through
    //! the public `Cache` API live in `tests/cache_ops.rs::s3_fifo`.

    use std::sync::Arc;

    use super::*;
    use crate::config::{CacheBuilder, DynPolicy};
    use crate::policy::{DefaultPolicy, Weight};

    type K = String;
    type V = String;

    fn build(shards: usize, max_weight: u64) -> Arc<Inner<K, V, DynPolicy<K, V>>> {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        let cfg = CacheBuilder::<K, V>::new()
            .max_weight(Weight::new(max_weight))
            .shards(shards)
            .policy(policy)
            .build();
        Arc::new(Inner::new(cfg))
    }

    fn queue_of<P>(inner: &Inner<K, V, P>, key: &K) -> Option<Queue>
    where
        P: Policy<K, V>,
    {
        let st = inner.shard_for(key).read();
        let id = entries_find(&st.entries, &inner.hasher, &st.slab, key)?;
        st.slab.get(id).map(|es| es.queue)
    }

    /// A key evicted from Small with `freq <= 1` lands in the ghost
    /// set; reinserting it before its fingerprint ages out routes the
    /// new slot directly into Main.
    #[test]
    fn ghost_hit_lands_in_main() {
        let inner = build(1, 2);
        // Fill Small to capacity with two cold (freq=0) keys.
        inner.insert("a".into(), "va".into());
        inner.insert("b".into(), "vb".into());
        // Push `a` out: small_max=0 means the next over-budget insert
        // drains Small first; tail=a (freq=0) → evict + ghost.insert(fp_a).
        inner.insert("c".into(), "vc".into());
        assert!(
            inner.get_served("a").is_none(),
            "a should have been evicted to seed ghost",
        );
        assert_eq!(
            queue_of(&inner, &"b".to_string()),
            Some(Queue::Small),
            "b stays in Small (it was the head, not the evicted tail)",
        );
        // Reinsert `a` while its fingerprint is still in ghost. The
        // install path consumes ghost membership and routes a into Main.
        inner.insert("a".into(), "va2".into());
        assert_eq!(
            queue_of(&inner, &"a".to_string()),
            Some(Queue::Main),
            "ghost-hit on insert must route the entry into Main",
        );
    }

    /// `Cache::clear` empties the slab, the FIFOs, the ghost set, and
    /// resets `small_weight` so the next install starts from a pristine
    /// Small-budget accounting baseline. Without the ghost reset, a
    /// fingerprint that survived clear would mis-route a new key into
    /// Main on the next reinsert; without the `small_weight` reset, the
    /// next reclaim would mis-balance Small vs Main.
    #[test]
    fn clear_resets_ghost_and_small_weight() {
        let inner = build(1, 2);
        inner.insert("a".into(), "v".into());
        inner.insert("b".into(), "v".into());
        // Force a Small eviction so `a`'s fingerprint enters ghost and
        // `small_weight` advances.
        inner.insert("c".into(), "v".into());
        {
            let st = inner.shards[0].read();
            assert!(st.small_weight > 0);
        }

        inner.clear();

        {
            let st = inner.shards[0].read();
            assert_eq!(st.small_weight, 0, "small_weight must reset on clear");
            assert!(st.small.tail().is_none(), "small must be empty after clear");
            assert!(st.main.tail().is_none(), "main must be empty after clear");
        }
        // Ghost-cleared sanity: reinserting `a` must land it in Small,
        // not Main (which would happen if its fingerprint had survived
        // clear).
        inner.insert("a".into(), "v".into());
        assert_eq!(
            queue_of(&inner, &"a".to_string()),
            Some(Queue::Small),
            "ghost must have been cleared so reinsert routes to Small",
        );
    }
}
