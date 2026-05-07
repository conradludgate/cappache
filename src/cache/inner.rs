use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use clashcore::CachePadded;
use clashcore::lock::RwLock;
use hashbrown::{DefaultHashBuilder, HashMap, HashTable};

/// One cache-padded shard. Aliased so the verbose `CachePadded<RwLock<...>>`
/// stack does not need to spell itself out at every use site (and so
/// clippy stops flagging `Inner::shards` for `type_complexity`).
type Shard<K, V> = CachePadded<RwLock<State<K, V>>>;
use std::hash::BuildHasher;
use tokio::sync::watch;

use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

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

/// Per-key Condvar-backed coalescing point for the sync
/// `get_or_insert_with` path. Lives alongside the async
/// [`MissFlight<V, E>`] entries in `State::inflight` under a sentinel
/// [`TypeId`] so the same in-flight book-keeping serves both worlds
/// without depending on the tokio runtime.
pub(crate) struct SyncMissFlight<V>
where
    V: Clone + Send + Sync + 'static,
{
    state: Mutex<SyncFlightState<V>>,
    cond: Condvar,
}

enum SyncFlightState<V> {
    Pending,
    Settled(V),
    Cancelled,
}

impl<V> SyncMissFlight<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn new() -> Self {
        Self {
            state: Mutex::new(SyncFlightState::Pending),
            cond: Condvar::new(),
        }
    }

    fn publish(&self, v: V) {
        let mut s = self.state.lock().expect("sync flight mutex poisoned");
        *s = SyncFlightState::Settled(v);
        self.cond.notify_all();
    }

    fn cancel(&self) {
        let mut s = self.state.lock().expect("sync flight mutex poisoned");
        if matches!(*s, SyncFlightState::Pending) {
            *s = SyncFlightState::Cancelled;
            self.cond.notify_all();
        }
    }

    /// Returns `Some(v)` once the leader publishes a value. Returns
    /// `None` when the leader cancelled (init future dropped or leader
    /// guard fired); the caller is then expected to re-enter the probe
    /// loop to either become leader or join a fresh follower.
    fn wait(&self) -> Option<V> {
        let mut s = self.state.lock().expect("sync flight mutex poisoned");
        loop {
            match &*s {
                SyncFlightState::Pending => {
                    s = self.cond.wait(s).expect("sync flight mutex poisoned");
                }
                SyncFlightState::Settled(v) => return Some(v.clone()),
                SyncFlightState::Cancelled => return None,
            }
        }
    }
}

/// Sentinel `E` used to tag sync lanes alongside the typed
/// `try_get_with` lanes. Lookups for a sync lane go through
/// [`InflightLane::downcast_sync`], not the generic `downcast::<E>` —
/// the marker exists purely so the `inflight` map can store both flight
/// kinds in the same `Vec<InflightLane<V>>` without a separate map.
struct SyncMarker;

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

fn cancel_sync<V>(a: &Arc<dyn Any + Send + Sync>)
where
    V: Clone + Send + Sync + 'static,
{
    if let Some(f) = (**a).downcast_ref::<SyncMissFlight<V>>() {
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

    fn new_sync(flight: Arc<SyncMissFlight<V>>) -> Self {
        Self {
            type_id: TypeId::of::<SyncMarker>(),
            flight: flight as Arc<dyn Any + Send + Sync>,
            cancel_fn: cancel_sync::<V>,
            _v: PhantomData,
        }
    }

    fn downcast_sync(&self) -> Option<Arc<SyncMissFlight<V>>> {
        if self.type_id != TypeId::of::<SyncMarker>() {
            return None;
        }
        Arc::clone(&self.flight)
            .downcast::<SyncMissFlight<V>>()
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
    /// Per-shard state, cache-padded to keep adjacent shards out of each
    /// other's cache lines. Single-key paths take exactly one shard's
    /// `RwLock` (always `.write()` today — the read paths still mutate
    /// LRU/expiry state); multi-key paths fold across all shards.
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
    pub(crate) hasher: DefaultHashBuilder,
    /// Per-shard weight cap: `ceil(max_weight / shards.len())`. With more
    /// than one shard the observed total weight may exceed
    /// `config.max_weight` by up to `shards.len() - 1` units of rounding
    /// slack; pass `.shards(1)` for a strict global cap.
    pub(crate) shard_max_weight: u64,
}

/// Stable handle into [`Slab`]. `NonZeroU32` caps slab population at 2^32-2,
/// which far exceeds any practical cache bounded by RAM, and lets
/// `Option<SlotId>` niche-pack into 4 bytes (matters for [`EntrySlot`]'s
/// LRU links and the slab free-list pointer that store one per ready
/// entry).
///
/// Encoding: `idx + 1`. Index 0 is represented as `NonZeroU32::new(1)`;
/// `to_index` reverses the offset. The offset is centralised in
/// [`SlotId::from_index`] / [`SlotId::to_index`] so callers never see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SlotId(NonZeroU32);

impl SlotId {
    fn from_index(idx: u32) -> Self {
        let raw = idx
            .checked_add(1)
            .expect("slab index + 1 overflows u32 (slab capacity exceeds u32::MAX - 1)");
        SlotId(NonZeroU32::new(raw).expect("idx + 1 is non-zero by construction"))
    }

    fn to_index(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Hash a key using the table's `BuildHasher`. Centralising the hash
/// derivation guarantees insert and lookup hash the same bytes the same
/// way; otherwise a `Borrow`-related mismatch would silently produce
/// missed lookups.
fn hash_key<H: BuildHasher, Q: Hash + ?Sized>(hasher: &H, key: &Q) -> u64 {
    hasher.hash_one(key)
}

/// `entries` lookup that compares keys through the slab — the table only
/// holds `SlotId` so equality lives on `EntrySlot::key`. Returns `None`
/// when no slot matches; an unmatched probe walks at most one bucket
/// chain on average.
fn entries_find<K, V, Q>(
    entries: &HashTable<SlotId>,
    hasher: &DefaultHashBuilder,
    slab: &Slab<K, V>,
    key: &Q,
) -> Option<SlotId>
where
    K: Hash + Eq + Borrow<Q>,
    Q: Hash + Eq + ?Sized,
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
fn entries_insert<K, V>(
    entries: &mut HashTable<SlotId>,
    hasher: &DefaultHashBuilder,
    slab: &Slab<K, V>,
    id: SlotId,
) where
    K: Hash + Eq,
{
    let key_hash = hash_key(
        hasher,
        &slab
            .get(id)
            .expect("entries_insert: slot must be occupied")
            .key,
    );
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
    /// Lookup table for `Ready` entries. Stores only `SlotId`; the key
    /// itself lives once on the slab as `EntrySlot::key`. Migrating from
    /// `HashMap<K, SlotId>` to `HashTable<SlotId>` removes one `K`
    /// allocation per ready entry (notable for `String`-keyed caches) and
    /// collapses the previous "find then remove" two-probe pattern into a
    /// single probe via `find_entry`.
    ///
    /// Equality during probes goes through [`Slab::get`] so the table can
    /// stay key-free; the `hasher` field below is the matching
    /// `BuildHasher` used both at insert time and at probe time so the
    /// stored hash matches what `find` recomputes.
    ///
    /// In-flight miss leaders live in [`Self::inflight`] under their own
    /// `(K, TypeId<E>)` lane so a `try_get_with::<…, E1>` call cannot
    /// deadlock waiting on a leader that will publish an unrelated `E2`.
    pub(crate) entries: HashTable<SlotId>,
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
    /// `true` while a `try_revalidate` job for this key is in flight.
    /// Held on the slot rather than a side `HashSet<K>` to skip the
    /// per-revalidate `K::clone()` + hash + insert; cleared by
    /// `refresh_inflight_remove` (or dropped with the slot on eviction).
    refresh_inflight: bool,
}

enum SlabEntry<K, V> {
    Occupied(EntrySlot<K, V>),
    /// `generation` is the generation of the **last** occupant; the next insert into
    /// this slot uses `generation.wrapping_add(1)`.
    Free {
        next_free: Option<SlotId>,
        generation: u32,
    },
}

/// Vector-backed slab with a free-list. Pure-safe (no `unsafe`).
struct Slab<K, V> {
    slots: Vec<SlabEntry<K, V>>,
    free_head: Option<SlotId>,
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

    /// Iterate `(SlotId, &EntrySlot)` over occupied slots only. Used by
    /// the expiry-heap rebuild path; ordering matches insertion index but
    /// callers don't depend on it.
    fn iter_occupied(&self) -> impl Iterator<Item = (SlotId, &EntrySlot<K, V>)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| match slot {
                SlabEntry::Occupied(es) => {
                    let id = SlotId::from_index(
                        u32::try_from(idx).expect("slab capacity exceeds u32::MAX"),
                    );
                    Some((id, es))
                }
                SlabEntry::Free { .. } => None,
            })
    }

    /// Install `entry` and return its slot id. Callers must set `entry.generation`
    /// to whatever this method returns from [`Self::next_generation_for`] beforehand
    /// — the slab does not mutate `entry` in place to keep the borrow simple.
    fn insert(&mut self, entry: EntrySlot<K, V>) -> SlotId {
        if let Some(id) = self.free_head {
            let slot = &mut self.slots[id.to_index()];
            let next_free = match slot {
                SlabEntry::Free { next_free, .. } => *next_free,
                SlabEntry::Occupied(_) => unreachable!("free list points at occupied slot"),
            };
            self.free_head = next_free;
            *slot = SlabEntry::Occupied(entry);
            self.len += 1;
            id
        } else {
            let idx = u32::try_from(self.slots.len()).expect("slab capacity exceeds u32::MAX");
            self.slots.push(SlabEntry::Occupied(entry));
            self.len += 1;
            SlotId::from_index(idx)
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
            Some(id) => {
                u32::try_from(id.to_index()).expect("slab capacity exceeds u32::MAX")
            }
            None => u32::try_from(self.slots.len()).expect("slab capacity exceeds u32::MAX"),
        }
    }

    fn remove(&mut self, id: SlotId) -> Option<EntrySlot<K, V>> {
        let slot = self.slots.get_mut(id.to_index())?;
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
        self.free_head = Some(id);
        self.len -= 1;
        Some(entry)
    }

    fn get(&self, id: SlotId) -> Option<&EntrySlot<K, V>> {
        match self.slots.get(id.to_index())? {
            SlabEntry::Occupied(e) => Some(e),
            SlabEntry::Free { .. } => None,
        }
    }

    fn get_mut(&mut self, id: SlotId) -> Option<&mut EntrySlot<K, V>> {
        match self.slots.get_mut(id.to_index())? {
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
            .then_with(|| self.slot.0.get().cmp(&other.slot.0.get()))
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

/// Push a deadline tuple into the expiry heap and bump `stale_pushes`
/// when the push leaves a previous (still-finite) tuple for the same slot
/// behind. When stale tuples outnumber live entries the heap is rebuilt
/// from the slab so memory stays bounded by `slab.len()`. Pure-`State`
/// helper so it's reachable from both the read and miss-probe paths.
fn expiry_push_extend<K, V>(
    st: &mut State<K, V>,
    deadline: Instant,
    slot: SlotId,
    generation: u32,
    superseded_finite: bool,
) {
    st.expiry.push(deadline, slot, generation);
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
    for (id, es) in st.slab.iter_occupied() {
        if let Some(dl) = es.entry.deadline {
            st.expiry.push(dl, id, es.generation);
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
fn drain_expired<K, V>(
    st: &mut State<K, V>,
    hasher: &DefaultHashBuilder,
    now: Instant,
    skip_key: Option<&K>,
) -> Vec<(K, V)>
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
        if let Some((k, v, _)) = evict_slot(st, hasher, top.slot) {
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

enum SyncMissStep<V>
where
    V: Clone + Send + Sync + 'static,
{
    Hit(V),
    Wait(Arc<SyncMissFlight<V>>),
    Leader { miss: Arc<SyncMissFlight<V>> },
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
        // `clashcore::RwLock` does not poison, so a panicking leader leaves
        // the shard in its last consistent state — we can take the write
        // lock here and clean up without the std::sync::Mutex `into_inner`
        // dance.
        let mut st = self.inner.shard_for(self.key).write();
        remove_lane(&mut st, self.key, &self.flight_any);
        drop(st);
        self.flight.cancel();
    }
}

/// RAII cleanup for the sync `get_or_insert_with` leader. Mirror of
/// [`LeaderGuard`] but parameterised over the Condvar-backed
/// [`SyncMissFlight`]; on cancel/panic it removes the lane and signals
/// any sync followers to retry.
struct SyncLeaderGuard<'a, K, V, P>
where
    K: Eq + Hash + 'static,
    V: Clone + Send + Sync + 'static,
    P: 'static,
{
    inner: &'a Inner<K, V, P>,
    key: &'a K,
    flight: Arc<SyncMissFlight<V>>,
    flight_any: Arc<dyn Any + Send + Sync>,
    armed: bool,
}

impl<K, V, P> Drop for SyncLeaderGuard<'_, K, V, P>
where
    K: Eq + Hash + 'static,
    V: Clone + Send + Sync + 'static,
    P: 'static,
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut st = self.inner.shard_for(self.key).write();
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
fn evict_slot<K, V>(
    st: &mut State<K, V>,
    hasher: &DefaultHashBuilder,
    id: SlotId,
) -> Option<(K, V, u64)>
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
    // Take the slot out first so the entries removal can compare against
    // a slab that no longer holds `id` — the probe predicate uses
    // `slab.get(id)`, which now returns `None`, so we hash by the taken
    // entry's key and match buckets by `SlotId` directly.
    let entry_slot = slab.remove(id)?;
    *total_weight = total_weight.saturating_sub(entry_slot.weight);
    let h = hash_key(hasher, &entry_slot.key);
    if let Ok(occ) = entries.find_entry(h, |&other| other == id) {
        occ.remove();
    }
    Some((entry_slot.key, entry_slot.entry.value, entry_slot.weight))
}

/// Allocate a slab slot for `key`/`entry`/`weight`, push it to MRU, register
/// it in `state.entries`, bump `total_weight`, and (if applicable) push the
/// deadline into `state.expiry`. Centralises every install site so the heap
/// invariant cannot be silently dropped.
fn install_ready<K, V>(
    st: &mut State<K, V>,
    hasher: &DefaultHashBuilder,
    key: K,
    entry: CacheEntry<V>,
    weight: u64,
) -> SlotId
where
    K: Eq + Hash,
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
        key,
        entry,
        weight,
        generation,
        lru_prev: None,
        lru_next: None,
        refresh_inflight: false,
    });
    debug_assert_eq!(
        id.to_index(),
        idx as usize,
        "slab.insert is expected to use the slot index it advertised"
    );
    evict_list.push_mru(slab, id);
    entries_insert(entries, hasher, slab, id);
    *total_weight = total_weight.saturating_add(weight);
    if let Some(dl) = deadline {
        expiry.push(dl, id, generation);
    }
    id
}

impl<K, V, P> Inner<K, V, P> {
    /// Returns the shard responsible for `key`. Single-key paths funnel
    /// through here so the shard mapping is centralised; cross-shard
    /// work (`len`, `weight`, `clear`) iterates `self.shards` directly.
    ///
    /// Kept on a bound-free impl block so that the `LeaderGuard::drop`
    /// path can call it without re-asserting the full `Inner` bounds.
    /// Borrowed verbatim from `clashcore::ClashCollection::determine_shard`:
    /// `(hash << 7) >> shift` keeps the high 7 bits of the hash for
    /// hashbrown's SIMD tag while picking a shard from the next bits.
    fn shard_for<Q>(&self, key: &Q) -> &Shard<K, V>
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
}

impl<K, V, P> Inner<K, V, P>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
    P: Policy<K, V>,
{
    pub(crate) fn new(config: CacheConfig<K, V, P>) -> Self {
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
        let shards = (0..shard_count)
            .map(|_| {
                CachePadded::new(RwLock::new(State {
                    entries: HashTable::with_capacity(initial_capacity),
                    inflight: HashMap::new(),
                    slab: Slab::with_capacity(initial_capacity),
                    evict_list: EvictList::default(),
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
        Self {
            config,
            shards,
            shard_shift,
            hasher: DefaultHashBuilder::default(),
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

        let (value, generation, update) = {
            let State {
                slab, evict_list, ..
            } = &mut *st;
            evict_list.promote_to_mru(slab, id);
            let es = slab.get_mut(id)?;
            let value = es.entry.value.clone();
            let generation = es.generation;
            // Disjoint-field borrow: `&es.key` (immutable) and `&mut es.entry`
            // (mutable) don't overlap, so the borrow checker accepts both.
            let update =
                update_read_deadline(&es.key, &mut es.entry, &self.config.policy, now);
            (value, generation, update)
        };
        if let Some(dl) = update.new_deadline {
            expiry_push_extend(&mut st, dl, id, generation, update.superseded_finite);
        }
        Some(value)
    }

    pub(crate) fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let st = self.shard_for(key).write();
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

    /// Sums per-shard `slab.len()`. Briefly takes each shard's lock in
    /// turn; not snapshot-consistent under concurrent mutation, but each
    /// shard's contribution is taken atomically.
    pub(crate) fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.write().slab.len())
            .sum()
    }

    /// Short-circuiting form of `len() == 0`: bails out on the first
    /// non-empty shard so a busy cache only pays for a single lock+probe.
    /// Inherits the non-snapshot caveat from [`Self::len`].
    pub(crate) fn is_empty(&self) -> bool {
        self.shards
            .iter()
            .all(|s| s.write().slab.len() == 0)
    }

    /// Sums per-shard `total_weight`. Same consistency caveat as
    /// [`Self::len`].
    pub(crate) fn weight(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.write().total_weight)
            .sum()
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
                evict_list,
                expiry,
                stale_pushes,
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
            *stale_pushes = 0;
            *total_weight = 0;

            // Cancel pending miss lanes before releasing the lock so
            // followers see Cancelled instead of waiting on a sender we
            // are about to drop.
            let cancelled: Vec<Vec<InflightLane<V>>> =
                inflight.drain().map(|(_, v)| v).collect();
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
            let Some((k, v, _)) = evict_slot(st, &self.hasher, tail_id) else {
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

    /// Single critical section for one `try_get_with` iteration: served hit, in-flight miss wait, or
    /// install miss leader. Avoids TOCTOU between separate `get_served` / miss checks / leader insert.
    fn miss_probe<E>(&self, key: &K) -> MissStep<V, E>
    where
        K: 'static,
        V: 'static,
        E: Clone + Send + Sync + 'static,
    {
        let now = Instant::now();
        let shard = self.shard_for(key);
        let mut st = shard.write();

        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let expired = st
                .slab
                .get(id)
                .is_some_and(|es| !entry_is_served(&es.entry, now));
            if expired
                && let Some((_, v, _)) = evict_slot(&mut st, &self.hasher, id)
            {
                drop(st);
                self.notify_lifecycle(LifecycleReason::TimeExpired, key.clone(), Some(v));
                st = shard.write();
            }
        }

        let now = Instant::now();
        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let still_served = st
                .slab
                .get(id)
                .is_some_and(|es| entry_is_served(&es.entry, now));
            if still_served {
                let (value, generation, update) = {
                    let State {
                        slab, evict_list, ..
                    } = &mut *st;
                    evict_list.promote_to_mru(slab, id);
                    let es = slab
                        .get_mut(id)
                        .expect("just confirmed Occupied above under our mutex");
                    let value = es.entry.value.clone();
                    let generation = es.generation;
                    let update =
                        update_read_deadline(key, &mut es.entry, &self.config.policy, now);
                    (value, generation, update)
                };
                if let Some(dl) = update.new_deadline {
                    expiry_push_extend(&mut st, dl, id, generation, update.superseded_finite);
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
                    let mut st = guard.inner.shard_for(&key).write();
                    remove_lane(&mut st, &key, &guard.flight_any);

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
                    guard.flight.publish(res.clone());
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

    /// Sync mirror of [`Self::miss_probe`]: returns a hit, a follower
    /// flight to wait on, or installs a fresh sync-leader lane. Sync and
    /// async flights coexist in the same `inflight` map under a sentinel
    /// `TypeId`, so a sync caller never joins an async leader (and vice
    /// versa) — both kinds run their own `init` if no compatible
    /// follower is found, matching the typed-lanes semantics.
    fn sync_miss_probe(&self, key: &K) -> SyncMissStep<V>
    where
        K: Eq + Hash + Clone,
        V: Clone + Send + Sync + 'static,
    {
        let now = Instant::now();
        let shard = self.shard_for(key);
        let mut st = shard.write();

        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let expired = st
                .slab
                .get(id)
                .is_some_and(|es| !entry_is_served(&es.entry, now));
            if expired
                && let Some((_, v, _)) = evict_slot(&mut st, &self.hasher, id)
            {
                drop(st);
                self.notify_lifecycle(LifecycleReason::TimeExpired, key.clone(), Some(v));
                st = shard.write();
            }
        }

        let now = Instant::now();
        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let still_served = st
                .slab
                .get(id)
                .is_some_and(|es| entry_is_served(&es.entry, now));
            if still_served {
                let (value, generation, update) = {
                    let State {
                        slab, evict_list, ..
                    } = &mut *st;
                    evict_list.promote_to_mru(slab, id);
                    let es = slab
                        .get_mut(id)
                        .expect("just confirmed Occupied above under our mutex");
                    let value = es.entry.value.clone();
                    let generation = es.generation;
                    let update =
                        update_read_deadline(key, &mut es.entry, &self.config.policy, now);
                    (value, generation, update)
                };
                if let Some(dl) = update.new_deadline {
                    expiry_push_extend(&mut st, dl, id, generation, update.superseded_finite);
                }
                return SyncMissStep::Hit(value);
            }
        }

        if let Some(lanes) = st.inflight.get(key)
            && let Some(lane) = lanes.iter().find_map(InflightLane::downcast_sync)
        {
            return SyncMissStep::Wait(lane);
        }

        let miss = Arc::new(SyncMissFlight::<V>::new());
        st.inflight
            .entry(key.clone())
            .or_default()
            .push(InflightLane::new_sync(Arc::clone(&miss)));
        SyncMissStep::Leader { miss }
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
            match self.sync_miss_probe(&key) {
                SyncMissStep::Hit(v) => return v,
                SyncMissStep::Wait(f) => {
                    if let Some(v) = f.wait() {
                        return v;
                    }
                    // Leader cancelled — its lane is gone; loop and either
                    // observe a fresh value or take leadership ourselves.
                }
                SyncMissStep::Leader { miss } => {
                    let init = init
                        .take()
                        .expect("init consumed: only the leader arm consumes it");
                    let flight_any: Arc<dyn Any + Send + Sync> =
                        Arc::clone(&miss) as Arc<dyn Any + Send + Sync>;
                    let mut guard = SyncLeaderGuard {
                        inner: self.as_ref(),
                        key: &key,
                        flight: miss,
                        flight_any,
                        armed: true,
                    };
                    let value = init(&key);
                    let mut st = guard.inner.shard_for(&key).write();
                    remove_lane(&mut st, &key, &guard.flight_any);
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

    /// Clears the per-slot `refresh_inflight` flag once a refresh job
    /// finishes (success, error, or future drop). Best-effort: if the slot
    /// was evicted while the refresh was in flight there is nothing to
    /// clear, which is fine — the next `try_revalidate` would land on a
    /// different slot anyway.
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
/// necessarily an entry count.
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

    /// Lock the niche-packed `Option<SlotId>` size. If this assertion ever
    /// fires, either the niche optimisation was lost (`SlotId` stopped
    /// using `NonZeroU32`) or the host pointer width changed in a way that
    /// invalidates the budget. Either case warrants a deliberate update —
    /// not an accidental regression that doubles every LRU link.
    #[test]
    fn option_slot_id_is_four_bytes() {
        assert_eq!(std::mem::size_of::<Option<SlotId>>(), 4);
        assert_eq!(std::mem::size_of::<SlotId>(), 4);
    }

    /// Lock the `EntrySlot` shape for a representative `<String, String>`
    /// cache. Composition (aarch64): `String` × 2 (48) + `CacheEntry`
    /// (`String` 24 + `Instant` 16 + `Option<Instant>` 24 = 64) + weight
    /// `u64` (8) + generation `u32` (4) + `Option<SlotId>` × 2 (8 with
    /// `NonZeroU32` niche) = 132, rounded to alignment-8 = 136. macOS
    /// arm64 currently reports 104 because `Instant` packs in 12 bytes
    /// here. Either way, the load-bearing claim is "the LRU links cost
    /// 8 bytes total, not 16" — if this assertion fires after touching
    /// `EntrySlot`, double-check `Option<SlotId>` is still niche-packed.
    #[test]
    fn entry_slot_size_locked() {
        let observed = std::mem::size_of::<EntrySlot<String, String>>();
        assert_eq!(
            observed, 104,
            "EntrySlot grew to {observed}; check niche packing on Option<SlotId>",
        );
    }
}

#[cfg(test)]
mod shard_tests {
    use super::*;
    use crate::config::{CacheBuilder, DynPolicy};
    use crate::policy::{DefaultPolicy, Weight};
    use std::sync::Arc;

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
        use crate::policy::{IdleTimeout, Weight};
        use std::time::Duration;

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
