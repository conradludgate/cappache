//! S3-FIFO storage primitives.
//!
//! - [`SlotKey`] / [`EntrySlab`]: versioned-key slab over `slotmap`.
//!   Stale heap tuples and stale `entries` rows self-clean on `slab.get`.
//! - [`EntrySlot`] / [`CacheEntry`]: the owned cache entry plus its
//!   intrusive FIFO links and S3-FIFO bookkeeping (`queue`, `freq`).
//! - [`Fifo`] / [`FifoNode`]: O(1) intrusive FIFO threaded through any
//!   slotmap-backed type that carries the `(prev, next)` pair.
//! - [`GhostSet`]: bounded fingerprint FIFO used for S3-FIFO's
//!   second-chance admission.
//!
//! Crate-private surface; consumed only from `super::inner` and
//! `super::expiry`. The orchestration that wires these into the cache
//! lives in `Inner::reclaim` and friends.
use std::collections::VecDeque;

use hashbrown::HashTable;
use slotmap::SlotMap;

slotmap::new_key_type! {
    /// Stable handle into [`EntrySlab`]. Wraps `slotmap::KeyData`
    /// (`u32` index + non-zero `u32` version), so:
    ///
    /// - the slab caps at 2^32-2 occupants, far past any RAM-bounded cache;
    /// - `Option<SlotKey>` niche-packs to 8 bytes (one machine word) via
    ///   the version field;
    /// - every `slab.get(key)` / `slab.remove(key)` validates the version
    ///   automatically, so a recycled slot never aliases a stale key.
    ///
    /// The version is what gates stale heap tuples in
    /// [`super::expiry::drain_expired`] (slot recycled → `slab.get`
    /// returns `None` → tuple is silently dropped); the deadline-equality
    /// check still gates *deadline-shifted* tuples for the same live slot.
    pub(crate) struct SlotKey;
}

/// Backing storage for every `Ready` entry in one shard. Versioned slot
/// handles let stale heap tuples self-discard without an explicit
/// generation field on [`EntrySlot`].
pub(super) type EntrySlab<K, V> = SlotMap<SlotKey, EntrySlot<K, V>>;

pub(crate) struct CacheEntry<V> {
    pub(crate) value: V,
    /// Quarter-second ticks since [`super::inner::Inner::start`] at
    /// which this entry was last installed. Floor-rounded; converted
    /// back to [`std::time::Instant`] only at the policy hook boundary.
    pub(crate) written_at: super::clock::Tick,
    /// Quarter-second ticks since `Inner::start` at which this entry
    /// stops being served (ceil-rounded), or [`super::clock::NO_DEADLINE`]
    /// if no time-based hook is active. Folded into a single `Tick`
    /// rather than `Option<Tick>` so the live-check is one comparison,
    /// not a discriminant + cmp.
    pub(crate) deadline: super::clock::Tick,
}

/// Which S3-FIFO queue a slot currently lives on. A slot is in exactly
/// one queue at a time, which is why [`EntrySlot`] only carries one
/// `(fifo_prev, fifo_next)` pair to thread it through whichever queue
/// owns it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Queue {
    Small,
    Main,
}

/// Owned cache entry held inside the slab. The `key` back-reference lets
/// eviction find and remove the corresponding `entries` map slot in O(1).
/// Slot identity / freshness lives on [`SlotKey`] (slotmap version), not
/// on a per-entry generation field.
pub(super) struct EntrySlot<K, V> {
    pub(super) key: K,
    pub(super) entry: CacheEntry<V>,
    /// Cached `Policy::weight` so eviction does not need to re-call the user policy.
    pub(super) weight: u64,
    /// Which FIFO (`Small` or `Main`) this slot is currently linked into.
    /// Read by `evict_slot` to pick which queue to unlink from and
    /// whether to debit `State::small_weight`. Set on install, mutated
    /// when `evict_from_small` promotes to Main.
    pub(super) queue: Queue,
    /// Saturating frequency counter (0..=`super::inner::FREQ_CAP`)
    /// consumed by `evict_from_small` (promote vs evict) and
    /// `evict_from_main` (recirculate vs evict). Read paths bump it
    /// without touching FIFO order. Stays as a plain `u8` (not atomic)
    /// because the read path already takes the shard write lock for
    /// deadline updates.
    pub(super) freq: u8,
    pub(super) fifo_prev: Option<SlotKey>,
    pub(super) fifo_next: Option<SlotKey>,
    /// `true` while a `try_revalidate` job for this key is in flight.
    /// Held on the slot rather than a side `HashSet<K>` to skip the
    /// per-revalidate `K::clone()` + hash + insert; cleared by
    /// `refresh_inflight_remove` (or dropped with the slot on eviction).
    pub(super) refresh_inflight: bool,
}

impl<K, V> FifoNode for EntrySlot<K, V> {
    fn fifo_prev(&self) -> Option<SlotKey> {
        self.fifo_prev
    }
    fn fifo_next(&self) -> Option<SlotKey> {
        self.fifo_next
    }
    fn set_fifo_prev(&mut self, prev: Option<SlotKey>) {
        self.fifo_prev = prev;
    }
    fn set_fifo_next(&mut self, next: Option<SlotKey>) {
        self.fifo_next = next;
    }
}

/// Intrusive FIFO link accessor. Implemented by any node held in a
/// [`SlotMap<SlotKey, T>`] that wishes to thread itself through a
/// [`Fifo`]. A node is in at most one [`Fifo`] at any time, so a single
/// `(prev, next)` pair on the node is enough to support multiple FIFOs
/// that this node may move between (`Small` ↔ `Main` in the S3-FIFO
/// reclaim path).
pub(super) trait FifoNode {
    fn fifo_prev(&self) -> Option<SlotKey>;
    fn fifo_next(&self) -> Option<SlotKey>;
    fn set_fifo_prev(&mut self, prev: Option<SlotKey>);
    fn set_fifo_next(&mut self, next: Option<SlotKey>);
}

/// Intrusive FIFO over slotmap-backed nodes. `head` is the insertion
/// end (newest), `tail` the eviction end (oldest); operations are
/// `O(1)` and never allocate.
#[derive(Default)]
pub(super) struct Fifo {
    head: Option<SlotKey>,
    tail: Option<SlotKey>,
}

impl Fifo {
    pub(super) fn push_head<T: FifoNode>(&mut self, slab: &mut SlotMap<SlotKey, T>, id: SlotKey) {
        let node = slab
            .get_mut(id)
            .expect("Fifo::push_head: slot must be occupied");
        node.set_fifo_prev(None);
        node.set_fifo_next(self.head);

        if let Some(old_head) = self.head {
            slab.get_mut(old_head)
                .expect("Fifo head must be occupied")
                .set_fifo_prev(Some(id));
        } else {
            self.tail = Some(id);
        }
        self.head = Some(id);
    }

    pub(super) fn pop_tail<T: FifoNode>(
        &mut self,
        slab: &mut SlotMap<SlotKey, T>,
    ) -> Option<SlotKey> {
        let id = self.tail?;
        self.unlink(slab, id);
        Some(id)
    }

    pub(super) fn unlink<T: FifoNode>(&mut self, slab: &mut SlotMap<SlotKey, T>, id: SlotKey) {
        let (prev, next) = {
            let n = slab.get(id).expect("Fifo::unlink: slot must be occupied");
            (n.fifo_prev(), n.fifo_next())
        };

        match prev {
            Some(p) => {
                slab.get_mut(p)
                    .expect("Fifo prev node must be occupied")
                    .set_fifo_next(next);
            }
            None => self.head = next,
        }
        match next {
            Some(n) => {
                slab.get_mut(n)
                    .expect("Fifo next node must be occupied")
                    .set_fifo_prev(prev);
            }
            None => self.tail = prev,
        }

        let n = slab
            .get_mut(id)
            .expect("Fifo::unlink: slot must be occupied");
        n.set_fifo_prev(None);
        n.set_fifo_next(None);
    }

    pub(super) fn promote_to_head<T: FifoNode>(
        &mut self,
        slab: &mut SlotMap<SlotKey, T>,
        id: SlotKey,
    ) {
        if self.head == Some(id) {
            return;
        }
        self.unlink(slab, id);
        self.push_head(slab, id);
    }

    pub(super) fn tail(&self) -> Option<SlotKey> {
        self.tail
    }
}

/// Bounded FIFO of fingerprints (already-hashed key images) used by
/// S3-FIFO as the second-chance signal: when an `evict_from_small`
/// drop happens at `freq <= 1`, the key's fingerprint goes into
/// [`GhostSet`]; if the key is reinserted before the fingerprint
/// rotates out, the install path routes the entry directly into Main
/// instead of Small.
///
/// Internally a `VecDeque<u64>` plus a `HashTable<u64>`: the deque
/// gives FIFO order, the table gives O(1) membership for `remove`.
/// `remove` is **tombstoning** — it deletes from the membership table
/// only and leaves the deque entry to be popped lazily on the next
/// `insert` overflow. That keeps `remove` O(1) at the cost of a
/// tombstone entry occupying one ring slot until it ages out, which
/// shrinks effective ghost capacity transiently under high promotion
/// rates. The trade-off is deliberate: the alternative (linear scan
/// on `remove`) is O(cap) and untenable for million-entry caches.
pub(super) struct GhostSet {
    ring: VecDeque<u64>,
    set: HashTable<u64>,
    cap: usize,
}

impl GhostSet {
    /// Constructs a ghost set bounded to `cap` fingerprints. The ring
    /// and membership table allocate lazily on first `insert`, so
    /// caches that never reach the eviction-from-Small path (no time
    /// budget pressure) carry zero ghost overhead.
    pub(super) fn new(cap: usize) -> Self {
        Self {
            ring: VecDeque::new(),
            set: HashTable::new(),
            cap,
        }
    }

    /// Records `fp` as recently-evicted-from-Small. No-op when `cap == 0`
    /// (used in tests / degenerate configs) or when `fp` is already
    /// tracked. On overflow, drops the oldest ring entry and best-effort
    /// removes it from the membership table (no-op if it was already a
    /// tombstone).
    pub(super) fn insert(&mut self, fp: u64) {
        if self.cap == 0 || self.contains(fp) {
            return;
        }
        if self.ring.len() >= self.cap
            && let Some(oldest) = self.ring.pop_front()
        {
            self.table_remove(oldest);
        }
        self.ring.push_back(fp);
        self.table_insert(fp);
    }

    /// Consumes the membership for `fp`. Returns `true` if the
    /// fingerprint was tracked (the install path then routes the
    /// reinserted entry into Main). The deque entry is left as a
    /// tombstone, see the type-level docstring.
    pub(super) fn remove(&mut self, fp: u64) -> bool {
        if !self.contains(fp) {
            return false;
        }
        self.table_remove(fp);
        true
    }

    pub(super) fn clear(&mut self) {
        self.ring.clear();
        self.set.clear();
    }

    #[cfg(test)]
    fn ring_len(&self) -> usize {
        self.ring.len()
    }

    #[cfg(test)]
    fn live_len(&self) -> usize {
        self.set.len()
    }

    fn contains(&self, fp: u64) -> bool {
        self.set.find(fp, |&entry| entry == fp).is_some()
    }

    fn table_insert(&mut self, fp: u64) {
        self.set.insert_unique(fp, fp, |&entry| entry);
    }

    fn table_remove(&mut self, fp: u64) {
        if let Ok(occ) = self.set.find_entry(fp, |&entry| entry == fp) {
            occ.remove();
        }
    }
}

#[cfg(feature = "typesize")]
mod typesize_impls {
    //! `TypeSize` impls for slab-backed types. `extra_size` only counts
    //! heap bytes beyond `mem::size_of::<Self>()`; the trait's default
    //! `get_size` adds the stack-resident size on top.

    use core::mem;

    use typesize::TypeSize;

    use super::{CacheEntry, EntrySlot, Fifo, GhostSet, Queue};

    impl TypeSize for Queue {}

    impl<V: TypeSize> TypeSize for CacheEntry<V> {
        fn extra_size(&self) -> usize {
            let Self {
                value,
                written_at,
                deadline,
            } = self;
            value.extra_size() + written_at.extra_size() + deadline.extra_size()
        }
    }

    impl<K: TypeSize, V: TypeSize> TypeSize for EntrySlot<K, V> {
        fn extra_size(&self) -> usize {
            // `SlotKey` comes from `slotmap::new_key_type!`; orphan rule
            // blocks a `TypeSize` impl so `fifo_prev` / `fifo_next` fall
            // through `_`. They're plain stack data, already counted in
            // `mem::size_of::<Self>()`.
            let Self {
                key,
                entry,
                weight,
                queue,
                freq,
                fifo_prev: _,
                fifo_next: _,
                refresh_inflight,
            } = self;
            key.extra_size()
                + entry.extra_size()
                + weight.extra_size()
                + queue.extra_size()
                + freq.extra_size()
                + refresh_inflight.extra_size()
        }
    }

    impl TypeSize for Fifo {
        fn extra_size(&self) -> usize {
            0
        }
    }

    impl TypeSize for GhostSet {
        fn extra_size(&self) -> usize {
            // VecDeque has a power-of-two capacity ring buffer; charge by
            // capacity * element size. HashTable's `allocation_size`
            // already includes its bucket array.
            self.ring.capacity() * mem::size_of::<u64>() + self.set.allocation_size()
        }
    }

    /// Bytes the slotmap allocates for `capacity` slots of element `T`.
    /// Matches `slotmap::SlotMap`'s internal `Vec<Slot<T>>` layout: each
    /// slot carries `MaybeUninit<T>` plus a `u32` version and a `u32`
    /// next-free pointer, rounded up to `align_of::<T>`.
    pub(crate) fn slot_storage_bytes<T>(capacity: usize) -> usize {
        let element = mem::size_of::<T>() + 2 * mem::size_of::<u32>();
        let align = mem::align_of::<T>().max(mem::align_of::<u32>());
        let stride = element.div_ceil(align) * align;
        capacity * stride
    }
}

#[cfg(feature = "typesize")]
pub(crate) use typesize_impls::slot_storage_bytes;

#[cfg(test)]
mod fifo_tests {
    use super::*;

    #[derive(Default)]
    struct TestNode {
        prev: Option<SlotKey>,
        next: Option<SlotKey>,
    }

    impl FifoNode for TestNode {
        fn fifo_prev(&self) -> Option<SlotKey> {
            self.prev
        }
        fn fifo_next(&self) -> Option<SlotKey> {
            self.next
        }
        fn set_fifo_prev(&mut self, p: Option<SlotKey>) {
            self.prev = p;
        }
        fn set_fifo_next(&mut self, n: Option<SlotKey>) {
            self.next = n;
        }
    }

    fn collect_tail_to_head(fifo: &Fifo, slab: &SlotMap<SlotKey, TestNode>) -> Vec<SlotKey> {
        let mut out = Vec::new();
        let mut cur = fifo.tail;
        while let Some(id) = cur {
            out.push(id);
            cur = slab.get(id).and_then(FifoNode::fifo_prev);
        }
        out
    }

    #[test]
    fn pop_tail_returns_oldest_first() {
        let mut slab: SlotMap<SlotKey, TestNode> = SlotMap::with_capacity_and_key(8);
        let mut fifo = Fifo::default();
        let a = slab.insert(TestNode::default());
        let b = slab.insert(TestNode::default());
        let c = slab.insert(TestNode::default());
        fifo.push_head(&mut slab, a);
        fifo.push_head(&mut slab, b);
        fifo.push_head(&mut slab, c);

        assert_eq!(fifo.head, Some(c));
        assert_eq!(fifo.tail, Some(a));
        assert_eq!(fifo.pop_tail(&mut slab), Some(a));
        assert_eq!(fifo.pop_tail(&mut slab), Some(b));
        assert_eq!(fifo.pop_tail(&mut slab), Some(c));
        assert_eq!(fifo.pop_tail(&mut slab), None);
        assert_eq!(fifo.head, None);
        assert_eq!(fifo.tail, None);
    }

    #[test]
    fn unlink_from_middle_preserves_neighbours() {
        let mut slab: SlotMap<SlotKey, TestNode> = SlotMap::with_capacity_and_key(8);
        let mut fifo = Fifo::default();
        let a = slab.insert(TestNode::default());
        let b = slab.insert(TestNode::default());
        let c = slab.insert(TestNode::default());
        fifo.push_head(&mut slab, a);
        fifo.push_head(&mut slab, b);
        fifo.push_head(&mut slab, c);

        fifo.unlink(&mut slab, b);
        assert_eq!(collect_tail_to_head(&fifo, &slab), vec![a, c]);
        assert_eq!(slab[b].prev, None);
        assert_eq!(slab[b].next, None);
    }

    #[test]
    fn unlink_head_and_tail_updates_endpoints() {
        let mut slab: SlotMap<SlotKey, TestNode> = SlotMap::with_capacity_and_key(8);
        let mut fifo = Fifo::default();
        let a = slab.insert(TestNode::default());
        let b = slab.insert(TestNode::default());
        fifo.push_head(&mut slab, a);
        fifo.push_head(&mut slab, b);

        fifo.unlink(&mut slab, b);
        assert_eq!(fifo.head, Some(a));
        assert_eq!(fifo.tail, Some(a));

        fifo.unlink(&mut slab, a);
        assert_eq!(fifo.head, None);
        assert_eq!(fifo.tail, None);
    }

    #[test]
    fn promote_to_head_moves_existing_node() {
        let mut slab: SlotMap<SlotKey, TestNode> = SlotMap::with_capacity_and_key(8);
        let mut fifo = Fifo::default();
        let a = slab.insert(TestNode::default());
        let b = slab.insert(TestNode::default());
        let c = slab.insert(TestNode::default());
        fifo.push_head(&mut slab, a);
        fifo.push_head(&mut slab, b);
        fifo.push_head(&mut slab, c);

        fifo.promote_to_head(&mut slab, a);
        assert_eq!(collect_tail_to_head(&fifo, &slab), vec![b, c, a]);
        assert_eq!(fifo.head, Some(a));
        assert_eq!(fifo.tail, Some(b));
    }

    #[test]
    fn promote_to_head_on_existing_head_is_noop() {
        let mut slab: SlotMap<SlotKey, TestNode> = SlotMap::with_capacity_and_key(8);
        let mut fifo = Fifo::default();
        let a = slab.insert(TestNode::default());
        let b = slab.insert(TestNode::default());
        fifo.push_head(&mut slab, a);
        fifo.push_head(&mut slab, b);

        fifo.promote_to_head(&mut slab, b);
        assert_eq!(fifo.head, Some(b));
        assert_eq!(fifo.tail, Some(a));
    }

    #[test]
    fn pop_tail_empty_is_none() {
        let mut slab: SlotMap<SlotKey, TestNode> = SlotMap::with_capacity_and_key(0);
        let mut fifo = Fifo::default();
        assert_eq!(fifo.pop_tail(&mut slab), None);
    }
}

#[cfg(test)]
mod ghost_set_tests {
    use super::*;

    #[test]
    fn insert_then_remove_returns_true_once() {
        let mut g = GhostSet::new(4);
        g.insert(42);
        assert!(g.remove(42));
        assert!(!g.remove(42));
    }

    #[test]
    fn remove_absent_returns_false() {
        let mut g = GhostSet::new(4);
        assert!(!g.remove(42));
    }

    #[test]
    fn double_insert_is_noop() {
        let mut g = GhostSet::new(4);
        g.insert(7);
        g.insert(7);
        assert_eq!(g.live_len(), 1);
        assert_eq!(g.ring_len(), 1);
    }

    #[test]
    fn overflow_drops_oldest() {
        let mut g = GhostSet::new(2);
        g.insert(1);
        g.insert(2);
        g.insert(3);
        assert!(!g.remove(1), "oldest fingerprint should have aged out");
        assert!(g.remove(2));
        assert!(g.remove(3));
    }

    #[test]
    fn tombstone_does_not_resurrect_on_overflow() {
        let mut g = GhostSet::new(2);
        g.insert(10);
        g.insert(20);
        assert!(g.remove(10));
        // Ring still holds [10 (tombstone), 20]; live = {20}.
        // Inserting 30 evicts the oldest ring entry (the tombstone) and
        // the live set must remain consistent ({20, 30}).
        g.insert(30);
        assert!(!g.remove(10));
        assert!(g.remove(20));
        assert!(g.remove(30));
    }

    #[test]
    fn zero_capacity_is_inert() {
        let mut g = GhostSet::new(0);
        g.insert(1);
        assert!(!g.remove(1));
        assert_eq!(g.live_len(), 0);
        assert_eq!(g.ring_len(), 0);
    }

    #[test]
    fn clear_empties_both_sides() {
        let mut g = GhostSet::new(4);
        g.insert(1);
        g.insert(2);
        g.clear();
        assert_eq!(g.live_len(), 0);
        assert_eq!(g.ring_len(), 0);
        assert!(!g.remove(1));
    }
}
