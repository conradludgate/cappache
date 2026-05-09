//! S3-FIFO storage primitives: versioned slot keys, owned slab entries,
//! intrusive FIFO links, and the bounded ghost set used by S3-FIFO's
//! second-chance admission.

use std::collections::VecDeque;

use hashbrown::HashTable;
use slotmap::SlotMap;

slotmap::new_key_type! {
    /// Versioned slab handle. The version is what makes a recycled
    /// slot observable as a different identity: a stale `SlotKey`
    /// retained outside the shard lock will fail the version check
    /// inside `slab.get` and return `None` instead of resolving to
    /// whatever entry now occupies the same slot.
    pub(crate) struct SlotKey;
}

pub(super) type EntrySlab<K, V> = SlotMap<SlotKey, EntrySlot<K, V>>;

pub(crate) struct CacheEntry<V> {
    pub(crate) value: V,
    /// Quarter-second ticks since `Inner::start` (floor-rounded).
    pub(crate) written_at: super::clock::Tick,
    /// Quarter-second ticks since `Inner::start` (ceil-rounded), or
    /// [`super::clock::NO_DEADLINE`] when no time-based hook is active.
    /// Folded into a single `Tick` (not `Option<Tick>`) so the
    /// live-check is one comparison.
    pub(crate) deadline: super::clock::Tick,
}

/// A slot lives on exactly one queue at a time, which is why
/// [`EntrySlot`] only carries one `(fifo_prev, fifo_next)` pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Queue {
    Small,
    Main,
}

pub(super) struct EntrySlot<K, V> {
    /// Back-reference so eviction can remove this slot's row from the
    /// `entries` table in O(1).
    pub(super) key: K,
    pub(super) entry: CacheEntry<V>,
    /// Cached `Policy::weight`, so eviction doesn't re-call the policy.
    pub(super) weight: u64,
    pub(super) queue: Queue,
    /// Saturating freq counter (0..=`super::inner::FREQ_CAP`). Read
    /// paths bump it under the shard write lock — no atomic needed.
    pub(super) freq: u8,
    pub(super) fifo_prev: Option<SlotKey>,
    pub(super) fifo_next: Option<SlotKey>,
    /// Intrusive sibling links for the deadline-bucket list this slot
    /// belongs to. `None` for slots without a finite deadline.
    /// Maintained alongside `entry.deadline`, which identifies the
    /// owning bucket.
    pub(super) expiry_prev: Option<SlotKey>,
    pub(super) expiry_next: Option<SlotKey>,
    /// `true` while a `try_revalidate` job for this key is in flight.
    /// Held on the slot to skip a per-revalidate `K::clone() + hash +
    /// insert` into a side set.
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

pub(super) trait FifoNode {
    fn fifo_prev(&self) -> Option<SlotKey>;
    fn fifo_next(&self) -> Option<SlotKey>;
    fn set_fifo_prev(&mut self, prev: Option<SlotKey>);
    fn set_fifo_next(&mut self, next: Option<SlotKey>);
}

/// Intrusive FIFO threaded through a slotmap. `head` is newest,
/// `tail` is oldest. All ops are O(1) and never allocate.
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

/// Bounded FIFO of recently-evicted-from-Small key fingerprints.
/// S3-FIFO's second-chance signal: a returning key whose fingerprint
/// is still in `GhostSet` skips Small and re-enters at Main.
///
/// `remove` is tombstoning — it deletes from `set` only and leaves the
/// deque entry to be popped lazily on the next `insert` overflow.
/// Keeps `remove` O(1) at the cost of transient ring slack under high
/// promotion rates. The alternative (linear scan) is O(cap) and
/// untenable for million-entry caches.
pub(super) struct GhostSet {
    ring: VecDeque<u64>,
    set: HashTable<u64>,
    cap: usize,
}

impl GhostSet {
    /// Lazy: ring and table allocate on first `insert`, so caches that
    /// never evict from Small carry zero ghost overhead.
    pub(super) fn new(cap: usize) -> Self {
        Self {
            ring: VecDeque::new(),
            set: HashTable::new(),
            cap,
        }
    }

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

    /// Consumes membership for `fp`. The deque entry is left as a
    /// tombstone (see type-level docstring).
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
            // `SlotKey` is foreign (slotmap) so the four sibling-link
            // fields fall through `_`; they're stack-only and already
            // counted in `mem::size_of::<Self>()`.
            let Self {
                key,
                entry,
                weight,
                queue,
                freq,
                fifo_prev: _,
                fifo_next: _,
                expiry_prev: _,
                expiry_next: _,
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
            self.ring.capacity() * mem::size_of::<u64>() + self.set.allocation_size()
        }
    }

    /// Bytes the slotmap allocates for `capacity` slots of `T`: each
    /// slot is `MaybeUninit<T> + u32 version + u32 next-free`, padded
    /// to `align_of::<T>`.
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
