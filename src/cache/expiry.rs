//! Time-based eviction: per-shard intrusive deadline buckets.
//!
//! Each shard owns an [`ExpiryBuckets`] — a `BTreeMap<Tick,
//! BucketHead>` whose values point at intrusive doubly-linked lists
//! threaded through `EntrySlot::expiry_prev` / `expiry_next`. Sibling
//! links live on the slab entries, mirroring the same pattern used by
//! [`super::slab::FifoNode`] for the S3-FIFO queues, so the bucket
//! map is the only auxiliary allocation: one entry per *distinct*
//! deadline tick, regardless of how many entries share it.
//!
//! ## Operations
//!
//! Let `B` be the number of distinct live deadline ticks
//! (`B ≤ slab.len()`) and `E` the number of entries drained.
//!
//! | Op                    | Cost          |
//! |-----------------------|---------------|
//! | [`push`]              | `O(log B)`    |
//! | [`unlink`]            | `O(log B)`    |
//! | [`reschedule`]        | `O(log B)`, with a fast `O(1)` early-exit when the deadline rounds to the same tick. |
//! | [`drain_expired`]     | `O(E + log B)` per drained bucket. |
//!
//! ## Why intrusive buckets
//!
//! The previous design kept a `BinaryHeap<(Tick, SlotKey)>` per shard
//! with lazy validation on pop. Read-extending policies (e.g.
//! `IdleTimeout`) push a fresh tuple on every hit, leaving stale
//! tuples behind that have to be tolerated on pop or swept by a
//! periodic full rebuild. Under sustained reads on small caches the
//! rebuild cadence approached one rebuild per read.
//!
//! Buckets eliminate stale state: every op updates the `BTreeMap`
//! and the list immediately, no rebuild path. [`reschedule`] further
//! short-circuits when the new deadline rounds to the same 250 ms
//! tick as the old — the dominant case under `IdleTimeout` whenever
//! reads land inside the same tick window.
//!
//! ## Drain
//!
//! [`drain_expired`] walks `by_tick` in ascending order and lifts
//! each ready bucket out of the map up front. The per-slot [`unlink`]
//! call inside `evict_slot` then no-ops on the bucket lookup (it's
//! already gone) and only zeroes the slab-side sibling pointers,
//! avoiding the cost of patching the bucket head/tail `N` times
//! while we're tearing it down anyway.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::hash::{BuildHasher, Hash};

use super::clock::{NO_DEADLINE, Tick};
use super::inner::{State, evict_slot};
use super::slab::{EntrySlab, SlotKey};

#[derive(Default)]
pub(super) struct ExpiryBuckets {
    by_tick: BTreeMap<Tick, BucketHead>,
}

#[derive(Clone, Copy)]
struct BucketHead {
    head: SlotKey,
    tail: SlotKey,
}

impl ExpiryBuckets {
    pub(super) fn clear(&mut self) {
        self.by_tick.clear();
    }
}

/// Link `slot` at the tail of `bucket(deadline)`. Caller must ensure
/// the slot's `expiry_prev` / `expiry_next` are both `None` (fresh
/// slots from `install_ready` satisfy this; otherwise call [`unlink`]
/// first).
pub(super) fn push<K, V>(
    buckets: &mut ExpiryBuckets,
    slab: &mut EntrySlab<K, V>,
    slot: SlotKey,
    deadline: Tick,
) {
    if deadline == NO_DEADLINE {
        return;
    }
    debug_assert!(
        slab[slot].expiry_prev.is_none() && slab[slot].expiry_next.is_none(),
        "expiry::push: slot already linked into a bucket",
    );
    match buckets.by_tick.entry(deadline) {
        Entry::Vacant(v) => {
            v.insert(BucketHead {
                head: slot,
                tail: slot,
            });
        }
        Entry::Occupied(mut o) => {
            let bucket = o.get_mut();
            let old_tail = bucket.tail;
            bucket.tail = slot;
            slab[slot].expiry_prev = Some(old_tail);
            slab[old_tail].expiry_next = Some(slot);
        }
    }
}

/// Remove `slot` from `bucket(deadline)`, dropping the bucket from
/// the `BTreeMap` if it becomes empty. Safe to call even when the
/// bucket has already been lifted out of the map (drain path) or
/// when `deadline` is `NO_DEADLINE` (no-op).
pub(super) fn unlink<K, V>(
    buckets: &mut ExpiryBuckets,
    slab: &mut EntrySlab<K, V>,
    slot: SlotKey,
    deadline: Tick,
) {
    if deadline == NO_DEADLINE {
        return;
    }
    let prev = std::mem::take(&mut slab[slot].expiry_prev);
    let next = std::mem::take(&mut slab[slot].expiry_next);

    // Patch sibling pointers. `get_mut` is defensive against the drain
    // path, where neighbours may have been slab-removed already.
    if let Some(p) = prev
        && let Some(es) = slab.get_mut(p)
    {
        es.expiry_next = next;
    }
    if let Some(n) = next
        && let Some(es) = slab.get_mut(n)
    {
        es.expiry_prev = prev;
    }

    // Patch bucket head/tail. The bucket may be missing during drain,
    // which removes the BTreeMap entry up front.
    let Some(bucket) = buckets.by_tick.get_mut(&deadline) else {
        return;
    };
    match (prev, next) {
        (None, None) => {
            buckets.by_tick.remove(&deadline);
        }
        (None, Some(n)) => bucket.head = n,
        (Some(p), None) => bucket.tail = p,
        (Some(_), Some(_)) => {} // mid-list: head/tail unchanged
    }
}

/// Move `slot` from `bucket(old)` to `bucket(new)`. Early-exits when
/// `old == new` — the dominant case under read-extending policies
/// when reads land inside the same 250 ms tick.
pub(super) fn reschedule<K, V>(
    buckets: &mut ExpiryBuckets,
    slab: &mut EntrySlab<K, V>,
    slot: SlotKey,
    old: Tick,
    new: Tick,
) {
    if old == new {
        return;
    }
    unlink(buckets, slab, slot, old);
    push(buckets, slab, slot, new);
}

/// Drain entries whose deadline is at or before `now_tick` into the
/// caller-supplied buffer, stopping after `budget` evictions. Returns
/// `true` iff at least one ready entry remains undrained (the caller
/// can lift the cap with `budget = usize::MAX` to finish the work).
///
/// `skip_key`, when set, suppresses eviction of that key (the caller
/// is about to replace the slot and emit its own lifecycle event).
///
/// Mid-bucket breaks re-insert the truncated `BucketHead` so the
/// remainder of the intrusive list stays reachable from `by_tick`.
pub(super) fn drain_expired<K, V, H>(
    st: &mut State<K, V>,
    hasher: &H,
    now_tick: Tick,
    skip_key: Option<&K>,
    out: &mut Vec<(K, V)>,
    budget: usize,
) -> bool
where
    K: Eq + Hash,
    H: BuildHasher,
{
    let start = out.len();
    while let Some((&tick, _)) = st.expiry.by_tick.first_key_value() {
        if tick > now_tick {
            return false;
        }
        // Outer-loop boundary check: a previous inner walk just
        // finished a bucket and consumed the budget. Don't disturb
        // `by_tick` — there's still a ready bucket waiting.
        if out.len() - start >= budget {
            return true;
        }
        // Lift the bucket out of the map up front. Per-slot `unlink`
        // calls inside `evict_slot` then no-op on the bucket lookup,
        // confining the bookkeeping to the slab fields.
        let bucket = st
            .expiry
            .by_tick
            .remove(&tick)
            .expect("first_key_value just yielded this tick");
        let mut cursor = Some(bucket.head);
        while let Some(slot) = cursor {
            let next = st.slab.get(slot).and_then(|es| es.expiry_next);
            if let Some(skip) = skip_key
                && st.slab.get(slot).is_some_and(|es| &es.key == skip)
            {
                cursor = next;
                continue;
            }
            if let Some((k, v, _)) = evict_slot(st, hasher, slot) {
                out.push((k, v));
                if out.len() - start >= budget {
                    cursor = next;
                    if let Some(c) = cursor {
                        // Stale cross-link: when `skip_key` was visited
                        // earlier in this walk, the prior evict's
                        // `unlink` propagated `expiry_next` across the
                        // gap, leaving `slab[c].expiry_prev = Some(skip)`.
                        // The skip slot's later `evict_slot` call will
                        // fix its own bucket pointer via the
                        // `(None, Some(c))` arm of `unlink`.
                        st.slab[c].expiry_prev = None;
                        st.expiry.by_tick.insert(
                            tick,
                            BucketHead {
                                head: c,
                                tail: bucket.tail,
                            },
                        );
                        return true;
                    }
                    // Bucket drained to its end on the budget boundary;
                    // fall through and let the outer loop probe.
                    break;
                }
            }
            cursor = next;
        }
    }
    // Loop exited naturally — either by_tick is empty or the next
    // bucket is in the future. Either way, no ready entry remains.
    false
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use super::super::slab::{CacheEntry, EntrySlab, EntrySlot, Queue};
    use super::*;

    fn fresh_slot() -> EntrySlot<u32, ()> {
        EntrySlot {
            key: 0,
            entry: CacheEntry {
                value: (),
                written_at: 0,
                deadline: NO_DEADLINE,
            },
            weight: 1,
            queue: Queue::Small,
            freq: 0,
            fifo_prev: None,
            fifo_next: None,
            expiry_prev: None,
            expiry_next: None,
            refresh_inflight: false,
        }
    }

    fn add_slot(slab: &mut EntrySlab<u32, ()>, key: u32, deadline: Tick) -> SlotKey {
        let mut s = fresh_slot();
        s.key = key;
        s.entry.deadline = deadline;
        slab.insert(s)
    }

    /// Walk `bucket(tick)` head→tail and tail→head; assert the two
    /// walks agree (modulo reversal), every slot's deadline matches
    /// the bucket tick, and head/tail boundary pointers are `None`.
    /// Returns the bucket size.
    fn validate_bucket(buckets: &ExpiryBuckets, slab: &EntrySlab<u32, ()>, tick: Tick) -> usize {
        let bucket = buckets.by_tick.get(&tick).expect("bucket exists");
        let mut fwd = Vec::new();
        let mut cur = Some(bucket.head);
        while let Some(s) = cur {
            assert_eq!(
                slab[s].entry.deadline, tick,
                "slot deadline matches bucket tick"
            );
            fwd.push(s);
            cur = slab[s].expiry_next;
        }
        assert_eq!(fwd.last().copied(), Some(bucket.tail), "tail matches");
        assert_eq!(
            slab[bucket.head].expiry_prev, None,
            "head's expiry_prev is None"
        );
        assert_eq!(
            slab[bucket.tail].expiry_next, None,
            "tail's expiry_next is None"
        );
        let mut bwd = Vec::new();
        let mut cur = Some(bucket.tail);
        while let Some(s) = cur {
            bwd.push(s);
            cur = slab[s].expiry_prev;
        }
        bwd.reverse();
        assert_eq!(fwd, bwd, "forward and backward walks agree");
        fwd.len()
    }

    /// Aggregate invariants across every bucket: total size equals the
    /// oracle count of finite deadlines, each slot belongs to exactly
    /// one bucket, and `NO_DEADLINE` slots stay unlinked.
    fn validate_global(
        buckets: &ExpiryBuckets,
        slab: &EntrySlab<u32, ()>,
        oracle: &HashMap<SlotKey, Tick>,
    ) {
        let mut total = 0;
        let mut seen: BTreeSet<SlotKey> = BTreeSet::new();
        for &tick in buckets.by_tick.keys() {
            total += validate_bucket(buckets, slab, tick);
            let mut cur = Some(buckets.by_tick[&tick].head);
            while let Some(s) = cur {
                assert!(seen.insert(s), "slot {s:?} appears in multiple buckets");
                cur = slab[s].expiry_next;
            }
        }
        let live_finite = oracle.values().filter(|&&t| t != NO_DEADLINE).count();
        assert_eq!(total, live_finite, "total bucket size matches oracle");
        for (&slot, &tick) in oracle {
            if tick == NO_DEADLINE {
                assert_eq!(
                    slab[slot].expiry_prev, None,
                    "NO_DEADLINE slot must be unlinked",
                );
                assert_eq!(
                    slab[slot].expiry_next, None,
                    "NO_DEADLINE slot must be unlinked",
                );
            } else {
                assert!(
                    seen.contains(&slot),
                    "slot {slot:?} (deadline {tick}) missing from buckets"
                );
            }
        }
    }

    #[test]
    fn push_creates_singleton_bucket() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let id = add_slot(&mut slab, 0, 5);
        push(&mut buckets, &mut slab, id, 5);
        assert_eq!(buckets.by_tick.len(), 1);
        assert_eq!(validate_bucket(&buckets, &slab, 5), 1);
    }

    #[test]
    fn push_appends_in_insertion_order() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let mut ids = Vec::new();
        for k in 0..5 {
            let id = add_slot(&mut slab, k, 7);
            push(&mut buckets, &mut slab, id, 7);
            ids.push(id);
        }
        let mut walk = Vec::new();
        let mut cur = Some(buckets.by_tick[&7].head);
        while let Some(s) = cur {
            walk.push(s);
            cur = slab[s].expiry_next;
        }
        assert_eq!(walk, ids);
    }

    #[test]
    fn unlink_singleton_drops_bucket() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let id = add_slot(&mut slab, 0, 5);
        push(&mut buckets, &mut slab, id, 5);
        unlink(&mut buckets, &mut slab, id, 5);
        assert!(buckets.by_tick.is_empty());
        assert_eq!(slab[id].expiry_prev, None);
        assert_eq!(slab[id].expiry_next, None);
    }

    #[test]
    fn unlink_no_deadline_is_noop() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let id = add_slot(&mut slab, 0, NO_DEADLINE);
        unlink(&mut buckets, &mut slab, id, NO_DEADLINE);
        assert!(buckets.by_tick.is_empty());
    }

    #[test]
    fn unlink_head_promotes_next() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let a = add_slot(&mut slab, 0, 3);
        let b = add_slot(&mut slab, 1, 3);
        let c = add_slot(&mut slab, 2, 3);
        push(&mut buckets, &mut slab, a, 3);
        push(&mut buckets, &mut slab, b, 3);
        push(&mut buckets, &mut slab, c, 3);
        unlink(&mut buckets, &mut slab, a, 3);
        assert_eq!(buckets.by_tick[&3].head, b);
        assert_eq!(validate_bucket(&buckets, &slab, 3), 2);
    }

    #[test]
    fn unlink_tail_promotes_prev() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let a = add_slot(&mut slab, 0, 3);
        let b = add_slot(&mut slab, 1, 3);
        let c = add_slot(&mut slab, 2, 3);
        push(&mut buckets, &mut slab, a, 3);
        push(&mut buckets, &mut slab, b, 3);
        push(&mut buckets, &mut slab, c, 3);
        unlink(&mut buckets, &mut slab, c, 3);
        assert_eq!(buckets.by_tick[&3].tail, b);
        assert_eq!(validate_bucket(&buckets, &slab, 3), 2);
    }

    #[test]
    fn unlink_middle_leaves_head_and_tail() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let a = add_slot(&mut slab, 0, 3);
        let b = add_slot(&mut slab, 1, 3);
        let c = add_slot(&mut slab, 2, 3);
        push(&mut buckets, &mut slab, a, 3);
        push(&mut buckets, &mut slab, b, 3);
        push(&mut buckets, &mut slab, c, 3);
        unlink(&mut buckets, &mut slab, b, 3);
        assert_eq!(buckets.by_tick[&3].head, a);
        assert_eq!(buckets.by_tick[&3].tail, c);
        assert_eq!(slab[a].expiry_next, Some(c));
        assert_eq!(slab[c].expiry_prev, Some(a));
        assert_eq!(validate_bucket(&buckets, &slab, 3), 2);
    }

    #[test]
    fn reschedule_same_tick_is_noop() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let a = add_slot(&mut slab, 0, 3);
        let b = add_slot(&mut slab, 1, 3);
        push(&mut buckets, &mut slab, a, 3);
        push(&mut buckets, &mut slab, b, 3);
        let head_before = buckets.by_tick[&3].head;
        let tail_before = buckets.by_tick[&3].tail;
        reschedule(&mut buckets, &mut slab, a, 3, 3);
        assert_eq!(buckets.by_tick[&3].head, head_before);
        assert_eq!(buckets.by_tick[&3].tail, tail_before);
        assert_eq!(validate_bucket(&buckets, &slab, 3), 2);
    }

    #[test]
    fn reschedule_relocates_across_buckets() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let a = add_slot(&mut slab, 0, 3);
        push(&mut buckets, &mut slab, a, 3);
        slab[a].entry.deadline = 9;
        reschedule(&mut buckets, &mut slab, a, 3, 9);
        assert!(!buckets.by_tick.contains_key(&3), "old bucket dropped");
        assert_eq!(validate_bucket(&buckets, &slab, 9), 1);
    }

    #[test]
    fn reschedule_to_no_deadline_unlinks() {
        let mut buckets = ExpiryBuckets::default();
        let mut slab = EntrySlab::with_key();
        let a = add_slot(&mut slab, 0, 3);
        push(&mut buckets, &mut slab, a, 3);
        slab[a].entry.deadline = NO_DEADLINE;
        reschedule(&mut buckets, &mut slab, a, 3, NO_DEADLINE);
        assert!(buckets.by_tick.is_empty());
    }

    /// Random-op fuzz: drives `push` / `unlink` / `reschedule` over a
    /// small slot pool with a small tick range (forces same-bucket
    /// collisions) and asserts every invariant after each step.
    #[test]
    fn random_op_stream_preserves_invariants() {
        const SEEDS: &[u64] = &[
            0x0123_4567_89ab_cdef,
            0xdead_beef_cafe_f00d,
            0xface_b00c_0c0a_c01a,
            0x1337_d00d_5ca1_ab1e,
        ];
        const N_SLOTS: usize = 40;
        const TICK_RANGE: u32 = 8;
        const ITERS: usize = 5_000;

        for &seed in SEEDS {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut buckets = ExpiryBuckets::default();
            let mut slab: EntrySlab<u32, ()> = EntrySlab::with_key();
            let mut oracle: HashMap<SlotKey, Tick> = HashMap::new();
            let mut slots = Vec::with_capacity(N_SLOTS);

            for k in 0..u32::try_from(N_SLOTS).unwrap() {
                let dl = if rng.bool() {
                    NO_DEADLINE
                } else {
                    rng.u32(0..TICK_RANGE)
                };
                let id = add_slot(&mut slab, k, dl);
                push(&mut buckets, &mut slab, id, dl);
                oracle.insert(id, dl);
                slots.push(id);
            }
            validate_global(&buckets, &slab, &oracle);

            for _ in 0..ITERS {
                let id = slots[rng.usize(0..slots.len())];
                let cur = oracle[&id];
                match rng.u8(0..3) {
                    0 => {
                        let new = if rng.bool() {
                            NO_DEADLINE
                        } else {
                            rng.u32(0..TICK_RANGE)
                        };
                        slab[id].entry.deadline = new;
                        reschedule(&mut buckets, &mut slab, id, cur, new);
                        oracle.insert(id, new);
                    }
                    1 => {
                        unlink(&mut buckets, &mut slab, id, cur);
                        slab[id].entry.deadline = NO_DEADLINE;
                        oracle.insert(id, NO_DEADLINE);
                    }
                    _ => {
                        if cur == NO_DEADLINE {
                            let new = rng.u32(0..TICK_RANGE);
                            slab[id].entry.deadline = new;
                            push(&mut buckets, &mut slab, id, new);
                            oracle.insert(id, new);
                        }
                    }
                }
                validate_global(&buckets, &slab, &oracle);
            }
        }
    }
}

#[cfg(feature = "typesize")]
mod typesize_impls {
    use core::mem;

    use typesize::TypeSize;

    use super::{BucketHead, ExpiryBuckets, Tick};

    impl TypeSize for ExpiryBuckets {
        fn extra_size(&self) -> usize {
            // BTreeMap doesn't expose its node-array overhead. Use the
            // payload-only floor — stable across stdlib BTreeMap shape
            // changes, undercounts by a small per-node constant.
            self.by_tick.len() * (mem::size_of::<Tick>() + mem::size_of::<BucketHead>())
        }
    }
}
