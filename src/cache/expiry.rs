//! Time-based eviction primitives.
//!
//! Per shard the cache keeps deadlines in [`ExpiryBuckets`]: a sorted
//! map from deadline tick to an intrusive doubly-linked list of slots
//! expiring at that tick. List links live on `EntrySlot`
//! (`expiry_prev` / `expiry_next`), so the structure adds no separate
//! per-slot allocation; the `BTreeMap` only carries one head/tail pair
//! per *distinct* deadline.
//!
//! No stale tuples accumulate: every state-changing op
//! ([`push`], [`unlink`], [`reschedule`]) updates the bucket
//! immediately. [`reschedule`] early-exits when the new deadline
//! rounds to the same tick as the old, which is the common case under
//! read-extending policies (e.g. `IdleTimeout`) sampling reads inside
//! one of the cache's 250 ms ticks.

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

/// Drain entries whose deadline is at or before `now_tick`. `skip_key`,
/// when set, suppresses eviction of that key (the caller is about to
/// replace the slot and emit its own lifecycle event).
pub(super) fn drain_expired<K, V, H>(
    st: &mut State<K, V>,
    hasher: &H,
    now_tick: Tick,
    skip_key: Option<&K>,
) -> Vec<(K, V)>
where
    K: Eq + Hash,
    H: BuildHasher,
{
    let mut out = Vec::new();
    while let Some((&tick, _)) = st.expiry.by_tick.first_key_value() {
        if tick > now_tick {
            break;
        }
        // Lift the bucket out of the map up front. Per-slot `unlink`
        // calls inside `evict_slot` then no-op on the bucket lookup,
        // confining the bookkeeping to the slab fields.
        let bucket = st.expiry.by_tick.remove(&tick).expect("first_key_value just yielded this tick");
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
            }
            cursor = next;
        }
    }
    out
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
