//! Time-based eviction primitives.
//!
//! - [`ExpiryHeap`] / [`HeapKey`]: min-heap of `(deadline, slot)` tuples
//!   over [`super::slab::SlotKey`]; tuples are validated lazily on pop
//!   so stale ones (recycled / deadline-shifted) self-discard.
//! - [`drain_expired`]: walks the heap up to `now`, evicting each
//!   live tuple via [`super::inner::evict_slot`].
//! - [`expiry_push_extend`] / [`rebuild_expiry`]: push helpers that
//!   keep the heap bounded by the live slab size under read-extending
//!   policies (idle TTL, custom `expire_after_read` hooks).
//!
//! Crate-private surface; consumed only from `super::inner`.
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hash::{BuildHasher, Hash};

use slotmap::Key;

use super::clock::{NO_DEADLINE, Tick};
use super::inner::{State, evict_slot};
use super::slab::SlotKey;

/// `(deadline, slot)` tuple stored in [`ExpiryHeap`]. Ordered by
/// `deadline` so a `BinaryHeap<Reverse<HeapKey>>` yields earliest
/// deadlines first; the slot's packed FFI form is a tiebreaker for
/// total ordering only — the slotmap version inside [`SlotKey`] is what
/// gates stale tuples on pop, not anything in the heap ordering.
///
/// `deadline` is a [`Tick`] (250 ms unit) offset from `Inner::start`
/// rather than an `Instant`, shrinking the per-tuple footprint relative
/// to a full `Instant` and dropping the size of the entry-side store
/// (see [`super::slab::CacheEntry::deadline`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct HeapKey {
    pub(super) deadline: Tick,
    pub(super) slot: SlotKey,
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
pub(super) struct ExpiryHeap(BinaryHeap<Reverse<HeapKey>>);

impl ExpiryHeap {
    pub(super) fn push(&mut self, deadline: Tick, slot: SlotKey) {
        self.0.push(Reverse(HeapKey { deadline, slot }));
    }

    pub(super) fn peek(&self) -> Option<HeapKey> {
        self.0.peek().map(|Reverse(k)| *k)
    }

    pub(super) fn pop(&mut self) -> Option<HeapKey> {
        self.0.pop().map(|Reverse(k)| k)
    }

    pub(super) fn clear(&mut self) {
        self.0.clear();
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.0.len()
    }
}

/// Push a deadline tuple into the expiry heap and bump `stale_pushes`
/// when the push leaves a previous (still-finite) tuple for the same slot
/// behind. When stale tuples outnumber live entries the heap is rebuilt
/// from the slab so memory stays bounded by `slab.len()`. Pure-`State`
/// helper so it's reachable from both the read and miss-probe paths.
pub(super) fn expiry_push_extend<K, V>(
    st: &mut State<K, V>,
    deadline: Tick,
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
        if es.entry.deadline != NO_DEADLINE {
            st.expiry.push(es.entry.deadline, id);
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
    while let Some(top) = st.expiry.peek() {
        if top.deadline > now_tick {
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
        if es.entry.deadline != top.deadline {
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

#[cfg(feature = "typesize")]
mod typesize_impls {
    use core::mem;

    use typesize::TypeSize;

    use super::{ExpiryHeap, HeapKey, Reverse};

    impl TypeSize for ExpiryHeap {
        fn extra_size(&self) -> usize {
            self.0.capacity() * mem::size_of::<Reverse<HeapKey>>()
        }
    }
}
