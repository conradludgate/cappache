//! Time-based eviction primitives.
//!
//! Min-heap of `(deadline, slot)` tuples; tuples are validated lazily on
//! pop so stale ones (recycled slot, deadline shifted by a read hook)
//! self-discard. [`expiry_push_extend`] tracks a `stale_pushes` counter
//! and rebuilds the heap from the slab once stale pushes exceed live
//! entries — keeps memory bounded under read-extending policies.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hash::{BuildHasher, Hash};

use slotmap::Key;

use super::clock::{NO_DEADLINE, Tick};
use super::inner::{State, evict_slot};
use super::slab::SlotKey;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct HeapKey {
    pub(super) deadline: Tick,
    pub(super) slot: SlotKey,
}

impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Slotmap version is the stale-tuple gate on pop; the FFI bytes
        // here only break ties to give `Ord` a total order.
        self.deadline
            .cmp(&other.deadline)
            .then_with(|| self.slot.data().as_ffi().cmp(&other.slot.data().as_ffi()))
    }
}

impl PartialOrd for HeapKey {
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

/// Push a deadline and trigger a rebuild once stale pushes outnumber
/// live entries. `superseded_finite = true` means the previous tuple
/// for this slot is now stale.
pub(super) fn expiry_push_extend<K, V>(
    st: &mut State<K, V>,
    deadline: Tick,
    slot: SlotKey,
    superseded_finite: bool,
) {
    st.expiry.push(deadline, slot);
    if superseded_finite {
        st.stale_pushes = st.stale_pushes.saturating_add(1);
        if st.stale_pushes > st.slab.len() as u64 {
            rebuild_expiry(st);
        }
    }
}

fn rebuild_expiry<K, V>(st: &mut State<K, V>) {
    st.expiry.clear();
    for (id, es) in &st.slab {
        if es.entry.deadline != NO_DEADLINE {
            st.expiry.push(es.entry.deadline, id);
        }
    }
    st.stale_pushes = 0;
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
    while let Some(top) = st.expiry.peek() {
        if top.deadline > now_tick {
            break;
        }
        st.expiry.pop();
        let Some(es) = st.slab.get(top.slot) else {
            continue;
        };
        // Read hook superseded this tuple — drop it, the live slot has
        // a fresh tuple in the heap already.
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
