//! S3-FIFO size-based eviction.
//!
//! Two-phase reclaim called before installing a new entry:
//! 1. `drain_expired` over the deadline heap (delegated to
//!    `super::expiry`).
//! 2. While the projected weight exceeds the per-shard cap, drain the
//!    Small queue first when it sits over its
//!    `shard_max_weight / SMALL_DENOMINATOR` budget, otherwise drain
//!    Main. Each step either promotes / recirculates the tail slot or
//!    evicts it; cold Small evictions seed the ghost set so a quick
//!    re-insert routes the entry back into Main.
//!
//! `Reclaimed<K, V>` keeps the two victim lists separate so the caller
//! can fire the right `LifecycleReason` for each phase.
//!
//! Crate-private surface; consumed only from `super::inner`.
use std::hash::{BuildHasher, Hash};
use std::time::Instant;

use super::expiry::drain_expired;
use super::inner::{Inner, SMALL_DENOMINATOR, State, entries_find, evict_slot};
use super::slab::Queue;

/// Result of a single [`reclaim`] pass. Split per lifecycle reason so the
/// caller fires `TimeExpired` for the deadline drain and `EvictedSize`
/// for the S3-FIFO size drain that brought weight back under capacity.
pub(super) struct Reclaimed<K, V> {
    pub(super) expired: Vec<(K, V)>,
    pub(super) evicted: Vec<(K, V)>,
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
///    Main. Each [`evict_from_small`] either promotes the tail slot to
///    Main (when its `freq > 1`) or evicts + records its fingerprint
///    in `ghost`. Each [`evict_from_main`] either recirculates the
///    tail (decrementing `freq`) or evicts. The skip-key short-circuit
///    is preserved on both queues: the skip slot, when surfacing at
///    the tail, is promoted back to its current queue's head.
///
/// Returns the two victim lists separately so the caller can fire the
/// correct lifecycle reason for each phase.
pub(super) fn reclaim<K, V, P, S>(
    inner: &Inner<K, V, P, S>,
    st: &mut State<K, V>,
    add_w: u64,
    skip_key: &K,
    now: Instant,
) -> Reclaimed<K, V>
where
    K: Eq + Hash + 'static,
    V: 'static,
    S: BuildHasher,
{
    let expired = drain_expired(st, &inner.hasher, now, Some(skip_key));

    // Hoist the existing-slot weight lookup out of the loop: the
    // skip_key short-circuit means its slot is never reclaimed in this
    // pass, so its weight is invariant across iterations. Only
    // `state.total_weight` changes per iteration; the projection then
    // becomes a couple of arithmetic ops instead of a HashTable probe.
    let displaced_weight = entries_find(&st.entries, &inner.hasher, &st.slab, skip_key)
        .and_then(|id| st.slab.get(id))
        .map_or(0, |es| es.weight);

    let max = inner.shard_max_weight;
    let small_max = max / SMALL_DENOMINATOR;
    let mut evicted = Vec::new();
    loop {
        if !projected_weight_over_capacity(st, displaced_weight, add_w, max) {
            break;
        }
        // Try the prescribed queue first; fall back to the other if
        // the prescribed one had nothing actionable. This handles
        // the transient cases where Small drained to empty mid-pass
        // (we still need to reclaim from Main) and vice versa.
        let small_first = st.small_weight > small_max;
        let primary = if small_first {
            evict_from_small(inner, st, skip_key, &mut evicted)
        } else {
            evict_from_main(inner, st, skip_key, &mut evicted)
        };
        if primary {
            continue;
        }
        let secondary = if small_first {
            evict_from_main(inner, st, skip_key, &mut evicted)
        } else {
            evict_from_small(inner, st, skip_key, &mut evicted)
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
fn evict_from_small<K, V, P, S>(
    inner: &Inner<K, V, P, S>,
    st: &mut State<K, V>,
    skip_key: &K,
    evicted: &mut Vec<(K, V)>,
) -> bool
where
    K: Eq + Hash + 'static,
    V: 'static,
    S: BuildHasher,
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
            inner.hasher.hash_one(&tail_slot.key),
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
        let Some((k, v, _)) = evict_slot(st, &inner.hasher, tail_id) else {
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
fn evict_from_main<K, V, P, S>(
    inner: &Inner<K, V, P, S>,
    st: &mut State<K, V>,
    skip_key: &K,
    evicted: &mut Vec<(K, V)>,
) -> bool
where
    K: Eq + Hash + 'static,
    V: 'static,
    S: BuildHasher,
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
        let Some((k, v, _)) = evict_slot(st, &inner.hasher, tail_id) else {
            return false;
        };
        evicted.push((k, v));
        true
    }
}

fn projected_weight_over_capacity<K, V>(
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
