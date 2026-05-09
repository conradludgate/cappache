//! S3-FIFO size-based eviction.
//!
//! Two-phase reclaim called before installing a new entry:
//! 1. Drain expired deadline buckets (delegated to [`drain_expired`]).
//! 2. While projected weight exceeds `shard_max_weight`, drain the
//!    Small queue while it's over its `max / SMALL_DENOMINATOR` budget,
//!    otherwise drain Main. Each Small step either promotes
//!    (`freq > 1`) or evicts and seeds the ghost set; each Main step
//!    either recirculates (`freq > 0`, decrement) or evicts.
//!
//! Skip-key short-circuit: the slot the caller is about to replace is
//! never reclaimed in this pass — it's rotated back to its queue's head.

use std::hash::{BuildHasher, Hash};
use std::time::Instant;

use super::expiry::drain_expired;
use super::inner::{
    Inner, SMALL_DENOMINATOR, State, WRITE_EXPIRY_BUDGET, entries_find, evict_slot,
};
use super::slab::Queue;

/// Push reclaim victims into `out` and return the boundary index:
/// `out[start..expired_end]` are time-expired (caller fires
/// `TimeExpired`), `out[expired_end..]` are size-evicted (caller fires
/// `EvictedSize`). `start` is the buffer's length on entry — supports
/// reusing a shared TLS buffer across nested calls.
pub(super) fn reclaim<K, V, P, S>(
    inner: &Inner<K, V, P, S>,
    st: &mut State<K, V>,
    add_w: u64,
    skip_key: &K,
    now: Instant,
    out: &mut Vec<(K, V)>,
) -> usize
where
    K: Eq + Hash + Send + 'static,
    V: Send + 'static,
    S: BuildHasher,
{
    let now_tick = super::clock::floor_tick(inner.start, now);
    let mut more_pending = drain_expired(
        st,
        &inner.hasher,
        now_tick,
        Some(skip_key),
        out,
        WRITE_EXPIRY_BUDGET,
    );
    let mut expired_end = out.len();

    // The skip slot's weight is invariant across iterations (never
    // reclaimed), so look it up once instead of per-loop.
    let displaced_weight = entries_find(&st.entries, &inner.hasher, &st.slab, skip_key)
        .and_then(|id| st.slab.get(id))
        .map_or(0, |es| es.weight);

    let max = inner.shard_max_weight;
    let small_max = max / SMALL_DENOMINATOR;
    loop {
        if !projected_weight_over_capacity(st, displaced_weight, add_w, max) {
            break;
        }
        // Lift the budget — drain remaining expired before touching
        // still-served entries. Expired entries are free to drop; size
        // eviction must always be the last resort.
        if more_pending {
            more_pending =
                drain_expired(st, &inner.hasher, now_tick, Some(skip_key), out, usize::MAX);
            expired_end = out.len();
            continue;
        }
        // Fall back to the other queue if the prescribed one had nothing
        // actionable (Small drained empty mid-pass and friends).
        let small_first = st.small_weight > small_max;
        let primary = if small_first {
            evict_from_small(inner, st, skip_key, out)
        } else {
            evict_from_main(inner, st, skip_key, out)
        };
        if primary {
            continue;
        }
        let secondary = if small_first {
            evict_from_main(inner, st, skip_key, out)
        } else {
            evict_from_small(inner, st, skip_key, out)
        };
        if !secondary {
            break;
        }
    }
    expired_end
}

/// Returns `true` when state changed (promotion / eviction / rotation).
fn evict_from_small<K, V, P, S>(
    inner: &Inner<K, V, P, S>,
    st: &mut State<K, V>,
    skip_key: &K,
    evicted: &mut Vec<(K, V)>,
) -> bool
where
    K: Eq + Hash + Send + 'static,
    V: Send + 'static,
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
        st.queue_promote_to_head(Queue::Small, tail_id);
        return true;
    }

    if freq > 1 {
        // Warm: Small → Main, freq reset.
        st.queue_unlink(Queue::Small, tail_id);
        let es = st
            .slab
            .get_mut(tail_id)
            .expect("Small tail must be occupied");
        let w = es.weight;
        es.queue = Queue::Main;
        es.freq = 0;
        st.small_weight = st.small_weight.saturating_sub(w);
        st.queue_push_head(Queue::Main, tail_id);
        true
    } else {
        // Cold: evict and seed ghost so a quick re-insert lands in Main.
        let Some((k, v, _)) = evict_slot(st, &inner.hasher, tail_id) else {
            return false;
        };
        st.ghost.insert(fp);
        evicted.push((k, v));
        true
    }
}

/// Returns `true` when state changed (recirculation / eviction / rotation).
fn evict_from_main<K, V, P, S>(
    inner: &Inner<K, V, P, S>,
    st: &mut State<K, V>,
    skip_key: &K,
    evicted: &mut Vec<(K, V)>,
) -> bool
where
    K: Eq + Hash + Send + 'static,
    V: Send + 'static,
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
        st.queue_promote_to_head(Queue::Main, tail_id);
        return true;
    }

    if freq > 0 {
        // Recirculate. Each Main slot is touched at most FREQ_CAP+1
        // times before freq=0 forces eviction, bounding the loop.
        st.queue_unlink(Queue::Main, tail_id);
        let es = st
            .slab
            .get_mut(tail_id)
            .expect("Main tail must be occupied");
        es.freq = es.freq.saturating_sub(1);
        st.queue_push_head(Queue::Main, tail_id);
        true
    } else {
        // Only Small evictions seed ghost; Main misses don't.
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
