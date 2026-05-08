//! Compact time storage. Each [`Inner`] records a single `start: Instant`
//! at construction; per-slot timestamps and heap keys are stored as
//! `u32` *ticks* (one tick = 250 ms) since that monotonic anchor instead
//! of full `Instant`s. Halves the bytes [`CacheEntry`] / [`HeapKey`]
//! carry per slot relative to a `u64` representation; conversion back
//! to [`Instant`] happens only at the [`Policy`] hook boundary, so the
//! public [`Policy`] API is unaffected.
//!
//! The 250 ms granularity is an opinionated trade: caches don't need
//! sub-second precision in practice, and quarter-second buckets cover
//! ~34 years of monotonic uptime in a single `u32`. Deadlines round
//! *up* (a TTL of 50 ms is honoured for the rest of the current tick,
//! up to ~250 ms total); `now` / `written_at` round *down*.
//!
//! Crate-private; consumed only from `super::inner` / `super::expiry`.
//!
//! [`Inner`]: super::inner::Inner
//! [`CacheEntry`]: super::slab::CacheEntry
//! [`HeapKey`]: super::expiry::HeapKey
//! [`Policy`]: crate::policy::Policy

use std::time::{Duration, Instant};

/// Quarter-second ticks since the cache's `start: Instant`. `u32`
/// covers ~34 years of uptime; values saturate to [`NO_DEADLINE`]
/// on overflow rather than wrapping.
pub(super) type Tick = u32;

/// Sentinel for "no time-based expiry on this slot". Aliasing the top
/// of the range lets the live-check collapse to the single comparison
/// `now < e.deadline` without an extra `Option` branch.
pub(super) const NO_DEADLINE: Tick = Tick::MAX;

/// One tick = 250 ms. Quarter-second granularity is the opinionated
/// floor on cache-relevant timing precision; pick smaller than this
/// and almost no real cache benefits.
const TICK: Duration = Duration::from_millis(250);

/// Floor `t` into ticks since `start`. Use for "now" and `written_at`:
/// values that should refer to the tick they fall *inside*. Saturates
/// to `NO_DEADLINE` on overflow.
#[inline]
pub(super) fn floor_tick(start: Instant, t: Instant) -> Tick {
    let elapsed = t.saturating_duration_since(start);
    let ticks = elapsed.as_millis() / TICK.as_millis();
    Tick::try_from(ticks).unwrap_or(NO_DEADLINE)
}

/// Ceil `t` into ticks since `start`. Use for deadlines so an entry is
/// served through the entire tick of computed expiry (a 50 ms TTL is
/// honoured for at least 50 ms, up to ~250 ms). Saturates to
/// `NO_DEADLINE` on overflow.
#[inline]
pub(super) fn ceil_tick(start: Instant, t: Instant) -> Tick {
    let elapsed = t.saturating_duration_since(start);
    let tick_ms = TICK.as_millis();
    let ticks = elapsed.as_millis().div_ceil(tick_ms);
    Tick::try_from(ticks).unwrap_or(NO_DEADLINE)
}

/// Materialise a tick back into an `Instant` for policy hooks that need
/// one. The result is the *start* of the named tick (i.e. floor),
/// matching how `floor_tick` interprets values.
#[inline]
pub(super) fn tick_to_instant(start: Instant, tick: Tick) -> Instant {
    start + TICK * tick
}
