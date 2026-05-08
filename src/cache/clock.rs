//! Compact time storage. Each [`Inner`] keeps a single `start: Instant`
//! and records per-slot timestamps as `u32` ticks (one tick = 250 ms)
//! relative to it. `u32` × 250 ms covers ~34 years of uptime; deadlines
//! round *up* (a sub-tick TTL is honoured for at least its requested
//! duration), `now`/`written_at` round *down*. Conversion back to
//! [`Instant`] happens only at the [`Policy`] hook boundary.
//!
//! [`Inner`]: super::inner::Inner
//! [`Policy`]: crate::policy::Policy

use std::time::{Duration, Instant};

pub(super) type Tick = u32;

/// "No time-based expiry". Picking the top of the range collapses the
/// live-check to a single `now < e.deadline` comparison.
pub(super) const NO_DEADLINE: Tick = Tick::MAX;

const TICK: Duration = Duration::from_millis(250);

#[inline]
pub(super) fn floor_tick(start: Instant, t: Instant) -> Tick {
    let elapsed = t.saturating_duration_since(start);
    let ticks = elapsed.as_millis() / TICK.as_millis();
    Tick::try_from(ticks).unwrap_or(NO_DEADLINE)
}

#[inline]
pub(super) fn ceil_tick(start: Instant, t: Instant) -> Tick {
    let elapsed = t.saturating_duration_since(start);
    let ticks = elapsed.as_millis().div_ceil(TICK.as_millis());
    Tick::try_from(ticks).unwrap_or(NO_DEADLINE)
}

#[inline]
pub(super) fn tick_to_instant(start: Instant, tick: Tick) -> Instant {
    start + TICK * tick
}
