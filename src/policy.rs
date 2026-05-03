//! Per-entry [`Policy`]: weight and how long an entry may live after reads and writes.
//!
//! The cache keeps **one** monotonic deadline per entry. Each hook returns a
//! [`Duration`] used with the event’s anchor time; use [`TIME_UNBOUNDED`] when this hook should
//! not impose a finite time bound.
//!
//! **Read path:** whether an entry may be returned is decided from **materialized** state (the
//! stored deadline) only. [`Policy::expire_after_read`] is called **after** a confirmed live hit to
//! compute the **next** deadline (see cache implementation).
//!
//! **TTL- and TTI-style helpers** ([`TtlFromWrite`], [`IdleTimeout`]) wrap a generic inner **`P:
//! Policy<K, V>`** and, on each expiry hook, return the **shorter** of the inner policy’s duration
//! and their own cap (so the effective deadline is the stricter of the two). [`Policy::weight`] is
//! delegated to the inner policy.

use std::time::{Duration, Instant};

/// Relative cost of an entry toward the cache weight budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Weight(pub u64);

impl Weight {
    pub const ZERO: Self = Self(0);

    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Duration large enough to mean “no practical time eviction” from this hook, but small enough to
/// add to typical [`Instant`] values without overflow. Equivalent to **100 × 365 × 24 h** (`876_000` hours).
pub const TIME_UNBOUNDED: Duration = Duration::from_hours(876_000);

/// Per-entry weighting and time limits after read and write events.
pub trait Policy<K, V>: Send + Sync {
    fn weight(&self, key: &K, value: &V) -> Weight;

    /// Returns how long this entry may remain after **`read_at`** (a successful read that served the value).
    ///
    /// The cache applies this **only after** confirming the entry is still live from the stored
    /// deadline; it is **not** consulted to reject the current `get`. `time_remaining` is how long
    /// that deadline allows from `read_at` (unbounded when no deadline is set).
    fn expire_after_read(
        &self,
        key: &K,
        value: &V,
        read_at: Instant,
        last_modified_at: Instant,
        time_remaining: Duration,
    ) -> Duration;

    fn expire_after_write(&self, key: &K, value: &V, written_at: Instant) -> Duration;
}

/// Unit weight and no finite time limit from either hook.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultPolicy;

impl<K, V> Policy<K, V> for DefaultPolicy {
    fn weight(&self, _key: &K, _value: &V) -> Weight {
        Weight::new(1)
    }

    fn expire_after_read(
        &self,
        _key: &K,
        _value: &V,
        _read_at: Instant,
        _last_modified_at: Instant,
        _time_remaining: Duration,
    ) -> Duration {
        TIME_UNBOUNDED
    }

    fn expire_after_write(&self, _key: &K, _value: &V, _written_at: Instant) -> Duration {
        TIME_UNBOUNDED
    }
}

/// Caps time-to-live from each **write** against an inner policy **`P`**.
///
/// On **write**, returns the **shorter** of `ttl` and `inner.expire_after_write(...)` (stricter
/// deadline wins).
///
/// On **read**, returns the **shorter** of `time_remaining` (what the cache believes is left until
/// the current deadline) and `inner.expire_after_read(...)`. That keeps the absolute deadline from
/// moving **later** than the inner policy would allow, while still allowing the cache to pass
/// through `time_remaining` when the inner policy is permissive (e.g. [`TIME_UNBOUNDED`]).
/// It is **not** `ttl.min(...)` on read: TTL is anchored at the last write; reads do not reset TTL
/// to a fresh `ttl` window.
#[derive(Debug, Clone)]
pub struct TtlFromWrite<P> {
    pub ttl: Duration,
    pub inner: P,
}

impl<P> TtlFromWrite<P> {
    #[must_use]
    pub fn new(ttl: Duration, inner: P) -> Self {
        Self { ttl, inner }
    }
}

impl<K, V, P> Policy<K, V> for TtlFromWrite<P>
where
    P: Policy<K, V> + Send + Sync,
{
    fn weight(&self, key: &K, value: &V) -> Weight {
        self.inner.weight(key, value)
    }

    fn expire_after_read(
        &self,
        key: &K,
        value: &V,
        read_at: Instant,
        last_modified_at: Instant,
        time_remaining: Duration,
    ) -> Duration {
        let inner_d =
            self.inner
                .expire_after_read(key, value, read_at, last_modified_at, time_remaining);
        time_remaining.min(inner_d)
    }

    fn expire_after_write(&self, key: &K, value: &V, written_at: Instant) -> Duration {
        let inner_d = self.inner.expire_after_write(key, value, written_at);
        self.ttl.min(inner_d)
    }
}

/// Caps time-to-idle on each **read** and **write** against an inner policy **`P`**.
///
/// Each hook returns `idle.min(inner.expire_after_*(...))`.
#[derive(Debug, Clone)]
pub struct IdleTimeout<P> {
    pub idle: Duration,
    pub inner: P,
}

impl<P> IdleTimeout<P> {
    #[must_use]
    pub fn new(idle: Duration, inner: P) -> Self {
        Self { idle, inner }
    }
}

impl<K, V, P> Policy<K, V> for IdleTimeout<P>
where
    P: Policy<K, V> + Send + Sync,
{
    fn weight(&self, key: &K, value: &V) -> Weight {
        self.inner.weight(key, value)
    }

    fn expire_after_read(
        &self,
        key: &K,
        value: &V,
        read_at: Instant,
        last_modified_at: Instant,
        time_remaining: Duration,
    ) -> Duration {
        let inner_d =
            self.inner
                .expire_after_read(key, value, read_at, last_modified_at, time_remaining);
        self.idle.min(inner_d)
    }

    fn expire_after_write(&self, key: &K, value: &V, written_at: Instant) -> Duration {
        let inner_d = self.inner.expire_after_write(key, value, written_at);
        self.idle.min(inner_d)
    }
}
