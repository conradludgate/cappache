//! Removal and replacement signals exposed to listeners.

/// Why an entry left the cache or was superseded.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleReason {
    /// Evicted to satisfy the configured weight budget.
    EvictedSize,
    /// Removed because a time-based deadline (from [`crate::policy::Policy`]) elapsed.
    TimeExpired,
    /// Explicitly removed by API call.
    Removed,
    /// Replaced or updated in place while still logically the same key.
    Updated,
}

/// Observer hook for cache lifecycle events.
pub trait LifecycleListener<K, V>: Send + Sync {
    /// `value` is the prior value when relevant (e.g. eviction, update).
    fn on_event(&self, reason: LifecycleReason, key: K, value: Option<V>);
}
