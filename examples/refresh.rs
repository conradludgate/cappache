//! Background refresh with `revalidate` and a tiny Tokio-spawn
//! `Executor`. A `LifecycleListener` prints every `Updated` event so
//! the success of each refresh is visible.
//!
//! `revalidate` (and its fallible sibling `try_revalidate`) is
//! fire-and-forget: it hands a `RefreshWork` future to the supplied
//! `Executor` and returns a `RevalidateScheduled` status. While a
//! refresh is in flight for a given key, further `revalidate` calls
//! for the same key return `SkippedInFlight` so duplicate refreshes
//! never overlap. A refresh on a key with no `Ready` entry is a
//! no-op (`AbsentKey`) — clache deliberately does not promote
//! refresh into an install; that is the miss path's job.
//!
//! Run with: `cargo run --example refresh`

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use clache::{
    Cache, CacheBuilder, Executor, LifecycleListener, LifecycleReason, Weight,
};

/// Forwards every refresh job to `tokio::spawn`. The blanket impl
/// covers both `Executor<RefreshWork<…, MyError, …>>` and
/// `Executor<RefreshWork<…, Infallible, …>>`, so the same instance
/// works with both `try_revalidate` and `revalidate`.
#[derive(Clone, Copy)]
struct TokioSpawn;

impl<F> Executor<F> for TokioSpawn
where
    F: Future<Output = ()> + Send + 'static,
{
    fn execute(&self, fut: F) {
        tokio::spawn(fut);
    }
}

struct UpdatesLogger;

impl LifecycleListener<String, u64> for UpdatesLogger {
    fn on_event(&self, reason: LifecycleReason, key: String, value: Option<u64>) {
        if matches!(reason, LifecycleReason::Updated) {
            println!("  refreshed {key}: prior value was {value:?}");
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let cache: Cache<String, u64> = Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(8))
            .lifecycle_listener(Arc::new(UpdatesLogger))
            .build(),
    );

    // Seed an entry, then refresh it. The init closure receives the
    // key by reference and returns a future producing the new value.
    let counter = Arc::new(AtomicU64::new(100));
    cache.insert("temperature".into(), 100);

    let counter_a = Arc::clone(&counter);
    let status = cache.revalidate("temperature".into(), &TokioSpawn, move |_k| {
        let counter = Arc::clone(&counter_a);
        async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            counter.fetch_add(1, Ordering::Relaxed);
            counter.load(Ordering::Relaxed)
        }
    });
    println!("first revalidate:  {status:?}");

    // Second call lands while the first refresh is still running.
    // The cache's per-key `refresh_inflight` flag short-circuits it.
    let counter_b = Arc::clone(&counter);
    let status = cache.revalidate("temperature".into(), &TokioSpawn, move |_k| {
        let counter = Arc::clone(&counter_b);
        async move { counter.fetch_add(1, Ordering::Relaxed) }
    });
    println!("second revalidate: {status:?}");

    // Wait for the spawned RefreshWork to drive the init future to
    // completion, install the new value, and fire `Updated`.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(cache.get("temperature"), Some(101));

    // Refreshing a key that has no `Ready` entry is a no-op.
    let status = cache.revalidate("absent".into(), &TokioSpawn, |_k| async { 0 });
    println!("absent revalidate: {status:?}");
}
