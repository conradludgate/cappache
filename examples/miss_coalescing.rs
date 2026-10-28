//! Async miss-coalescing with `try_get_with`: many concurrent callers
//! asking for the same missing key share **one** `init` invocation.
//! The example counts how many times `init` runs across 50 concurrent
//! lookups for the same key — under single-flight, that is exactly one.
//!
//! Run with: `cargo run --example miss_coalescing`

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use cappache::{Cache, CacheBuilder, Weight};

#[derive(Debug, Clone, PartialEq, Eq)]
struct NotFound;

async fn fetch(id: u64, calls: Arc<AtomicUsize>) -> Result<String, NotFound> {
    calls.fetch_add(1, Ordering::Relaxed);
    // Pretend this is an HTTP / DB lookup. The 50 ms sleep is wide
    // enough that the first 49 followers all park on the same flight
    // before the leader's init resolves.
    tokio::time::sleep(Duration::from_millis(50)).await;
    if id == 0 {
        Err(NotFound)
    } else {
        Ok(format!("user-{id}"))
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let cache: Cache<u64, String> =
        Cache::new(CacheBuilder::new().max_weight(Weight::new(64)).build());
    let calls = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..50 {
        let cache = cache.clone();
        let calls = Arc::clone(&calls);
        handles.push(tokio::spawn(async move {
            cache
                .try_get_with(42, move || fetch(42, Arc::clone(&calls)))
                .await
        }));
    }

    for h in handles {
        let v = h.await.unwrap().unwrap();
        assert_eq!(v, "user-42");
    }

    // Single-flight: prints `1`. The first `try_get_with` to land on
    // an empty slot becomes the leader and runs init; the other 49
    // park on the same flight and receive a clone of the same value.
    println!(
        "init invocations across 50 concurrent callers: {}",
        calls.load(Ordering::Relaxed)
    );

    // Errors are not cached. A failing `init` returns `Err` to every
    // follower on the same flight, then the lane is cleared so a
    // later call re-runs `init`.
    let err = cache
        .try_get_with(0, move || fetch(0, Arc::clone(&calls)))
        .await;
    assert_eq!(err, Err(NotFound));
    println!("missing key returned: {err:?}");
}
