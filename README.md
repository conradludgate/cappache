# clache

A concurrent in-memory cache for Rust. Sharded under the hood, S3-FIFO
eviction, sync and async miss-coalescing, caller-driven background
refresh, all behind a `#![deny(unsafe_code)]` boundary.

- **Sync core**: `get`, `insert`, `remove`, `contains_key`, `len`,
  `weight`, `clear`.
- **Sync miss-coalescing**: `get_with_blocking` / `try_get_with_blocking`
  run the user closure once across concurrent callers on the same key
  without needing an async runtime.
- **Async miss-coalescing**: `try_get_with` (`Result<V, E>`) and
  `get_with` (infallible) park concurrent followers on the same
  in-flight leader.
- **Background refresh**: `try_revalidate` / `revalidate` schedule a
  `RefreshWork` future on a caller-supplied `Executor`; the per-key
  refresh-in-flight flag suppresses duplicates.
- **Time and weight via `Policy`**: TTL, idle timeout, custom
  per-entry weighting, all composable. `TtlFromWrite` and
  `IdleTimeout` ship in-tree; bring your own for richer rules.
- **Swappable `BuildHasher`**: defaults to `foldhash::fast::RandomState`;
  `.hasher(ahash::RandomState::default())` or
  `.hasher(xxhash_rust::xxh3::Xxh3Builder::new())` works the same way.
- **Lifecycle observability**: a `LifecycleListener` callback fires for
  `EvictedSize` / `TimeExpired` / `Removed` / `Updated`.

## Install

`clache` is not yet on crates.io. Pull it as a git dependency:

```toml
[dependencies]
clache = { git = "https://github.com/conradludgate/clache.git" }
```

MSRV: Rust **1.95**.

## Quick start

A 30-second tour. Build a cache, insert something, read it back, watch
an eviction fire.

```rust
use std::sync::Arc;

use clache::{Cache, CacheBuilder, LifecycleListener, LifecycleReason, Weight};

struct StdoutListener;
impl LifecycleListener<String, String> for StdoutListener {
    fn on_event(&self, reason: LifecycleReason, key: String, value: Option<String>) {
        println!("[{reason:?}] {key} -> {value:?}");
    }
}

fn main() {
    let cache: Cache<String, String> = Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(4))
            .lifecycle_listener(Arc::new(StdoutListener))
            .build(),
    );

    for k in ["a", "b", "c", "d", "e", "f"] {
        cache.insert(k.into(), format!("value-{k}"));
    }

    // Some entries have been evicted to keep the cache under
    // `max_weight`; others remain. Exact identities depend on which
    // shard each key hashes into.
    assert!(cache.len() <= 4 + cache.weight() as usize);
}
```

`Cache` is `Clone`: cloning yields another handle on the same shared
state, so spawning workers around a single cache is the intended
pattern. The cache is dropped when the last handle is dropped.

## How-to

Each recipe below has a complete runnable counterpart under
[`examples/`](examples/). Start with the recipe, run the example.

### Memoize an async lookup

`try_get_with` runs the user `init` closure once across any number of
concurrent callers on the same missing key:

```rust
let value = cache
    .try_get_with(user_id, || async move { fetch_user(user_id).await })
    .await?;
```

If `init` returns `Err`, every follower on the flight observes the same
error, the lane clears, and the next call re-runs `init` (errors are
*not* cached). The infallible variant is `get_with`.

Full example: [`examples/miss_coalescing.rs`](examples/miss_coalescing.rs).

### Cap entries by TTL or idle timeout

Time-based eviction comes from your `Policy`, not a separate cache
flag. `TtlFromWrite` and `IdleTimeout` wrap an inner policy and apply
`.min()` to the inner deadline so the strictest layer wins:

```rust
use std::time::Duration;
use clache::policy::{IdleTimeout, TtlFromWrite, DefaultPolicy};

let policy = IdleTimeout::new(
    Duration::from_secs(60),
    TtlFromWrite::new(Duration::from_secs(600), DefaultPolicy),
);
let cache = Cache::new(
    CacheBuilder::new().max_weight(Weight::new(1024)).policy(policy).build(),
);
```

Time eviction is opportunistic: a deadline is checked on the next
access, and reclaim drains past-deadline entries before installing a
new one. There is no background sweeper thread.

Full example (with a custom `Policy`):
[`examples/time_policies.rs`](examples/time_policies.rs).

### Keep a hot key fresh in the background

`revalidate` (and the fallible sibling `try_revalidate`) hands a
`RefreshWork` future to the caller's `Executor` and returns immediately:

```rust
use std::future::Future;
use clache::Executor;

#[derive(Clone, Copy)]
struct TokioSpawn;
impl<F: Future<Output = ()> + Send + 'static> Executor<F> for TokioSpawn {
    fn execute(&self, fut: F) { tokio::spawn(fut); }
}

let status = cache.revalidate(key, &TokioSpawn, |k| {
    let k = k.clone();
    async move { fetch_fresh(&k).await }
});
```

The returned `RevalidateScheduled` distinguishes `Submitted`,
`SkippedInFlight` (a refresh is already running for that key), and
`AbsentKey` (no `Ready` entry to refresh — refresh deliberately does
not promote into an install).

Full example: [`examples/refresh.rs`](examples/refresh.rs).

### Observe evictions

Pass an `Arc<dyn LifecycleListener<K, V>>` to the builder:

```rust
let cache = Cache::new(
    CacheBuilder::new()
        .max_weight(Weight::new(64))
        .lifecycle_listener(Arc::new(StdoutListener))
        .build(),
);
```

Reasons: `EvictedSize` (capacity reclaim), `TimeExpired` (deadline
fired), `Removed` (explicit `remove` / `clear`), `Updated` (the prior
value when a key is replaced).

Full example: [`examples/basics.rs`](examples/basics.rs).

### Swap the hasher

`.hasher(...)` accepts any `BuildHasher + Clone + Send + Sync + 'static`
and propagates the change into the type so the cache calls the new
hasher directly without a vtable:

```rust
let cache = Cache::new(
    CacheBuilder::new()
        .max_weight(Weight::new(1024))
        .hasher(ahash::RandomState::default())
        .build(),
);
```

The default is `foldhash::fast::RandomState`. `xxhash_rust::xxh3::Xxh3Builder`
and `ahash::RandomState` are the other common picks.

## Reference

API documentation: `cargo doc --open --no-deps`.

| Surface | Where |
| --- | --- |
| Public API | `Cache`, `CacheBuilder`, `Weight`, `Policy`, `LifecycleListener`, `Executor` re-exported from the crate root |
| Built-in policies | `clache::policy::{DefaultPolicy, TtlFromWrite, IdleTimeout}` |
| Refresh executor | `clache::rt::{Executor, RevalidateScheduled}` |
| Test suites | `cargo test --test cache_surface` (builder + introspection), `cargo test --test cache_ops` (data paths, concurrency, refresh) |
| Microbenchmarks | `cargo bench --bench compare_cache` (vs `moka`, `quick_cache`), `cargo bench --bench concurrent` |

## Design

The engine lives under [`src/cache/`](src/cache/), split by concern:

| Module | Responsibility |
| --- | --- |
| [`inner.rs`](src/cache/inner.rs) | `Inner` + `State`; install / lookup / clear / insert / remove |
| [`slab.rs`](src/cache/slab.rs) | `EntrySlab`, intrusive `Fifo`, `GhostSet` (S3-FIFO storage) |
| [`expiry.rs`](src/cache/expiry.rs) | Min-heap of deadlines with lazy validation on pop |
| [`flight.rs`](src/cache/flight.rs) | Miss-coalescing lanes (`MissFlight`, `SyncMissFlight`) |
| [`reclaim.rs`](src/cache/reclaim.rs) | S3-FIFO size-based eviction |
| [`miss.rs`](src/cache/miss.rs) | `try_get_with` / `try_get_with_blocking` orchestration |

A few choices worth knowing about:

- **Sharded `RwLock<State>`** via [`clashcore::ClashCollection`].
  Default shard count tracks `available_parallelism()` rounded up to a
  power of two; the minimum is `2` (`clashcore`'s precondition). The
  global weight cap is honoured per shard, so the observed total may
  overshoot by up to `shards − 1` units of rounding slack.

[`clashcore::ClashCollection`]: https://docs.rs/clashcore/latest/clashcore/sharded/struct.ClashCollection.html
- **S3-FIFO eviction** (Yang et al., SOSP'23). Each shard keeps a
  Small queue, a Main queue, and a bounded ghost set of fingerprints.
  Cold tail evictions seed the ghost set so a returning key skips
  Small and re-enters at Main. Keeps reclaim work O(1) per victim and
  beats classic LRU on most ARC / OLTP / search traces.
- **Lazy deadline maintenance**. A min-heap of `(deadline, slot)`
  tuples is validated on pop: a tuple is honoured only if the slot's
  current deadline still matches. Read-extending policies push a fresh
  tuple instead of mutating in place; once stale tuples exceed live
  entries, the heap is rebuilt from scratch so memory stays bounded.
- **Lock never crosses `.await`**. Async miss leaders run their
  `init` future with the shard lock released; followers park on a
  `tokio::sync::watch` (async) or `(Mutex, Condvar)` (sync) flight.
- **Deferred drops**. Heavy `Drop` impls fire after the shard lock is
  released, so a cache holding `Arc<LargeThing>` does not stall other
  keys when an entry leaves.

## Project status

Pre-1.0; expect breaking changes between minor versions until the API
is reviewed against real-world workloads. The crate is `#![deny(unsafe_code)]`
and tests cover builder validation, all data paths, single-flight
correctness under concurrency, refresh scheduling, and white-box
S3-FIFO routing decisions.
