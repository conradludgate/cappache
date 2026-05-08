//! **Contract tests** for cache **data paths** (`get`, `insert`, miss helpers, `revalidate`, …).
//!
//! Run **`cargo test --test cache_ops`** for these checks. Use **`cargo test --test cache_surface`**
//! and **`cargo test --lib`** for builder/introspection and small compile-only tests.
//!
//! **Revalidate and data plane:** after `try_revalidate` returns `Submitted` and the executor runs
//! the `RefreshWork` job future to completion, `Cache::get` must show the refreshed value for that
//! key. Init errors may be returned from `try_revalidate` or only after the async job—see
//! `try_revalidate_initializer_error_propagates_as_err`.
//!
//! Prefer `get_with` + `.await` inside `#[tokio::test]` over `get_with_blocking` on the Tokio
//! worker when the initializer would need the same runtime (see crate rustdoc on blocking hazards).

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clache::{
    Cache, CacheBuilder, DefaultPolicy, DynPolicy, Executor, IdleTimeout, LifecycleListener,
    LifecycleReason, Policy, RevalidateScheduled, TIME_UNBOUNDED, TtlFromWrite, Weight,
};

type K = String;
type V = String;

/// `BuildHasher` that pins every key to shard 0 *while preserving per-key
/// fingerprints*. Wraps `DefaultHasher` (fixed initial state) and clears
/// the top 8 bits of `finish()` so [`clashcore::ClashCollection::determine_shard`]'s
/// `(h << 7) >> 63` always picks shard 0, but lower bits stay derived
/// from the actual key bytes. Used by tests that want single-shard
/// determinism without collapsing every ghost-set fingerprint to a
/// single value (which would break S3-FIFO's hot-vs-cold routing).
#[derive(Clone, Copy, Default)]
struct SingleShardHasher;

struct SingleShardHasherImpl(std::collections::hash_map::DefaultHasher);

impl std::hash::Hasher for SingleShardHasherImpl {
    fn finish(&self) -> u64 {
        self.0.finish() & ((1u64 << 56) - 1)
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

impl std::hash::BuildHasher for SingleShardHasher {
    type Hasher = SingleShardHasherImpl;
    fn build_hasher(&self) -> SingleShardHasherImpl {
        SingleShardHasherImpl(std::collections::hash_map::DefaultHasher::new())
    }
}

/// Type alias for the `SingleShardHasher`-backed cache used by every
/// "this test wants single-shard determinism" fixture below.
type SingleShardCache<K, V> = Cache<K, V, DynPolicy<K, V>, SingleShardHasher>;

fn test_cache() -> Cache<K, V> {
    let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
    Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(10_000))
            .policy(policy)
            .build(),
    )
}

fn test_cache_ttl_zero_write() -> Cache<K, V> {
    let policy: DynPolicy<K, V> = Arc::new(TtlFromWrite::new(Duration::ZERO, DefaultPolicy));
    Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(10_000))
            .policy(policy)
            .build(),
    )
}

fn test_cache_max_weight_one() -> SingleShardCache<K, V> {
    let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
    // `shards(2)` is the minimum the cache supports; `SingleShardHasher`
    // pins every key into shard 0, and `max_weight = 2` rounds to a
    // shard cap of 1 — same as the original `shards(1)` fixture.
    Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(2))
            .shards(2)
            .hasher(SingleShardHasher)
            .policy(policy)
            .build(),
    )
}

type ListenerEvent = (LifecycleReason, String, Option<String>);
type ListenerEvents = Arc<Mutex<Vec<ListenerEvent>>>;

struct RecordingListener {
    events: ListenerEvents,
}

impl RecordingListener {
    fn new_pair() -> (Arc<Self>, ListenerEvents) {
        let events: ListenerEvents = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(Self {
                events: events.clone(),
            }),
            events,
        )
    }
}

impl LifecycleListener<String, String> for RecordingListener {
    fn on_event(&self, reason: LifecycleReason, key: String, value: Option<String>) {
        self.events.lock().unwrap().push((reason, key, value));
    }
}

async fn with_timeout<F: Future>(limit: Duration, fut: F) -> F::Output {
    tokio::time::timeout(limit, fut)
        .await
        .expect("test exceeded timeout; raise limit in with_timeout(...) caller")
}

/// Counts `execute` calls; accepts any `Fut: Future + Send + 'static` with `Send` output (Tokio-style).
#[derive(Clone, Default)]
struct CountingExecutor {
    executed: Arc<AtomicUsize>,
}

impl CountingExecutor {
    fn executed_count(&self) -> usize {
        self.executed.load(Ordering::SeqCst)
    }
}

impl<Fut> Executor<Fut> for CountingExecutor
where
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    fn execute(&self, _fut: Fut) {
        self.executed.fetch_add(1, Ordering::SeqCst);
    }
}

/// Counts `execute` calls and holds onto each submitted future (boxed as
/// `Any`) so the cache observes a steady "in flight" job. Use this when
/// asserting `SkippedInFlight` against a follow-up submission — dropping
/// the future would clear `refresh_inflight` and let the second call
/// return `Submitted` instead.
#[derive(Clone, Default)]
struct HoldingExecutor {
    executed: Arc<AtomicUsize>,
    held: Arc<Mutex<Vec<Box<dyn std::any::Any + Send>>>>,
}

impl HoldingExecutor {
    fn executed_count(&self) -> usize {
        self.executed.load(Ordering::SeqCst)
    }
}

impl<Fut> Executor<Fut> for HoldingExecutor
where
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    fn execute(&self, fut: Fut) {
        self.executed.fetch_add(1, Ordering::SeqCst);
        self.held.lock().unwrap().push(Box::new(fut));
    }
}

/// For `#[tokio::test]` only: schedules work on the test's **current-thread** runtime via [`tokio::spawn`].
/// Drive completion with awaits / [`tokio::task::yield_now`] so spawned tasks run on that scheduler.
#[derive(Clone, Copy, Default)]
struct TokioSpawnExecutor;

impl<Fut> Executor<Fut> for TokioSpawnExecutor
where
    Fut: Future<Output = ()> + Send + 'static,
{
    fn execute(&self, fut: Fut) {
        tokio::spawn(fut);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InitError(i32);

mod sync {
    use super::*;

    #[test]
    fn get_miss_returns_none() {
        let cache = test_cache();
        assert_eq!(cache.get(&"missing".to_string()), None);
    }

    #[test]
    fn insert_get_remove_round_trip() {
        let cache = test_cache();
        let k = "key".to_string();
        let v = "value".to_string();
        assert_eq!(cache.get(&k), None);
        assert_eq!(cache.insert(k.clone(), v.clone()), None);
        assert_eq!(cache.get(&k), Some(v.clone()));
        assert_eq!(cache.remove(&k), Some(v));
        assert_eq!(cache.get(&k), None);
    }

    #[test]
    fn remove_absent_key_returns_none() {
        let cache = test_cache();
        assert_eq!(cache.remove(&"nope".to_string()), None);
    }

    #[test]
    fn insert_replace_returns_previous() {
        let cache = test_cache();
        let k = "k".to_string();
        assert_eq!(cache.insert(k.clone(), "v1".to_string()), None);
        assert_eq!(
            cache.insert(k.clone(), "v2".to_string()),
            Some("v1".to_string())
        );
        assert_eq!(cache.get(&k), Some("v2".to_string()));
    }

    #[test]
    fn eq_hash_key_aliasing_same_logical_key() {
        let cache = test_cache();
        let k_a: String = "shared".to_string();
        let k_b: String = "shared".to_string();
        assert_eq!(k_a, k_b);
        assert!(cache.insert(k_a, "data".to_string()).is_none());
        assert_eq!(cache.get(&k_b), Some("data".to_string()));
    }

    #[test]
    fn get_returns_some_after_insert_when_default_policy_serves() {
        let cache = test_cache();
        let k = "live".to_string();
        let _ = cache.insert(k.clone(), "stored".to_string());
        assert_eq!(cache.get(&k), Some("stored".to_string()));
    }
}

mod miss {
    use super::*;

    #[tokio::test]
    async fn try_get_with_miss_runs_initializer_once() {
        let cache = test_cache();
        let calls = Arc::new(AtomicUsize::new(0));
        let k = "miss1".to_string();
        let calls_f = Arc::clone(&calls);
        let out = cache
            .try_get_with(k.clone(), || async move {
                calls_f.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>("built".to_string())
            })
            .await;
        assert_eq!(out, Ok("built".to_string()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let calls_f = Arc::clone(&calls);
        let out2 = cache
            .try_get_with(k, || async move {
                calls_f.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>("should_not_run".to_string())
            })
            .await;
        assert_eq!(out2, Ok("built".to_string()));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn try_get_with_err_returns_clone_and_no_ok_install() {
        let cache = test_cache();
        let k = "err_key".to_string();
        let e = InitError(7);
        let got = cache
            .try_get_with(
                k.clone(),
                || async move { Err::<String, InitError>(e.clone()) },
            )
            .await;
        assert_eq!(got, Err(InitError(7)));
        assert_eq!(
            cache.get(&k),
            None,
            "prefer no entry after failed init (document if impl differs)"
        );
    }

    #[tokio::test]
    async fn try_get_with_err_three_waiters_same_key_single_init() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let k = "err_barrier".to_string();
            let barrier = Arc::new(tokio::sync::Barrier::new(3));
            let init_count = Arc::new(AtomicUsize::new(0));

            // The leader must hit an `.await` inside `init` so other tasks can run `miss_probe` and
            // observe `MapSlot::Miss` before the leader finishes with `Err` and removes the slot.
            // A purely synchronous `Err` future completes in one poll, so one task can run the whole
            // flight before peers are scheduled (then each sees an empty map and becomes leader again).
            let mut handles = Vec::new();
            for _ in 0..3 {
                let c = Arc::clone(&cache);
                let k = k.clone();
                let b = Arc::clone(&barrier);
                let ic = Arc::clone(&init_count);
                handles.push(tokio::spawn(async move {
                    b.wait().await;
                    c.try_get_with(k, move || async move {
                        ic.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Err::<String, ()>(())
                    })
                    .await
                }));
            }

            let mut results = Vec::new();
            for h in handles {
                results.push(h.await.expect("join"));
            }

            assert_eq!(init_count.load(Ordering::SeqCst), 1);
            for r in results {
                assert_eq!(r, Err(()));
            }
        })
        .await;
    }

    #[tokio::test]
    async fn try_get_with_closure_closes_over_key() {
        let cache = test_cache();
        let k = "closure_key".to_string();
        let k_ref = k.clone();
        let out = cache
            .try_get_with(k.clone(), move || async move {
                Ok::<_, ()>(format!("seen-{k_ref}"))
            })
            .await;
        assert_eq!(out, Ok("seen-closure_key".to_string()));
    }

    #[tokio::test]
    async fn get_with_hit_does_not_run_initializer() {
        let cache = test_cache();
        let k = "hit1".to_string();
        let _ = cache.insert(k.clone(), "cached".to_string());

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_c = Arc::clone(&calls);
        let v = cache
            .get_with(k.clone(), || async move {
                calls_c.fetch_add(1, Ordering::SeqCst);
                "wrong".to_string()
            })
            .await;
        assert_eq!(v, "cached");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn get_with_miss_matches_try_get_with_ok() {
        let cache = test_cache();
        let k = "gw1".to_string();
        let v = cache
            .get_with(k.clone(), || async { "fresh".to_string() })
            .await;
        assert_eq!(v, "fresh");
        assert_eq!(cache.get(&k), Some("fresh".to_string()));
    }

    #[test]
    fn get_with_blocking_hit_does_not_run_initializer() {
        let cache = test_cache();
        let k = "block_hit".to_string();
        let _ = cache.insert(k.clone(), "cached".to_string());
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_c = Arc::clone(&calls);
        let v = cache.get_with_blocking(k.clone(), |_| {
            calls_c.fetch_add(1, Ordering::SeqCst);
            "wrong".to_string()
        });
        assert_eq!(v, "cached");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn try_get_with_err_then_ok_same_key_installs_ok() {
        let cache = test_cache();
        let k = "recover".to_string();
        let first = cache
            .try_get_with(k.clone(), || async move { Err::<String, ()>(()) })
            .await;
        assert_eq!(first, Err(()));
        assert_eq!(cache.get(&k), None);

        let second = cache
            .try_get_with(k.clone(), || async move { Ok::<_, ()>("ok".to_string()) })
            .await;
        assert_eq!(second, Ok("ok".to_string()));
        assert_eq!(cache.get(&k), Some("ok".to_string()));
    }

    #[test]
    fn get_with_blocking_miss_on_sync_thread() {
        let cache = test_cache();
        let k = "block1".to_string();
        let v = cache.get_with_blocking(k.clone(), |_| "blocked".to_string());
        assert_eq!(v, "blocked");
        assert_eq!(cache.get(&k), Some("blocked".to_string()));
    }

    /// Sync sibling of `try_get_with_err_then_ok_same_key_installs_ok`:
    /// `try_get_with_blocking` returns the leader's `Err` to followers,
    /// does **not** cache it, and the next call re-runs `init` and
    /// installs the resulting value.
    #[test]
    fn try_get_with_blocking_err_then_ok_same_key_installs_ok() {
        let cache = test_cache();
        let k = "sync_recover".to_string();
        let first: Result<String, ()> = cache.try_get_with_blocking(k.clone(), |_| Err(()));
        assert_eq!(first, Err(()));
        assert_eq!(cache.get(&k), None);

        let second: Result<String, ()> =
            cache.try_get_with_blocking(k.clone(), |_| Ok("ok".to_string()));
        assert_eq!(second, Ok("ok".to_string()));
        assert_eq!(cache.get(&k), Some("ok".to_string()));
    }

    /// Concurrent sync followers on a fallible init that returns `Err`
    /// must all observe the leader's outcome rather than each running
    /// `init` themselves. Distinguishes a working `SyncMissFlight::wait`
    /// from a stub that returns `None`: under the stub, followers would
    /// loop back through the probe, find no installed entry (Err is not
    /// cached), become leaders, and re-run `init`, blowing past the
    /// single-flight guarantee on the error path.
    #[test]
    fn concurrent_try_get_with_blocking_err_runs_init_once() {
        use std::sync::Barrier;
        use std::thread;

        #[derive(Clone, PartialEq, Eq, Debug)]
        struct E;

        let cache = Arc::new(test_cache());
        let k = "sync_err_shared".to_string();
        let calls = Arc::new(AtomicUsize::new(0));

        let workers = 8usize;
        let barrier = Arc::new(Barrier::new(workers));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let cache = Arc::clone(&cache);
            let k = k.clone();
            let calls = Arc::clone(&calls);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                cache.try_get_with_blocking::<_, E>(k.clone(), |_| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Err(E)
                })
            }));
        }
        for h in handles {
            assert_eq!(h.join().unwrap(), Err(E));
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "sync miss coalescing must run a failing init exactly once across waiters",
        );
        assert_eq!(cache.get(&k), None, "Err outcomes must not be cached");
    }

    /// Sync miss-coalescing: many sync waiters on the same key share one
    /// init invocation. Mirrors the async `concurrent_get_with_same_key`
    /// guarantee but exercises the Condvar-backed path.
    #[test]
    fn concurrent_get_with_blocking_same_key_single_initializer() {
        use std::sync::Barrier;
        use std::thread;

        let cache = Arc::new(test_cache());
        let k = "shared".to_string();
        let calls = Arc::new(AtomicUsize::new(0));

        let workers = 8usize;
        let barrier = Arc::new(Barrier::new(workers));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let cache = Arc::clone(&cache);
            let k = k.clone();
            let calls = Arc::clone(&calls);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                cache.get_with_blocking(k.clone(), |_| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    "computed".to_string()
                })
            }));
        }
        for h in handles {
            assert_eq!(h.join().unwrap(), "computed");
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "sync miss coalescing must run init exactly once across waiters",
        );
    }
}

mod revalidate_tests {
    //! Executor scheduling and `SkippedInFlight` behaviour. `CountingExecutor` records schedules
    //! without running work. For **`RefreshWork`** completing on the test runtime, use
    //! `TokioSpawnExecutor` from a **`#[tokio::test]`** and `wait_get` until `get` matches.

    use super::*;

    async fn wait_get(cache: &Cache<K, V>, key: &str, want: &str) {
        let key = key.to_string();
        let want = want.to_string();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if cache.get(&key) == Some(want.clone()) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("timeout waiting for cache get after refresh");
    }

    #[test]
    fn try_revalidate_first_call_submitted_and_executes() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let k = "rev1".to_string();
        cache.insert(k.clone(), "old".into());
        let res = cache.try_revalidate(k.clone(), &ex, |_key| async move {
            Ok::<_, InitError>("nv".to_string())
        });
        let scheduled = res.expect("schedule ok");
        assert_eq!(scheduled, RevalidateScheduled::Submitted);
        assert_eq!(ex.executed_count(), 1);
    }

    #[test]
    fn try_revalidate_second_while_in_flight_skipped() {
        let cache = test_cache();
        let ex = HoldingExecutor::default();
        let k = "rev2".to_string();
        cache.insert(k.clone(), "old".into());
        let _ = cache
            .try_revalidate(k.clone(), &ex, |_key| async move {
                Ok::<_, InitError>("a".to_string())
            })
            .expect("first");
        let second = cache
            .try_revalidate(
                k,
                &ex,
                |_key| async move { Ok::<_, InitError>("b".to_string()) },
            )
            .expect("second ok");
        assert_eq!(second, RevalidateScheduled::SkippedInFlight);
        assert_eq!(ex.executed_count(), 1);
    }

    #[test]
    fn try_revalidate_initializer_error_propagates_as_err() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let k = "init_err".to_string();
        cache.insert(k.clone(), "old".into());
        let res = cache.try_revalidate(k, &ex, |_key| async move {
            Err::<String, InitError>(InitError(42))
        });
        assert_eq!(res, Err(InitError(42)));
    }

    #[test]
    fn try_revalidate_absent_key_returns_absent_and_does_not_schedule() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let res = cache.try_revalidate("missing".to_string(), &ex, |_key| async move {
            Ok::<_, InitError>("never_run".to_string())
        });
        assert_eq!(res, Ok(RevalidateScheduled::AbsentKey));
        assert_eq!(
            ex.executed_count(),
            0,
            "AbsentKey must not hand a job to the executor"
        );
    }

    #[test]
    fn revalidate_absent_key_returns_absent_and_does_not_schedule() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let s = cache.revalidate("missing_inf".to_string(), &ex, |_key| async move {
            "never_run".to_string()
        });
        assert_eq!(s, RevalidateScheduled::AbsentKey);
        assert_eq!(ex.executed_count(), 0);
    }

    #[test]
    fn revalidate_submitted_when_idle() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let k = "rev_inf".to_string();
        cache.insert(k.clone(), "old".into());
        let s = cache.revalidate(
            k.clone(),
            &ex,
            |_key| async move { "from_reval".to_string() },
        );
        assert_eq!(s, RevalidateScheduled::Submitted);
        assert_eq!(ex.executed_count(), 1);
    }

    #[test]
    fn revalidate_skipped_when_try_in_flight() {
        let cache = test_cache();
        let ex = HoldingExecutor::default();
        let k = "rev_inf2".to_string();
        cache.insert(k.clone(), "old".into());
        let _ = cache
            .try_revalidate(k.clone(), &ex, |_key| async move {
                Ok::<_, InitError>("x".to_string())
            })
            .unwrap();
        let s = cache.revalidate(k, &ex, |_key| async move { "y".to_string() });
        assert_eq!(s, RevalidateScheduled::SkippedInFlight);
        assert_eq!(ex.executed_count(), 1);
    }

    #[tokio::test]
    async fn refresh_replacing_existing_value_emits_updated_with_prior() {
        with_timeout(Duration::from_secs(30), async {
            let (listener, log) = RecordingListener::new_pair();
            let listener: Arc<dyn LifecycleListener<K, V> + Send + Sync> = listener;
            let cache = Cache::new(
                CacheBuilder::new()
                    .max_weight(Weight::new(10))
                    .lifecycle_listener(listener)
                    .build(),
            );
            cache.insert("k".into(), "old".into());

            let ex = TokioSpawnExecutor;
            assert_eq!(
                cache.try_revalidate("k".into(), &ex, |_k| async move {
                    Ok::<_, InitError>("new".into())
                }),
                Ok(RevalidateScheduled::Submitted)
            );
            wait_get(&cache, "k", "new").await;

            // Mirror the lifecycle emitted by `Cache::insert` when it
            // replaces an existing key, so callers can audit refresh-driven
            // replacements via the same Updated path.
            let saw_updated = log.lock().unwrap().iter().any(|(reason, key, val)| {
                *reason == LifecycleReason::Updated
                    && key == "k"
                    && val.as_ref() == Some(&"old".to_string())
            });
            assert!(
                saw_updated,
                "expected Updated(k, Some(old)); got {:?}",
                log.lock().unwrap()
            );
        })
        .await;
    }

    #[tokio::test]
    async fn try_revalidate_run_executor_updates_get_and_second_submit_not_skipped() {
        with_timeout(Duration::from_secs(30), async {
            let cache = test_cache();
            cache.insert("rwk".into(), "old".into());
            let ex = TokioSpawnExecutor;
            assert_eq!(
                cache.try_revalidate("rwk".into(), &ex, |_k| async move {
                    Ok::<_, InitError>("mid".into())
                }),
                Ok(RevalidateScheduled::Submitted)
            );
            wait_get(&cache, "rwk", "mid").await;
            assert_eq!(cache.get("rwk"), Some("mid".into()));
            assert_eq!(
                cache.try_revalidate("rwk".into(), &ex, |_k| async move {
                    Ok::<_, InitError>("new".into())
                }),
                Ok(RevalidateScheduled::Submitted)
            );
            wait_get(&cache, "rwk", "new").await;
            assert_eq!(cache.get("rwk"), Some("new".into()));
        })
        .await;
    }

    /// A `try_revalidate` whose initializer fails must release the
    /// per-slot in-flight flag so the caller can retry. The
    /// successful sibling case is covered by the test above (which
    /// resubmits after the executor completes); this one pins the
    /// error path: a synchronously-failing first attempt must not
    /// leave the slot wedged in `SkippedInFlight`.
    #[test]
    fn try_revalidate_after_initializer_error_can_be_resubmitted() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let k = "rev_err_clear".to_string();
        cache.insert(k.clone(), "old".into());
        let r1 = cache.try_revalidate(k.clone(), &ex, |_| async move {
            Err::<String, InitError>(InitError(1))
        });
        assert_eq!(r1, Err(InitError(1)));
        let r2 = cache.try_revalidate(k.clone(), &ex, |_| async move {
            Ok::<_, InitError>("new".to_string())
        });
        assert_eq!(
            r2,
            Ok(RevalidateScheduled::Submitted),
            "in-flight flag did not clear after the failed first revalidate"
        );
    }

    /// A panic in the initializer closure (before any future is even
    /// constructed) must not wedge the per-slot `refresh_inflight` flag.
    /// Without RAII cleanup the next `try_revalidate` returns
    /// `SkippedInFlight` forever.
    #[test]
    fn try_revalidate_after_initializer_panic_can_be_resubmitted() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let k = "rev_panic".to_string();
        cache.insert(k.clone(), "old".into());

        let cache_ref = &cache;
        let ex_ref = &ex;
        let k_panic = k.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            cache_ref.try_revalidate(
                k_panic,
                ex_ref,
                |_| -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<String, InitError>> + Send>,
                > { panic!("init blew up") },
            )
        }));
        assert!(result.is_err(), "panic should propagate to caller");

        let r2 = cache.try_revalidate(k.clone(), &ex, |_| async move {
            Ok::<_, InitError>("new".to_string())
        });
        assert_eq!(
            r2,
            Ok(RevalidateScheduled::Submitted),
            "panic in init must clear refresh_inflight"
        );
    }

    /// Captures `RefreshWork` and drops it without polling. Mimics a runtime
    /// that shuts down before the spawned job runs.
    #[derive(Clone, Default)]
    struct DroppingExecutor {
        seen: Arc<AtomicUsize>,
    }

    impl<Fut> Executor<Fut> for DroppingExecutor
    where
        Fut: Future + Send + 'static,
    {
        fn execute(&self, fut: Fut) {
            self.seen.fetch_add(1, Ordering::SeqCst);
            drop(fut);
        }
    }

    /// `RefreshWork::poll` is the only call site for `refresh_inflight_remove`,
    /// so a future that is dropped without ever being polled would otherwise
    /// strand the flag. `Drop` on `RefreshWork` clears it so the slot is
    /// schedulable again.
    #[test]
    fn try_revalidate_when_refresh_work_dropped_unpolled_can_be_resubmitted() {
        let cache = test_cache();
        let ex = DroppingExecutor::default();
        let k = "rev_dropped".to_string();
        cache.insert(k.clone(), "old".into());

        let r1 = cache.try_revalidate(k.clone(), &ex, |_| async move {
            tokio::task::yield_now().await;
            Ok::<_, InitError>("never_seen".to_string())
        });
        assert_eq!(r1, Ok(RevalidateScheduled::Submitted));
        assert_eq!(ex.seen.load(Ordering::SeqCst), 1);

        let counting = CountingExecutor::default();
        let r2 = cache.try_revalidate(k, &counting, |_| async move {
            Ok::<_, InitError>("after_drop".to_string())
        });
        assert_eq!(
            r2,
            Ok(RevalidateScheduled::Submitted),
            "RefreshWork drop must clear refresh_inflight"
        );
    }
}

// Revalidate data plane: after `Submitted` and the executor runs `RefreshWork` to completion,
// `get` should reflect the new value; on refresh `Err`, document or test whether `get` keeps the prior
// value. `try_revalidate` may return `Err(E)` synchronously (first-poll ready) or only via the job.

mod capacity {
    use super::*;

    #[test]
    fn max_weight_one_forbids_two_unit_weight_entries_both_visible() {
        let cache = test_cache_max_weight_one();
        assert_eq!(cache.insert("a".into(), "va".into()), None);
        assert_eq!(cache.insert("b".into(), "vb".into()), None);
        let a_live = cache.get("a").is_some();
        let b_live = cache.get("b").is_some();
        assert!(
            !(a_live && b_live),
            "with max_weight=1 and DefaultPolicy (weight 1 per entry), both keys must not remain \
             visible simultaneously once the core enforces the budget (eviction may be lazy on \
             insert—see DESIGN.md—relax only if the implementation deliberately defers eviction)"
        );
    }

    #[test]
    fn max_weight_two_keeps_both_unit_weight_entries() {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(4))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(policy)
                .build(),
        );
        assert_eq!(cache.insert("a".into(), "va".into()), None);
        assert_eq!(cache.insert("b".into(), "vb".into()), None);
        assert_eq!(cache.get("a"), Some("va".into()));
        assert_eq!(cache.get("b"), Some("vb".into()));
    }
}

mod lifecycle {
    use super::*;

    #[test]
    fn remove_after_insert_records_removed_with_prior_value() {
        let (listener, log) = RecordingListener::new_pair();
        let listener: Arc<dyn LifecycleListener<String, String> + Send + Sync> = listener;
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(10))
                .lifecycle_listener(listener)
                .build(),
        );
        let k = "tracked".to_string();
        let _ = cache.insert(k.clone(), "v".into());
        let _ = cache.remove(&k);

        let evts = log.lock().unwrap();
        assert!(
            evts.iter().any(|(reason, key, val)| {
                *reason == LifecycleReason::Removed
                    && key == "tracked"
                    && val.as_ref() == Some(&"v".to_string())
            }),
            "expected Removed(tracked, Some(v)); got {evts:?}"
        );
    }

    /// An entry that has expired but is still resident in the slab
    /// fires `TimeExpired` (with the prior value) the first time
    /// any subsequent operation triggers an opportunistic drain.
    /// Here the drain runs as part of `reclaim` during a second
    /// insert; the listener observes the expiry as a side-effect
    /// of unrelated traffic.
    #[test]
    fn expired_entry_drained_by_subsequent_insert_emits_time_expired() {
        let (listener, log) = RecordingListener::new_pair();
        let listener: Arc<dyn LifecycleListener<K, V> + Send + Sync> = listener;
        let policy: DynPolicy<K, V> = Arc::new(TtlFromWrite::new(Duration::ZERO, DefaultPolicy));
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(16))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(policy)
                .lifecycle_listener(listener)
                .build(),
        );
        cache.insert("k1".into(), "v1".into());
        cache.insert("k2".into(), "v2".into());
        let events = log.lock().unwrap().clone();
        assert!(
            events.iter().any(|(r, k, v)| {
                *r == LifecycleReason::TimeExpired
                    && k == "k1"
                    && v.as_ref() == Some(&"v1".to_string())
            }),
            "expected TimeExpired(k1, Some(v1)); got {events:?}"
        );
    }

    /// Replacing a key — even one whose deadline has already passed
    /// — emits `Updated` with the prior value, never `TimeExpired`.
    /// The opportunistic drain that runs alongside the install must
    /// recognise the insert key and skip it so the lifecycle taxonomy
    /// stays unambiguous: `Updated` always means "explicit replace by
    /// the user", `TimeExpired` always means "deadline drove this".
    #[test]
    fn replacing_expired_key_emits_updated_not_time_expired() {
        let (listener, log) = RecordingListener::new_pair();
        let listener: Arc<dyn LifecycleListener<K, V> + Send + Sync> = listener;
        let policy: DynPolicy<K, V> = Arc::new(TtlFromWrite::new(Duration::ZERO, DefaultPolicy));
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(16))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(policy)
                .lifecycle_listener(listener)
                .build(),
        );
        cache.insert("k".into(), "v1".into());
        cache.insert("k".into(), "v2".into());
        let events = log.lock().unwrap().clone();
        assert!(
            events.iter().any(|(r, k, v)| {
                *r == LifecycleReason::Updated && k == "k" && v.as_ref() == Some(&"v1".to_string())
            }),
            "expected Updated(k, Some(v1)); got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|(r, k, _)| *r == LifecycleReason::TimeExpired && k == "k"),
            "TimeExpired must not fire for the key being replaced; got {events:?}"
        );
    }
}

mod concurrency {
    use tokio::sync::Barrier;

    use super::*;

    #[tokio::test]
    async fn concurrent_miss_same_key_single_initializer() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let k = "race1".to_string();
            let calls = Arc::new(AtomicUsize::new(0));
            let barrier = Arc::new(Barrier::new(2));

            let c1 = Arc::clone(&cache);
            let c2 = Arc::clone(&cache);
            let k1 = k.clone();
            let k2 = k.clone();
            let calls1 = Arc::clone(&calls);
            let calls2 = Arc::clone(&calls);
            let b1 = Arc::clone(&barrier);
            let b2 = Arc::clone(&barrier);

            let h1 = tokio::spawn(async move {
                b1.wait().await;
                c1.try_get_with(k1, || async move {
                    calls1.fetch_add(1, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                    Ok::<_, ()>("shared".to_string())
                })
                .await
            });
            let h2 = tokio::spawn(async move {
                b2.wait().await;
                c2.try_get_with(k2, || async move {
                    calls2.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, ()>("shared".to_string())
                })
                .await
            });

            let (j1, j2) = tokio::join!(h1, h2);
            let r1 = j1.expect("join").expect("try_get_with");
            let r2 = j2.expect("join").expect("try_get_with");
            assert_eq!(r1, "shared");
            assert_eq!(r2, "shared");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "at most one miss initializer"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn concurrent_miss_distinct_keys_both_run() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let calls = Arc::new(AtomicUsize::new(0));

            let h1 = {
                let cache = Arc::clone(&cache);
                let calls = Arc::clone(&calls);
                tokio::spawn(async move {
                    cache
                        .try_get_with("a".to_string(), || async move {
                            calls.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, ()>("va".to_string())
                        })
                        .await
                })
            };
            let h2 = {
                let cache = Arc::clone(&cache);
                let calls = Arc::clone(&calls);
                tokio::spawn(async move {
                    cache
                        .try_get_with("b".to_string(), || async move {
                            calls.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, ()>("vb".to_string())
                        })
                        .await
                })
            };
            let (j1, j2) = tokio::join!(h1, h2);
            let r1 = j1.expect("join").expect("try_get_with");
            let r2 = j2.expect("join").expect("try_get_with");
            assert_eq!(r1, "va".to_string());
            assert_eq!(r2, "vb".to_string());
            assert_eq!(calls.load(Ordering::SeqCst), 2);
        })
        .await;
    }

    #[tokio::test]
    async fn concurrent_get_with_same_key_single_initializer() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let k = "race_gw".to_string();
            let calls = Arc::new(AtomicUsize::new(0));
            let barrier = Arc::new(Barrier::new(2));

            let c1 = Arc::clone(&cache);
            let c2 = Arc::clone(&cache);
            let k1 = k.clone();
            let k2 = k.clone();
            let calls1 = Arc::clone(&calls);
            let calls2 = Arc::clone(&calls);
            let b1 = Arc::clone(&barrier);
            let b2 = Arc::clone(&barrier);

            let h1 = tokio::spawn(async move {
                b1.wait().await;
                c1.get_with(k1, || async move {
                    calls1.fetch_add(1, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                    "shared".to_string()
                })
                .await
            });
            let h2 = tokio::spawn(async move {
                b2.wait().await;
                c2.get_with(k2, || async move {
                    calls2.fetch_add(1, Ordering::SeqCst);
                    "shared".to_string()
                })
                .await
            });

            let (j1, j2) = tokio::join!(h1, h2);
            let r1 = j1.expect("join");
            let r2 = j2.expect("join");
            assert_eq!(r1, "shared");
            assert_eq!(r2, "shared");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "at most one miss initializer for get_with"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn try_get_with_follower_awaits_miss_leader_without_second_initializer() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let k = "miss_follow".to_string();
            let (tx, rx) = tokio::sync::oneshot::channel();
            let k1 = k.clone();
            let c1 = Arc::clone(&cache);
            let leader = tokio::spawn(async move {
                c1.try_get_with(k1, || async move {
                    rx.await.expect("release leader");
                    Ok::<_, ()>("v".into())
                })
                .await
            });
            for _ in 0..64 {
                tokio::task::yield_now().await;
            }
            let calls = Arc::new(AtomicUsize::new(0));
            let calls_f = Arc::clone(&calls);
            let c2 = Arc::clone(&cache);
            let follower = tokio::spawn(async move {
                c2.try_get_with(k, || async move {
                    calls_f.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, ()>("wrong".into())
                })
                .await
            });
            tokio::time::sleep(Duration::from_millis(30)).await;
            tx.send(()).expect("unblock leader");
            let (r1, r2) = tokio::join!(leader, follower);
            assert_eq!(r1.expect("leader join"), Ok("v".into()));
            assert_eq!(r2.expect("follower join"), Ok("v".into()));
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "follower must await the in-flight miss, not run a second initializer"
            );
        })
        .await;
    }
}

mod borrow_lookup {
    //! `Cache::get`, `remove`, and `contains_key` accept any `&Q` for which
    //! `K: Borrow<Q>`, so a `Cache<String, _>` is queryable with `&str`
    //! without owning a `String`.

    use super::*;

    #[test]
    fn get_remove_contains_accept_str_for_string_keyed_cache() {
        let cache = test_cache();
        cache.insert("hello".into(), "world".into());

        assert!(cache.contains_key("hello"));
        assert_eq!(cache.get("hello"), Some("world".to_string()));
        assert_eq!(cache.remove("hello"), Some("world".to_string()));
        assert!(!cache.contains_key("hello"));
        assert_eq!(cache.get("hello"), None);
    }

    #[test]
    fn contains_key_does_not_bump_freq() {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(4))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(policy)
                .build(),
        );
        cache.insert("a".into(), "va".into());
        cache.insert("b".into(), "vb".into());
        // `a` is the Small-tail. If `contains_key` had bumped its
        // S3-FIFO freq counter the next over-budget insert might still
        // evict it (a single bump is not enough to hit the freq>1
        // promotion threshold), but at the very least the cold-tail
        // eviction below would not be guaranteed under heavier reads.
        // The point of this test is the asymmetry vs `get`: containment
        // probes leave the freq counter untouched.
        assert!(cache.contains_key("a"));
        cache.insert("c".into(), "vc".into());
        assert_eq!(
            cache.get("a"),
            None,
            "contains_key must not have bumped a's freq counter"
        );
    }

    /// `get` bumps the touched slot's S3-FIFO freq counter without
    /// reordering it. Pair with `contains_key_does_not_bump_freq`:
    /// together they pin down the asymmetry — only value-returning
    /// reads count as a "touch" for the freq-driven promotion path.
    /// Two reads are needed to clear the `freq > 1` promotion bar so a
    /// Small-tail slot survives the next over-budget reclaim by being
    /// migrated into Main rather than evicted.
    #[test]
    fn repeated_get_promotes_warm_entry_into_main() {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(8))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(policy)
                .build(),
        );
        cache.insert("a".into(), "va".into());
        cache.insert("b".into(), "vb".into());
        cache.insert("c".into(), "vc".into());
        cache.insert("d".into(), "vd".into());
        // Two reads each lifts `a` and `b` to freq=2, above the
        // `freq > 1` Small-tail promotion threshold. `c` and `d` stay
        // at freq=0 (cold).
        for _ in 0..2 {
            assert_eq!(cache.get("a"), Some("va".into()));
            assert_eq!(cache.get("b"), Some("vb".into()));
        }
        cache.insert("e".into(), "ve".into());
        cache.insert("f".into(), "vf".into());
        // `a` and `b` migrated to Main on Small-tail processing; the
        // cold tails `c` and `d` were evicted to make room.
        assert_eq!(cache.get("a"), Some("va".into()), "warm a survived in Main");
        assert_eq!(cache.get("b"), Some("vb".into()), "warm b survived in Main");
        assert_eq!(cache.get("c"), None, "cold c was evicted from Small");
        assert_eq!(cache.get("d"), None, "cold d was evicted from Small");
        assert_eq!(cache.get("e"), Some("ve".into()));
        assert_eq!(cache.get("f"), Some("vf".into()));
    }
}

mod cancellation {
    //! `try_get_with` must not deadlock subsequent callers when the leader's
    //! init future is cancelled (caller dropped, e.g. via `tokio::select!`
    //! / `timeout`) or panics partway through. The leader cleanup guard
    //! removes the `MapSlot::Miss` slot and signals followers to retry.

    use super::*;

    #[tokio::test]
    async fn cancelled_leader_lets_follower_progress() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let key = "cancel".to_string();
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel::<()>();

            let c1 = Arc::clone(&cache);
            let k1 = key.clone();
            let leader = tokio::spawn(async move {
                c1.try_get_with(k1, move || async move {
                    let _ = entered_tx.send(());
                    std::future::pending::<()>().await;
                    Ok::<_, ()>("never".to_string())
                })
                .await
            });

            entered_rx.await.expect("leader entered init");
            leader.abort();
            let _ = leader.await;

            let res = cache
                .try_get_with(key.clone(), || async move {
                    Ok::<_, ()>("recovered".to_string())
                })
                .await;
            assert_eq!(res, Ok("recovered".to_string()));
            assert_eq!(cache.get(&key), Some("recovered".to_string()));
        })
        .await;
    }

    #[tokio::test]
    async fn distinct_error_types_run_independently_on_same_key() {
        // Each `try_get_with::<_, E>` keys its lane on `(K, TypeId<E>)`, so a
        // leader publishing `Result<V, E1>` cannot satisfy a concurrent
        // follower expecting `Result<V, E2>`. Both initializers must run
        // exactly once and produce their own typed result.
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let key = "typed".to_string();
            let barrier = Arc::new(tokio::sync::Barrier::new(2));
            let init_calls = Arc::new(AtomicUsize::new(0));

            let c_unit = Arc::clone(&cache);
            let k_unit = key.clone();
            let b_unit = Arc::clone(&barrier);
            let n_unit = Arc::clone(&init_calls);
            let task_unit = tokio::spawn(async move {
                b_unit.wait().await;
                c_unit
                    .try_get_with(k_unit, move || async move {
                        n_unit.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Ok::<_, ()>("unit-ok".to_string())
                    })
                    .await
            });

            #[derive(Clone, Debug, PartialEq, Eq)]
            struct OtherErr;

            let c_other = Arc::clone(&cache);
            let k_other = key.clone();
            let b_other = Arc::clone(&barrier);
            let n_other = Arc::clone(&init_calls);
            let task_other = tokio::spawn(async move {
                b_other.wait().await;
                c_other
                    .try_get_with(k_other, move || async move {
                        n_other.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Ok::<_, OtherErr>("other-ok".to_string())
                    })
                    .await
            });

            let r_unit = task_unit.await.expect("unit task join");
            let r_other = task_other.await.expect("other task join");

            // Each lane ran its own initializer; lanes did not coalesce.
            assert_eq!(init_calls.load(Ordering::SeqCst), 2);
            // Both must succeed; the second leader replaces the first's
            // installed value (last-write-wins under our install ordering).
            assert!(matches!(&r_unit, Ok(v) if v == "unit-ok" || v == "other-ok"));
            assert!(matches!(&r_other, Ok(v) if v == "unit-ok" || v == "other-ok"));
            assert!(cache.get(&key).is_some(), "some value must remain");
        })
        .await;
    }

    #[tokio::test]
    async fn panicking_leader_lets_follower_progress() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let key = "panic".to_string();

            let c1 = Arc::clone(&cache);
            let k1 = key.clone();
            // tokio catches the panic and returns Err(JoinError); the leader
            // future is dropped during unwind, firing LeaderGuard::drop.
            let leader = tokio::spawn(async move {
                c1.try_get_with(k1, || async {
                    tokio::task::yield_now().await;
                    panic!("leader init panic");
                    #[allow(unreachable_code)]
                    Ok::<_, ()>("unreachable".to_string())
                })
                .await
            });
            let join = leader.await;
            assert!(join.is_err(), "leader task should have panicked");

            let res = cache
                .try_get_with(key.clone(), || async move {
                    Ok::<_, ()>("recovered".to_string())
                })
                .await;
            assert_eq!(res, Ok("recovered".to_string()));
            assert_eq!(cache.get(&key), Some("recovered".to_string()));
        })
        .await;
    }

    /// Sync sibling of `panicking_leader_lets_follower_progress`: a
    /// panic from the closure passed to `get_with_blocking` must
    /// release the leader's inflight lane so a follow-up call can
    /// install a value instead of joining a flight that will never
    /// settle.
    #[test]
    fn panicking_sync_init_lets_next_caller_progress() {
        let cache: Arc<Cache<K, V>> = Arc::new(test_cache());
        let key = "sync_panic".to_string();
        let panic_cache = Arc::clone(&cache);
        let panic_key = key.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            panic_cache.get_with_blocking(panic_key, |_| -> String { panic!("init blew up") });
        }));
        assert!(result.is_err(), "panic should propagate to the caller");
        let v = cache.get_with_blocking(key.clone(), |_| "recovered".to_string());
        assert_eq!(v, "recovered");
        assert_eq!(cache.get(&key), Some("recovered".to_string()));
    }

    /// A sync follower already blocked inside `get_with_blocking`
    /// must wake when the leader's initializer panics — the
    /// leader's drop guard cancels the in-flight lane and signals
    /// every Condvar waiter, who then re-enter the probe loop and
    /// install their own value.
    #[test]
    fn panicking_sync_leader_lets_concurrent_follower_progress() {
        use std::sync::{Barrier, mpsc};
        use std::thread;

        let cache: Arc<Cache<K, V>> = Arc::new(test_cache());
        let key = "sync_panic_follower".to_string();
        let leader_started = Arc::new(Barrier::new(2));
        let release_leader = Arc::new(Barrier::new(2));

        let c_leader = Arc::clone(&cache);
        let k_leader = key.clone();
        let started = Arc::clone(&leader_started);
        let release = Arc::clone(&release_leader);
        let leader = thread::spawn(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                c_leader.get_with_blocking(k_leader, |_| -> String {
                    started.wait();
                    release.wait();
                    panic!("sync init blew up");
                });
            }));
        });

        leader_started.wait();

        let (tx, rx) = mpsc::channel();
        let c_follow = Arc::clone(&cache);
        let k_follow = key.clone();
        let follower = thread::spawn(move || {
            let v = c_follow.get_with_blocking(k_follow, |_| "follower".to_string());
            let _ = tx.send(v);
        });

        thread::sleep(Duration::from_millis(80));
        release_leader.wait();
        leader.join().expect("leader thread join");

        let v = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("follower deadlocked: leader's panic did not signal sync waiters");
        assert_eq!(
            v, "follower",
            "follower must install its own value once the leader's flight is cancelled"
        );
        follower.join().expect("follower thread join");
    }
}

mod lanes {
    //! Concurrent `try_get_with` miss and `try_revalidate` on the same key: miss initializer runs
    //! once; revalidate returns `Submitted` and the executor records one `execute`.
    //!
    //! **Typed lanes:** `try_revalidate` uses `RefreshWork<OkIntoResult<Fut>>` while misses use their
    //! own future types (see DESIGN.md). Do not assume `(key, TypeId)` coalescing across miss vs
    //! refresh unless the implementation explicitly unifies those lanes—adjust
    //! `RevalidateScheduled` / `executed_count` expectations here if refresh waits on miss flight.

    use super::*;

    #[tokio::test]
    async fn concurrent_get_and_revalidate_on_present_key_both_progress() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let k = "lane1".to_string();
            cache.insert(k.clone(), "warm".into());
            let ex = CountingExecutor::default();

            let read_calls = Arc::new(AtomicUsize::new(0));
            let rc = Arc::clone(&read_calls);
            let c_get = Arc::clone(&cache);
            let k_get = k.clone();
            let getter = tokio::spawn(async move {
                let v = c_get.get(&k_get);
                rc.fetch_add(1, Ordering::SeqCst);
                v
            });

            let c_rev = Arc::clone(&cache);
            let k_rev = k.clone();
            let ex_rev = ex.clone();
            let rev = tokio::spawn(async move {
                tokio::task::yield_now().await;
                c_rev.try_revalidate(k_rev, &ex_rev, |_key| async move {
                    Ok::<_, InitError>("rv".to_string())
                })
            });

            let (g, r) = tokio::join!(getter, rev);
            let get_res = g.expect("get join");
            let rev_res = r.expect("rev join").expect("try_revalidate");
            assert_eq!(get_res.as_deref(), Some("warm"));
            assert_eq!(rev_res, RevalidateScheduled::Submitted);
            assert_eq!(ex.executed_count(), 1);
            assert_eq!(read_calls.load(Ordering::SeqCst), 1);
        })
        .await;
    }
}

mod policy_time {
    //! Wall-clock tests for [`TtlFromWrite`] and [`IdleTimeout`] against [`Cache::get`] / deadlines.

    use super::*;

    #[tokio::test]
    async fn idle_timeout_expires_after_idle_without_reads() {
        with_timeout(Duration::from_secs(30), async {
            // TTLs sit comfortably above the 250ms internal tick so the
            // ceil-rounded deadline is observably in the past after the sleep.
            let idle = Duration::from_millis(500);
            let policy: DynPolicy<K, V> = Arc::new(IdleTimeout::new(idle, DefaultPolicy));
            let cache = Cache::<K, V>::new(
                CacheBuilder::<K, V>::new()
                    .max_weight(Weight::new(100))
                    .policy(policy)
                    .build(),
            );
            cache.insert("k".into(), "v".into());
            tokio::time::sleep(Duration::from_millis(900)).await;
            assert_eq!(cache.get("k"), None);
        })
        .await;
    }

    #[tokio::test]
    async fn idle_timeout_read_extends_deadline() {
        with_timeout(Duration::from_secs(30), async {
            let idle = Duration::from_millis(500);
            let policy: DynPolicy<K, V> = Arc::new(IdleTimeout::new(idle, DefaultPolicy));
            let cache = Cache::<K, V>::new(
                CacheBuilder::<K, V>::new()
                    .max_weight(Weight::new(100))
                    .policy(policy)
                    .build(),
            );
            cache.insert("k".into(), "v".into());
            tokio::time::sleep(Duration::from_millis(250)).await;
            assert_eq!(cache.get("k"), Some("v".into()));
            tokio::time::sleep(Duration::from_millis(250)).await;
            assert_eq!(cache.get("k"), Some("v".into()));
            tokio::time::sleep(Duration::from_millis(900)).await;
            assert_eq!(cache.get("k"), None);
        })
        .await;
    }

    #[tokio::test]
    async fn ttl_from_write_expires_after_ttl_without_reads() {
        with_timeout(Duration::from_secs(30), async {
            let ttl = Duration::from_millis(500);
            let policy: DynPolicy<K, V> = Arc::new(TtlFromWrite::new(ttl, DefaultPolicy));
            let cache = Cache::<K, V>::new(
                CacheBuilder::<K, V>::new()
                    .max_weight(Weight::new(100))
                    .policy(policy)
                    .build(),
            );
            cache.insert("k".into(), "v".into());
            tokio::time::sleep(Duration::from_millis(900)).await;
            assert_eq!(cache.get("k"), None);
        })
        .await;
    }

    #[tokio::test]
    async fn ttl_expire_then_try_get_with_runs_initializer() {
        with_timeout(Duration::from_secs(30), async {
            let ttl = Duration::from_millis(500);
            let policy: DynPolicy<K, V> = Arc::new(TtlFromWrite::new(ttl, DefaultPolicy));
            let cache = Arc::new(Cache::<K, V>::new(
                CacheBuilder::<K, V>::new()
                    .max_weight(Weight::new(100))
                    .policy(policy)
                    .build(),
            ));
            let k = "tk".to_string();
            cache.insert(k.clone(), "old".into());
            tokio::time::sleep(Duration::from_millis(900)).await;
            let calls = Arc::new(AtomicUsize::new(0));
            let calls_c = Arc::clone(&calls);
            let out = cache
                .clone()
                .try_get_with(k.clone(), move || {
                    let c = Arc::clone(&calls_c);
                    async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, ()>("fresh".to_string())
                    }
                })
                .await
                .expect("try_get_with");
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(out, "fresh");
            assert_eq!(cache.get(&k), Some("fresh".into()));
        })
        .await;
    }

    /// Read-extending policies (`IdleTimeout`) shift an entry's
    /// deadline forward on every read, leaving the original heap
    /// tuple stale. A subsequent unrelated write triggers
    /// opportunistic drain of the heap; that drain must validate
    /// the popped tuple against the live entry's *current* deadline
    /// and skip it instead of evicting a still-fresh entry.
    #[tokio::test]
    async fn idle_timeout_does_not_evict_recently_read_entry_after_first_deadline_passes() {
        with_timeout(Duration::from_secs(30), async {
            let idle = Duration::from_millis(600);
            let policy: DynPolicy<K, V> = Arc::new(IdleTimeout::new(idle, DefaultPolicy));
            let cache = Cache::new(
                CacheBuilder::new()
                    .max_weight(Weight::new(16))
                    .shards(2)
                    .hasher(SingleShardHasher)
                    .policy(policy)
                    .build(),
            );
            cache.insert("k".into(), "v".into());
            tokio::time::sleep(Duration::from_millis(500)).await;
            assert_eq!(cache.get("k"), Some("v".to_string()));
            // The original (~+750ms ceil-tick) deadline is now in the past;
            // the read-extended deadline (~+1250ms) is still ahead.
            tokio::time::sleep(Duration::from_millis(400)).await;
            cache.insert("trigger".into(), "tv".into());
            assert_eq!(
                cache.get("k"),
                Some("v".to_string()),
                "stale heap tuple drove a spurious eviction of a still-live entry"
            );
        })
        .await;
    }

    /// Sustained read traffic against `IdleTimeout` accumulates stale
    /// heap tuples, which the cache periodically rebuilds from the
    /// slab to keep memory bounded. After the rebuild, all entries
    /// must remain individually reachable by deadline-driven
    /// eviction: an unrelated insert triggered after every entry has
    /// idled out fires a `TimeExpired` event for each.
    #[tokio::test]
    async fn idle_timeout_evicts_quiesced_entries_after_sustained_reads() {
        with_timeout(Duration::from_secs(30), async {
            let (listener, log) = RecordingListener::new_pair();
            let listener: Arc<dyn LifecycleListener<K, V> + Send + Sync> = listener;
            let idle = Duration::from_millis(500);
            let policy: DynPolicy<K, V> = Arc::new(IdleTimeout::new(idle, DefaultPolicy));
            let cache = Cache::new(
                CacheBuilder::new()
                    .max_weight(Weight::new(128))
                    .shards(2)
                    .hasher(SingleShardHasher)
                    .policy(policy)
                    .lifecycle_listener(listener)
                    .build(),
            );
            for i in 0..4 {
                cache.insert(format!("k{i}"), format!("v{i}"));
            }
            for _ in 0..32 {
                for i in 0..4 {
                    let _ = cache.get(&format!("k{i}"));
                }
                tokio::task::yield_now().await;
            }
            tokio::time::sleep(Duration::from_millis(900)).await;
            cache.insert("trigger".into(), "tv".into());

            let events = log.lock().unwrap().clone();
            for i in 0..4 {
                let key = format!("k{i}");
                assert!(
                    events
                        .iter()
                        .any(|(r, k, _)| *r == LifecycleReason::TimeExpired && *k == key),
                    "expected TimeExpired({key}); got {events:?}"
                );
            }
        })
        .await;
    }
}

mod mutation_guards {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Instant;

    use super::*;

    #[derive(Clone, Default)]
    struct LenWeightPolicy;

    impl Policy<K, V> for LenWeightPolicy {
        fn weight(&self, _k: &K, v: &V) -> Weight {
            Weight::new(v.len() as u64)
        }

        fn expire_after_read(
            &self,
            _k: &K,
            _v: &V,
            _read_at: Instant,
            _written_at: Instant,
            _time_remaining: Duration,
        ) -> Duration {
            TIME_UNBOUNDED
        }

        fn expire_after_write(&self, _k: &K, _v: &V, _written_at: Instant) -> Duration {
            TIME_UNBOUNDED
        }
    }

    #[test]
    fn replace_under_budget_pressure_evicts_non_insert_key_first() {
        let (listener, log) = RecordingListener::new_pair();
        let listener: Arc<dyn LifecycleListener<K, V> + Send + Sync> = listener;
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(24))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(Arc::new(LenWeightPolicy))
                .lifecycle_listener(listener)
                .build(),
        );
        let keep = "keep".to_string();
        let repl = "repl".to_string();
        cache.insert(keep.clone(), "k".repeat(5));
        cache.insert(repl.clone(), "r".repeat(5));

        cache.insert(repl.clone(), "R".repeat(9));

        let evicted_first = {
            let evts = log.lock().unwrap();
            evts.iter()
                .find(|(reason, _, _)| *reason == LifecycleReason::EvictedSize)
                .map(|(reason, key, val)| (*reason, key.clone(), val.clone()))
        };
        assert!(
            evicted_first.is_some(),
            "expected a size eviction while resizing repl under a tight budget"
        );
        assert_eq!(
            evicted_first.unwrap().1,
            keep,
            "must evict another Ready slot before evicting the insert key being resized"
        );
    }

    #[derive(Clone)]
    struct WriteUnboundedWitness(Arc<AtomicBool>);

    impl Policy<K, V> for WriteUnboundedWitness {
        fn weight(&self, _k: &K, _v: &V) -> Weight {
            Weight::new(1)
        }

        fn expire_after_read(
            &self,
            _k: &K,
            _v: &V,
            _read_at: Instant,
            _written_at: Instant,
            time_remaining: Duration,
        ) -> Duration {
            self.0
                .store(time_remaining == TIME_UNBOUNDED, Ordering::SeqCst);
            TIME_UNBOUNDED
        }

        fn expire_after_write(&self, _k: &K, _v: &V, _written_at: Instant) -> Duration {
            TIME_UNBOUNDED
        }
    }

    #[test]
    fn get_passes_sentinel_time_remaining_when_write_is_unbounded_and_deadline_none() {
        let witness = Arc::new(AtomicBool::new(false));
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(100))
                .policy(Arc::new(WriteUnboundedWitness(witness.clone())))
                .build(),
        );
        let k = "k".to_string();
        cache.insert(k.clone(), "v".into());
        assert!(cache.get(&k).is_some());
        assert!(
            witness.load(Ordering::SeqCst),
            "live hits must pass the TIME_UNBOUNDED sentinel as time_remaining when no finite deadline is stored"
        );
    }

    #[derive(Clone)]
    struct ReadSentinelWitness {
        reads: Arc<AtomicUsize>,
        second_hit_tr_is_sentinel: Arc<AtomicBool>,
    }

    impl Policy<K, V> for ReadSentinelWitness {
        fn weight(&self, _k: &K, _v: &V) -> Weight {
            Weight::new(1)
        }

        fn expire_after_read(
            &self,
            _k: &K,
            _v: &V,
            _read_at: Instant,
            _written_at: Instant,
            time_remaining: Duration,
        ) -> Duration {
            let n = self.reads.fetch_add(1, Ordering::SeqCst);
            if n > 0 {
                self.second_hit_tr_is_sentinel
                    .store(time_remaining == TIME_UNBOUNDED, Ordering::SeqCst);
            }
            TIME_UNBOUNDED
        }

        fn expire_after_write(&self, _k: &K, _v: &V, _written_at: Instant) -> Duration {
            TIME_UNBOUNDED
        }
    }

    /// A read hook that returns the literal [`TIME_UNBOUNDED`] sentinel
    /// must fold the deadline to `None` (skipping the expiry heap), so
    /// the next hit still observes the sentinel as `time_remaining`
    /// rather than a finite drift. Exercises the read-side fold
    /// branch in `update_read_deadline`, complementing
    /// [`get_passes_sentinel_time_remaining_when_write_is_unbounded_and_deadline_none`]
    /// which only covers the write-side fold.
    #[test]
    fn read_returning_sentinel_keeps_deadline_none_for_next_hit() {
        let reads = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicBool::new(false));
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(100))
                .policy(Arc::new(ReadSentinelWitness {
                    reads: reads.clone(),
                    second_hit_tr_is_sentinel: second.clone(),
                }))
                .build(),
        );
        let k = "k".to_string();
        cache.insert(k.clone(), "v".into());
        assert!(cache.get(&k).is_some());
        assert_eq!(reads.load(Ordering::SeqCst), 1);
        assert!(cache.get(&k).is_some());
        assert!(
            second.load(Ordering::SeqCst),
            "after folding a sentinel read allowance into None, the next hit must still observe TIME_UNBOUNDED remaining"
        );
    }

    #[test]
    fn remove_restores_budget_for_len_weighted_values() {
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(20))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(Arc::new(LenWeightPolicy))
                .build(),
        );
        let ten = "x".repeat(10);
        assert_eq!(cache.insert("a".into(), ten.clone()), None);
        assert_eq!(cache.remove("a"), Some(ten));
        assert_eq!(cache.insert("b".into(), "b".repeat(5)), None);
        assert_eq!(cache.insert("c".into(), "c".repeat(5)), None);
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    /// Replacing a sole entry whose weight already exceeds the budget
    /// must complete: there is nothing else to evict, and the install
    /// path is allowed to leave the cache temporarily over-budget
    /// (otherwise callers could never replace an oversized entry).
    /// Regression guard for the `slab.len() <= 1` short-circuit in
    /// `evict_from_small` / `evict_from_main` — without it the
    /// reclaim loop would keep promoting the entry being replaced back
    /// to its current queue's head and never terminate.
    #[test]
    fn oversized_entry_can_be_replaced_when_it_alone_exceeds_budget() {
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(8))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(Arc::new(LenWeightPolicy))
                .build(),
        );
        cache.insert("k".into(), "12345".into());
        cache.insert("k".into(), "67890".into());
        assert_eq!(cache.get("k"), Some("67890".into()));
    }
}

mod deferred {
    use super::*;

    #[test]
    fn get_returns_none_when_write_ttl_cap_is_zero_entry_not_served() {
        let cache = test_cache_ttl_zero_write();
        let k = "ttl0".to_string();
        let _ = cache.insert(k.clone(), "v".into());
        assert_eq!(
            cache.get(&k),
            None,
            "TtlFromWrite(ZERO): expire_after_write is zero — get must not serve the entry once the \
             core compares deadline to now (see Policy docs)"
        );
    }
}

mod clear {
    //! `Cache::clear` resets the cache *and* unblocks any callers
    //! currently waiting on a single-flight initializer. The contract
    //! is: a caller that started before `clear()` must never deadlock
    //! and must observe a fresh cache (its retry installs its own
    //! value rather than receiving the pre-clear leader's eventual
    //! result).

    use super::*;

    /// An in-flight `try_get_with` follower wakes when `clear()`
    /// runs, retries against the now-empty cache, and installs its
    /// own initializer's value. The leader is kept parked beyond
    /// the assertion so the only path that can satisfy the
    /// follower's await is the cancel signal from `clear()`.
    #[tokio::test]
    async fn clear_during_inflight_try_get_with_lets_follower_install_fresh_value() {
        with_timeout(Duration::from_secs(10), async {
            let cache = Arc::new(test_cache());
            let key = "shared".to_string();
            let leader_blocked = Arc::new(tokio::sync::Notify::new());
            let release_leader = Arc::new(tokio::sync::Notify::new());

            let c_leader = Arc::clone(&cache);
            let k_leader = key.clone();
            let blocked = Arc::clone(&leader_blocked);
            let release = Arc::clone(&release_leader);
            let leader = tokio::spawn(async move {
                c_leader
                    .try_get_with(k_leader, move || async move {
                        blocked.notify_one();
                        release.notified().await;
                        Ok::<_, InitError>("leader".to_string())
                    })
                    .await
            });
            leader_blocked.notified().await;

            let c_follow = Arc::clone(&cache);
            let k_follow = key.clone();
            let follower = tokio::spawn(async move {
                c_follow
                    .try_get_with(k_follow, || async move {
                        Ok::<_, InitError>("follower".to_string())
                    })
                    .await
            });
            for _ in 0..16 {
                tokio::task::yield_now().await;
            }

            cache.clear();

            let res = follower
                .await
                .expect("follower task panicked")
                .expect("follower returned err");
            assert_eq!(
                res, "follower",
                "follower must re-enter the probe loop and produce its own value, not the leader's"
            );

            release_leader.notify_one();
            let _ = leader.await;
        })
        .await;
    }

    /// Sync sibling of the previous test: a thread blocked inside
    /// `get_with_blocking` must also wake when `clear()` runs and
    /// install its own value on retry.
    #[test]
    fn clear_during_inflight_get_with_blocking_lets_follower_install_fresh_value() {
        use std::sync::{Barrier, mpsc};
        use std::thread;

        let cache: Arc<Cache<K, V>> = Arc::new(test_cache());
        let key = "sync_clear_follower".to_string();
        let leader_started = Arc::new(Barrier::new(2));
        let release_leader = Arc::new(Barrier::new(2));

        let c_leader = Arc::clone(&cache);
        let k_leader = key.clone();
        let started = Arc::clone(&leader_started);
        let release = Arc::clone(&release_leader);
        let leader = thread::spawn(move || {
            c_leader.get_with_blocking(k_leader, |_| {
                started.wait();
                release.wait();
                "leader".to_string()
            })
        });

        leader_started.wait();

        let (tx, rx) = mpsc::channel();
        let c_follow = Arc::clone(&cache);
        let k_follow = key.clone();
        let follower = thread::spawn(move || {
            let v = c_follow.get_with_blocking(k_follow, |_| "follower".to_string());
            let _ = tx.send(v);
        });

        thread::sleep(Duration::from_millis(80));
        cache.clear();

        let v = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("follower deadlocked: clear() did not signal sync waiters");
        assert_eq!(
            v, "follower",
            "sync follower must install its own value after the cancel-and-retry cycle"
        );
        follower.join().expect("follower thread");

        release_leader.wait();
        leader.join().expect("leader thread");
    }
}

mod insert_vs_leader {
    //! `Cache::insert` runs while a `try_get_with` / `get_with_blocking`
    //! leader's initialiser is still awaiting. The leader resumes, takes
    //! the shard lock, removes its lane, and installs its value. Without
    //! a dedup at install time, both rows survive in `entries` (and both
    //! slots in the slab); `len` / `weight` then double-count and the
    //! "ghost" slot leaks until S3-FIFO reclaim removes it.

    use super::*;

    #[tokio::test]
    async fn try_get_with_leader_install_after_concurrent_insert_keeps_one_row() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let key = "k".to_string();

            let (leader_in_init_tx, leader_in_init_rx) = tokio::sync::oneshot::channel::<()>();
            let (release_leader_tx, release_leader_rx) = tokio::sync::oneshot::channel::<()>();

            let c_leader = Arc::clone(&cache);
            let k_leader = key.clone();
            let leader = tokio::spawn(async move {
                c_leader
                    .try_get_with(k_leader, move || async move {
                        let _ = leader_in_init_tx.send(());
                        release_leader_rx.await.expect("released");
                        Ok::<_, ()>("v2".to_string())
                    })
                    .await
            });

            leader_in_init_rx.await.expect("leader entered init");

            assert_eq!(cache.insert(key.clone(), "v1".into()), None);

            release_leader_tx.send(()).expect("released");
            let leader_out = leader.await.expect("leader join").expect("leader ok");
            assert_eq!(leader_out, "v2");

            assert_eq!(cache.get(&key), Some("v2".to_string()));
            assert_eq!(
                cache.len(),
                1,
                "leader's install must not duplicate the entry produced by the racing insert",
            );
            assert_eq!(
                cache.weight(),
                1,
                "weight must reflect a single live entry, not two",
            );
        })
        .await;
    }

    #[test]
    fn get_with_blocking_leader_install_after_concurrent_insert_keeps_one_row() {
        use std::sync::{Barrier, mpsc};
        use std::thread;

        let cache: Arc<Cache<K, V>> = Arc::new(test_cache());
        let key = "sk".to_string();
        let leader_started = Arc::new(Barrier::new(2));
        let release_leader = Arc::new(Barrier::new(2));

        let c_leader = Arc::clone(&cache);
        let k_leader = key.clone();
        let started = Arc::clone(&leader_started);
        let release = Arc::clone(&release_leader);
        let (tx, rx) = mpsc::channel();
        let leader = thread::spawn(move || {
            let v = c_leader.get_with_blocking(k_leader, |_| {
                started.wait();
                release.wait();
                "v2".to_string()
            });
            let _ = tx.send(v);
        });

        leader_started.wait();
        assert_eq!(cache.insert(key.clone(), "v1".into()), None);
        release_leader.wait();

        let leader_out = rx.recv_timeout(Duration::from_secs(5)).expect("leader v");
        assert_eq!(leader_out, "v2");
        leader.join().expect("leader thread");

        assert_eq!(cache.get(&key), Some("v2".to_string()));
        assert_eq!(cache.len(), 1, "sync leader must not duplicate the row");
        assert_eq!(cache.weight(), 1);
    }
}

mod inflight_accounting {
    //! `len` and `weight` enumerate `Ready` slots only — a parked
    //! `try_get_with` leader has reserved an inflight lane but no slab
    //! slot, so it must not show up in either count until the leader
    //! actually publishes a value.

    use super::*;

    #[tokio::test]
    async fn len_and_weight_do_not_count_inflight_miss_leaders() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let key = "inflight".to_string();
            let (entered_tx, entered_rx) = tokio::sync::oneshot::channel::<()>();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();

            let c_leader = Arc::clone(&cache);
            let k_leader = key.clone();
            let leader = tokio::spawn(async move {
                c_leader
                    .try_get_with(k_leader, move || async move {
                        let _ = entered_tx.send(());
                        release_rx.await.expect("released");
                        Ok::<_, ()>("v".to_string())
                    })
                    .await
            });

            entered_rx.await.expect("leader entered init");
            assert_eq!(cache.len(), 0, "inflight leader must not count toward len");
            assert_eq!(cache.weight(), 0, "inflight leader has no installed weight");
            assert!(cache.is_empty(), "is_empty must agree with len/weight");

            release_tx.send(()).expect("released");
            let _ = leader.await.expect("leader join");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.weight(), 1);
            assert!(!cache.is_empty());
        })
        .await;
    }
}

mod clear_during_revalidate {
    //! `Cache::clear` runs while a `RefreshWork` is held by an executor that
    //! has not polled it. Clearing the cache evicts the slot that owns the
    //! `refresh_inflight` flag, so the next `try_revalidate` must not return
    //! `SkippedInFlight` (it now sees a fresh slot or an absent key, both of
    //! which are valid scheduling outcomes).

    use super::*;

    #[test]
    fn clear_during_inflight_try_revalidate_does_not_wedge_flag() {
        let cache = test_cache();
        let ex = HoldingExecutor::default();
        let k = "rev_clear".to_string();
        cache.insert(k.clone(), "old".into());
        let first = cache
            .try_revalidate(k.clone(), &ex, |_| async move {
                tokio::task::yield_now().await;
                Ok::<_, InitError>("new".to_string())
            })
            .expect("first submit");
        assert_eq!(first, RevalidateScheduled::Submitted);

        cache.clear();

        let second = cache
            .try_revalidate(k.clone(), &ex, |_| async move {
                Ok::<_, InitError>("post_clear".to_string())
            })
            .expect("second submit");
        assert_eq!(
            second,
            RevalidateScheduled::AbsentKey,
            "clear evicts the slot, so revalidate has no Ready entry to refresh"
        );

        cache.insert(k.clone(), "fresh".into());
        let third = cache
            .try_revalidate(k, &ex, |_| async move {
                Ok::<_, InitError>("after_reinsert".to_string())
            })
            .expect("third submit");
        assert_eq!(
            third,
            RevalidateScheduled::Submitted,
            "newly inserted slot must be schedulable; flag must not have leaked across clear"
        );
    }
}

mod s3_fifo {
    //! Black-box behaviours unique to the S3-FIFO eviction policy. The
    //! white-box invariants that need to peek at `EntrySlot::queue` /
    //! `State::small_weight` / `State::ghost` live in
    //! `src/cache/inner.rs::s3_fifo_tests` instead — `pub(crate)` test
    //! hooks are not visible from this integration-test crate.

    use std::time::Instant;

    use super::*;

    #[derive(Clone, Default)]
    struct LenWeightPolicy;

    impl Policy<K, V> for LenWeightPolicy {
        fn weight(&self, _k: &K, v: &V) -> Weight {
            Weight::new(v.len() as u64)
        }
        fn expire_after_read(
            &self,
            _k: &K,
            _v: &V,
            _read_at: Instant,
            _written_at: Instant,
            _time_remaining: Duration,
        ) -> Duration {
            TIME_UNBOUNDED
        }
        fn expire_after_write(&self, _k: &K, _v: &V, _written_at: Instant) -> Duration {
            TIME_UNBOUNDED
        }
    }

    fn small_cache(max: u64) -> SingleShardCache<K, V> {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        // `max` is the cap callers want inside shard 0; doubling here
        // compensates for the shard-budget rounding that `shards(2)`
        // imposes (`shard_max_weight = ceil(max_weight / 2)`).
        Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(max * 2))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(policy)
                .build(),
        )
    }

    /// A warm key (freq pushed past the `freq > 1` Small-tail promotion
    /// bar) survives a long burst of cold inserts. The headline
    /// regression test for the DS1/S2/S3 trace hit-ratio gap that
    /// motivated S3-FIFO in the first place.
    #[test]
    fn scan_does_not_evict_hot_key() {
        let cache = small_cache(20);
        let hot = "hot".to_string();
        cache.insert(hot.clone(), "v".into());
        // Three reads bumps freq to FREQ_CAP=3, well past the
        // `freq > 1` promotion bar.
        for _ in 0..3 {
            let _ = cache.get(&hot);
        }
        for i in 0..200 {
            cache.insert(format!("cold-{i}"), "v".into());
        }
        assert_eq!(
            cache.get(&hot),
            Some("v".into()),
            "warm key (freq=3) must survive a 200-entry cold scan",
        );
    }

    /// `evict_from_small` uses a strict `freq > 1` test to decide
    /// promote vs evict, so a single-read entry (`freq == 1`) is still
    /// considered cold and evicted. Distinguishes the live `>` from a
    /// `>=` slip that would let warm-once entries dodge eviction.
    #[test]
    fn freq_one_does_not_promote_on_small_reclaim() {
        let cache = small_cache(2);
        cache.insert("once".into(), "v1".into());
        let _ = cache.get("once");
        cache.insert("cold".into(), "vc".into());

        cache.insert("new".into(), "vn".into());

        assert_eq!(
            cache.get("once"),
            None,
            "freq=1 must evict on Small reclaim, not promote to Main",
        );
        assert_eq!(cache.get("cold"), Some("vc".into()));
        assert_eq!(cache.get("new"), Some("vn".into()));
    }

    /// Two slots in Small, one read twice (warm) and one untouched
    /// (cold). The next over-budget insert promotes the warm one into
    /// Main and evicts the cold one.
    #[test]
    fn low_freq_evicted_first() {
        let cache = small_cache(2);
        cache.insert("warm".into(), "vw".into());
        cache.insert("cold".into(), "vc".into());
        for _ in 0..2 {
            let _ = cache.get("warm");
        }
        cache.insert("new".into(), "vn".into());
        assert_eq!(cache.get("warm"), Some("vw".into()));
        assert_eq!(
            cache.get("cold"),
            None,
            "cold (freq=0) must lose the eviction race against warm (freq=2)",
        );
        assert_eq!(cache.get("new"), Some("vn".into()));
    }

    /// `evict_from_main`'s recirculation loop decrements `freq` each
    /// pass, so each Main slot is touched at most `FREQ_CAP + 1` times
    /// before forced eviction. This bound makes the loop terminate even
    /// when every Main entry sits at saturated `freq`. If termination
    /// were broken, every pressure insert would hang and `cargo test`
    /// would time out; the explicit `weight()` assertion adds a
    /// non-hang failure mode.
    #[test]
    fn m_recirculation_terminates() {
        let cache = small_cache(8);
        for i in 0..8 {
            cache.insert(format!("warm{i}"), "v".into());
        }
        // Four read passes saturates every slot's `freq` at FREQ_CAP=3.
        // The first over-budget insert promotes them all into Main with
        // freq reset to 0; subsequent inserts drive the Main
        // recirculation path.
        for _ in 0..4 {
            for i in 0..8 {
                let _ = cache.get(&format!("warm{i}"));
            }
        }
        for i in 0..32 {
            cache.insert(format!("p{i}"), "v".into());
            assert!(
                cache.weight() <= 8,
                "weight {} exceeded max=8 after pressure insert {i}; recirculation loop did not drop anything",
                cache.weight(),
            );
        }
    }

    /// Replacing the Small-tail entry with a heavier value triggers
    /// reclaim that immediately surfaces the skip key. The skip-key
    /// short-circuit must rotate it back to Small head and proceed
    /// with another victim instead of evicting the entry the caller is
    /// in the middle of installing.
    #[test]
    fn skip_key_short_circuit_in_small() {
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(16))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(Arc::new(LenWeightPolicy))
                .build(),
        );
        cache.insert("tail".into(), "x".into());
        cache.insert("middle".into(), "x".into());
        cache.insert("head".into(), "x".repeat(6));

        cache.insert("tail".into(), "x".repeat(5));

        assert_eq!(
            cache.get("tail"),
            Some("x".repeat(5)),
            "tail key must survive its own replacement reclaim",
        );
    }

    /// Replacing a key that lives in Main (because reads promoted it)
    /// with a heavier value forces Main-tail reclaim. The same skip-key
    /// short-circuit applies on Main as on Small.
    #[test]
    fn skip_key_short_circuit_in_main() {
        let cache = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(24))
                .shards(2)
                .hasher(SingleShardHasher)
                .policy(Arc::new(LenWeightPolicy))
                .build(),
        );
        // Phase 1: insert two keys, warm `victim` past the freq>1
        // promotion bar.
        cache.insert("victim".into(), "x".into());
        cache.insert("filler".into(), "x".repeat(10));
        for _ in 0..3 {
            let _ = cache.get("victim");
        }
        // Phase 2: trigger reclaim that promotes `victim` into Main.
        // (`filler` is the cold Small-tail; one over-budget insert is
        // enough to push `victim` into Main.)
        cache.insert("trigger".into(), "x".into());
        // `victim` should now be in Main (we have no white-box hook
        // here, but skip-key behaviour below is the load-bearing
        // assertion either way).
        assert_eq!(cache.get("victim"), Some("x".into()));

        // Phase 3: replace `victim` with a heavier value. Reclaim sees
        // the skip key in either queue first; in either case the
        // short-circuit must keep `victim` alive in the cache with the
        // new value.
        cache.insert("victim".into(), "x".repeat(10));
        assert_eq!(
            cache.get("victim"),
            Some("x".repeat(10)),
            "victim must survive replacement regardless of whether the \
             skip-key check fired in Small or Main",
        );
    }
}
