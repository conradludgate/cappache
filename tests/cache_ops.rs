//! **TDD / contract tests** for cache **data paths** (`get`, `insert`, miss helpers, `revalidate`, …).
//!
//! `Cache` operation bodies are still `todo!()`. **`cargo test --test cache_ops`** fails until the
//! core is implemented (red phase). Run **`cargo test --test cache_surface`** and **`cargo test --lib`**
//! for green checks (`RefreshWork` / `OkIntoResult` unit tests live under `src/cache.rs`).
//!
//! **Revalidate + data plane (once wired):** after `try_revalidate` returns `Submitted` and the
//! executor runs the `RefreshWork` job future to completion, `Cache::get` must show the refreshed
//! value for that key. Init errors may be returned from `try_revalidate` or only after the async
//! job—see `try_revalidate_initializer_error_propagates_as_err`.
//!
//! Prefer `get_with` + `.await` inside `#[tokio::test]` over `get_with_blocking` on the Tokio worker
//! when the initializer would need the same runtime (see crate rustdoc on blocking hazards).

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clache::{
    Cache, CacheBuilder, DefaultPolicy, Executor, LifecycleListener, LifecycleReason,
    RevalidateScheduled, TtlFromWrite, Weight,
};

type K = String;
type V = String;

fn test_cache() -> Cache<K, V> {
    Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(10_000))
            .policy(Arc::new(DefaultPolicy))
            .build(),
    )
}

fn test_cache_ttl_zero_write() -> Cache<K, V> {
    Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(10_000))
            .policy(Arc::new(TtlFromWrite::new(Duration::ZERO, DefaultPolicy)))
            .build(),
    )
}

fn test_cache_max_weight_one() -> Cache<K, V> {
    Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(1))
            .policy(Arc::new(DefaultPolicy))
            .build(),
    )
}

struct RecordingListener {
    events: Arc<Mutex<Vec<(LifecycleReason, String, Option<String>)>>>,
}

impl RecordingListener {
    fn new_pair() -> (
        Arc<Self>,
        Arc<Mutex<Vec<(LifecycleReason, String, Option<String>)>>>,
    ) {
        let events = Arc::new(Mutex::new(Vec::new()));
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
        .unwrap_or_else(|_| panic!("test exceeded {limit:?}"))
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

            let handles = [
                {
                    let c = Arc::clone(&cache);
                    let k = k.clone();
                    let b = Arc::clone(&barrier);
                    let ic = Arc::clone(&init_count);
                    tokio::spawn(async move {
                        b.wait().await;
                        c.try_get_with(k, || async move {
                            ic.fetch_add(1, Ordering::SeqCst);
                            Err::<String, ()>(())
                        })
                        .await
                    })
                },
                {
                    let c = Arc::clone(&cache);
                    let k = k.clone();
                    let b = Arc::clone(&barrier);
                    let ic = Arc::clone(&init_count);
                    tokio::spawn(async move {
                        b.wait().await;
                        c.try_get_with(k, || async move {
                            ic.fetch_add(1, Ordering::SeqCst);
                            Err::<String, ()>(())
                        })
                        .await
                    })
                },
                {
                    let c = Arc::clone(&cache);
                    let k = k.clone();
                    let b = Arc::clone(&barrier);
                    let ic = Arc::clone(&init_count);
                    tokio::spawn(async move {
                        b.wait().await;
                        c.try_get_with(k, || async move {
                            ic.fetch_add(1, Ordering::SeqCst);
                            Err::<String, ()>(())
                        })
                        .await
                    })
                },
            ];

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
        let v = cache.get_with_blocking(k.clone(), || async move {
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
        let v = cache.get_with_blocking(k.clone(), || async { "blocked".to_string() });
        assert_eq!(v, "blocked");
        assert_eq!(cache.get(&k), Some("blocked".to_string()));
    }
}

mod revalidate_tests {
    //! Scheduling-only contract today; once `try_revalidate` runs jobs, pair with an executor that
    //! drives `RefreshWork` to completion and assert `get` / lifecycle (see `src/cache.rs` tests).

    use super::*;

    #[test]
    fn try_revalidate_first_call_submitted_and_executes() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let k = "rev1".to_string();
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
        let ex = CountingExecutor::default();
        let k = "rev2".to_string();
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
        let res = cache.try_revalidate(k, &ex, |_key| async move {
            Err::<String, InitError>(InitError(42))
        });
        assert_eq!(res, Err(InitError(42)));
    }

    #[test]
    fn revalidate_submitted_when_idle() {
        let cache = test_cache();
        let ex = CountingExecutor::default();
        let k = "rev_inf".to_string();
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
        let ex = CountingExecutor::default();
        let k = "rev_inf2".to_string();
        let _ = cache
            .try_revalidate(k.clone(), &ex, |_key| async move {
                Ok::<_, InitError>("x".to_string())
            })
            .unwrap();
        let s = cache.revalidate(k, &ex, |_key| async move { "y".to_string() });
        assert_eq!(s, RevalidateScheduled::SkippedInFlight);
        assert_eq!(ex.executed_count(), 1);
    }
}

// Revalidate data plane (when core exists): after Submitted and the executor runs RefreshWork to
// Ready, get must reflect the new value; on refresh Err, get keeps the prior value unless the
// contract says otherwise; document whether try_revalidate Err(E) is sync-only or also async.

mod capacity {
    use super::*;

    #[test]
    fn max_weight_one_forbids_two_unit_weight_entries_both_visible() {
        let cache = test_cache_max_weight_one();
        assert_eq!(cache.insert("a".into(), "va".into()), None);
        assert_eq!(cache.insert("b".into(), "vb".into()), None);
        let a_live = cache.get(&"a".into()).is_some();
        let b_live = cache.get(&"b".into()).is_some();
        assert!(
            !(a_live && b_live),
            "with max_weight=1 and DefaultPolicy (weight 1 per entry), both keys must not remain \
             visible simultaneously once the core enforces the budget (eviction may be lazy on \
             insert—see DESIGN.md—relax only if the implementation deliberately defers eviction)"
        );
    }
}

mod lifecycle {
    use super::*;

    #[test]
    fn remove_after_insert_records_removed_with_prior_value() {
        let (listener, log) = RecordingListener::new_pair();
        let listener: Arc<dyn LifecycleListener<String, String> + Send + Sync> = listener;
        let cache = Cache::new(CacheBuilder::new().lifecycle_listener(listener).build());
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
    async fn try_get_with_miss_once_and_revalidate_submits_under_concurrency() {
        with_timeout(Duration::from_secs(30), async {
            let cache = Arc::new(test_cache());
            let k = "lane1".to_string();
            let ex = CountingExecutor::default();

            let miss_calls = Arc::new(AtomicUsize::new(0));
            let mc = Arc::clone(&miss_calls);
            let c_miss = Arc::clone(&cache);
            let k_miss = k.clone();
            let miss = tokio::spawn(async move {
                c_miss
                    .try_get_with(k_miss, || async move {
                        mc.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, ()>("mv".to_string())
                    })
                    .await
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

            let (m, r) = tokio::join!(miss, rev);
            let miss_res = m.expect("miss join").expect("try_get_with");
            let rev_res = r.expect("rev join").expect("try_revalidate");
            assert_eq!(miss_res, "mv");
            assert_eq!(rev_res, RevalidateScheduled::Submitted);
            assert_eq!(ex.executed_count(), 1);
            assert_eq!(miss_calls.load(Ordering::SeqCst), 1);
        })
        .await;
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
