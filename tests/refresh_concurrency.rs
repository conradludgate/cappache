//! Deterministic regression tests for `try_revalidate` concurrency bugs.
//!
//! Both bugs in scope here are races that close in nanoseconds without
//! explicit synchronisation. Rather than spinning on probabilistic
//! timing (or forking `clashcore` for `loom`/`shuttle`), we exploit two
//! synchronisation hooks the cache already exposes:
//!
//! 1. `LifecycleListener::on_event` runs *outside* the shard write
//!    lock (see `apply_refresh_ok` in `src/cache/inner.rs`), so a
//!    listener that blocks on a `Barrier` widens the post-install
//!    window into a fully ordered single-step transition.
//! 2. The `Executor` trait is hand-rollable, so a `CapturingExecutor`
//!    parks each `RefreshWork` future and lets the test thread step
//!    it with `Waker::noop()` (Rust 1.95+).
//!
//! `#[test]` functions live in submodules added by the follow-up
//! commits; this file currently carries scaffolding only.

#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Barrier, Mutex};
use std::task::{Context, Poll, Waker};

use clache::{Executor, LifecycleListener, LifecycleReason};

type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Captures every future submitted to it so the test thread can drive
/// each one to completion explicitly. The cache treats the executor as
/// fire-and-forget, so capture order matches submission order.
#[derive(Default)]
pub(crate) struct CapturingExecutor {
    captured: Mutex<Vec<BoxFuture>>,
}

impl CapturingExecutor {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Pops the most-recently submitted future. Panics if nothing has
    /// been captured yet — surface that loudly so a missing submit is
    /// not silently swallowed.
    pub(crate) fn take_latest(&self) -> BoxFuture {
        self.captured
            .lock()
            .unwrap()
            .pop()
            .expect("CapturingExecutor: no future submitted")
    }

    pub(crate) fn submitted(&self) -> usize {
        self.captured.lock().unwrap().len()
    }
}

impl<F> Executor<F> for CapturingExecutor
where
    F: Future<Output = ()> + Send + 'static,
{
    fn execute(&self, fut: F) {
        self.captured.lock().unwrap().push(Box::pin(fut));
    }
}

/// Lifecycle listener that pauses the firing thread inside a matching
/// `Updated` event. The listener does `entered.wait()` followed by
/// `release.wait()` on each `Updated`; the test thread coordinates by
/// arriving at the same barriers from the other side.
///
/// Other lifecycle reasons (e.g. `EvictedSize` from incidental reclaim)
/// pass through untouched so they cannot consume a barrier slot
/// accidentally.
pub(crate) struct BlockingListener {
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl BlockingListener {
    /// Builds a listener and returns the entered/release barriers the
    /// test thread should `wait()` on.
    pub(crate) fn pair() -> (Arc<Self>, Arc<Barrier>, Arc<Barrier>) {
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let listener = Arc::new(Self {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        });
        (listener, entered, release)
    }
}

impl LifecycleListener<String, String> for BlockingListener {
    fn on_event(&self, reason: LifecycleReason, _key: String, _value: Option<String>) {
        if reason == LifecycleReason::Updated {
            self.entered.wait();
            self.release.wait();
        }
    }
}

/// Drives a captured future to `Ready(())` using `Waker::noop()`. The
/// init futures these tests build (`async { Ok(_) }`) settle in a
/// single poll; the `yield_now` keeps the helper safe if a future ever
/// returns `Pending` for a Drop or finalisation step.
pub(crate) fn drive_to_completion(mut fut: BoxFuture) {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(()) => return,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InitError(u32);

mod bug1 {
    use std::thread;

    use clache::{Cache, CacheBuilder, RevalidateScheduled, Weight};

    use super::{
        Arc, BlockingListener, CapturingExecutor, InitError, LifecycleListener,
        drive_to_completion,
    };

    /// Bug #1: between `apply_refresh_ok` releasing the shard lock and
    /// `RefreshWork::poll`'s post-`apply_refresh_ok`
    /// `refresh_inflight_remove`, a racing `try_revalidate` can mark
    /// the freshly installed slot in-flight, only to have its flag
    /// wiped by the redundant remove. A subsequent submission then
    /// sees a clear slot and schedules a duplicate refresh.
    ///
    /// Fixed in: `fix: drop redundant refresh_inflight_remove on Ok`.
    #[test]
    fn refresh_completion_does_not_clear_inflight_on_freshly_installed_slot() {
        let (listener, entered, release) = BlockingListener::pair();
        let listener: Arc<dyn LifecycleListener<String, String> + Send + Sync> = listener;
        let cache: Cache<String, String> = Cache::new(
            CacheBuilder::new()
                .max_weight(Weight::new(10_000))
                .lifecycle_listener(listener)
                .build(),
        );
        let ex = CapturingExecutor::new();
        let k = "k".to_string();
        cache.insert(k.clone(), "old".into());

        assert_eq!(
            cache
                .try_revalidate(k.clone(), &*ex, |_| async move {
                    Ok::<_, InitError>("new".to_string())
                })
                .expect("submit ok"),
            RevalidateScheduled::Submitted,
        );

        let work = ex.take_latest();
        let driver = thread::spawn(move || drive_to_completion(work));

        // Wait until `apply_refresh_ok` has released the shard lock and
        // entered the lifecycle callback. The freshly installed slot
        // is now visible with `refresh_inflight=false`, but the
        // post-`apply_refresh_ok` `refresh_inflight_remove` has not
        // run yet.
        entered.wait();

        // Race a fresh `try_revalidate` into that window. It marks the
        // new slot in-flight (true) and parks another `RefreshWork`.
        assert_eq!(
            cache
                .try_revalidate(k.clone(), &*ex, |_| async move {
                    Ok::<_, InitError>("racing".to_string())
                })
                .expect("submit ok"),
            RevalidateScheduled::Submitted,
        );

        // Release the listener; the driver runs the (pre-fix) redundant
        // `refresh_inflight_remove`, wiping the racing flag.
        release.wait();
        driver.join().expect("driver panicked");

        // A third submission must observe the racing job still in
        // flight. Pre-fix the flag was cleared, so this returns
        // `Submitted` (duplicate refresh). Post-fix it returns
        // `SkippedInFlight`.
        assert_eq!(
            cache
                .try_revalidate(k.clone(), &*ex, |_| async move {
                    Ok::<_, InitError>("third".to_string())
                })
                .expect("submit ok"),
            RevalidateScheduled::SkippedInFlight,
            "racing refresh's in-flight flag was wiped by the redundant remove",
        );
    }
}

mod bug2 {
    use clache::{Cache, CacheBuilder, RevalidateScheduled, Weight};

    use super::{CapturingExecutor, InitError, drive_to_completion};

    /// Bug #2: `apply_refresh_ok` resurrects a refreshed value into a
    /// slot the user already removed. The race here is sequential —
    /// `try_revalidate` schedules the job, the user `remove`s the
    /// entry before the executor runs, and the refresh nonetheless
    /// installs the value. This contradicts the documented
    /// `RevalidateScheduled::AbsentKey` invariant: refresh must not
    /// promote into install for an entry the user has decided to
    /// drop.
    ///
    /// Fixed in: `fix: skip refresh install when slot identity changed`.
    #[test]
    fn refresh_does_not_resurrect_removed_entry() {
        let cache: Cache<String, String> = Cache::new(
            CacheBuilder::new().max_weight(Weight::new(10_000)).build(),
        );
        let ex = CapturingExecutor::new();
        let k = "k".to_string();
        cache.insert(k.clone(), "old".into());

        assert_eq!(
            cache
                .try_revalidate(k.clone(), &*ex, |_| async move {
                    Ok::<_, InitError>("new".to_string())
                })
                .expect("submit ok"),
            RevalidateScheduled::Submitted,
        );

        assert_eq!(cache.remove(&k), Some("old".into()));

        drive_to_completion(ex.take_latest());

        assert!(
            cache.get(&k).is_none(),
            "removed entry must stay removed after a refresh completes",
        );
    }

    /// Bug #2 (variant): if the user `remove`s the entry mid-refresh
    /// and then `insert`s a fresh value, the in-flight refresh must
    /// not stomp the user's new value.
    ///
    /// Fixed in: `fix: skip refresh install when slot identity changed`.
    #[test]
    fn refresh_does_not_overwrite_reinserted_entry() {
        let cache: Cache<String, String> = Cache::new(
            CacheBuilder::new().max_weight(Weight::new(10_000)).build(),
        );
        let ex = CapturingExecutor::new();
        let k = "k".to_string();
        cache.insert(k.clone(), "old".into());

        assert_eq!(
            cache
                .try_revalidate(k.clone(), &*ex, |_| async move {
                    Ok::<_, InitError>("refreshed".to_string())
                })
                .expect("submit ok"),
            RevalidateScheduled::Submitted,
        );

        let _ = cache.remove(&k);
        cache.insert(k.clone(), "user".into());

        drive_to_completion(ex.take_latest());

        assert_eq!(
            cache.get(&k),
            Some("user".into()),
            "user-driven re-insert must not be overwritten by the in-flight refresh",
        );
    }
}
