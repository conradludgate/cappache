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
