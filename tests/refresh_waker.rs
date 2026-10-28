//! Property-based regression for `RefreshWork::poll` waker forwarding.
//!
//! `RefreshWork::poll` is a thin pass-through to its inner init future
//! — but a future change that intercepted, replaced, or stored the
//! `Waker` would silently break callers waiting on long-running
//! refresh jobs. `futures-testing` fuzz-interleaves polls, drives,
//! waker swaps, and cancellations on the future under test, asserting
//! the waker was woken whenever the driver reported progress; it
//! catches lost / stale / cancel-unsafe wakers that deterministic
//! tests miss.
//!
//! The state-mutation races (Bug #1, Bug #2) live in
//! `tests/refresh_concurrency.rs`; this file is the lightweight
//! property test for the `Future` plumbing itself.

use std::future::Future;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use cappache::{Cache, CacheBuilder, Executor, Weight};
use futures_testing::{drive_fn, testcase};
use tokio::sync::{Mutex as AsyncMutex, mpsc};

#[derive(Debug, Clone, PartialEq, Eq)]
struct InitError;

type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

#[derive(Default)]
struct CapturingExecutor {
    captured: Mutex<Vec<BoxFuture>>,
}

impl CapturingExecutor {
    fn take_latest(&self) -> Option<BoxFuture> {
        self.captured.lock().unwrap().pop()
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

#[test]
fn refresh_work_forwards_waker_correctly() {
    futures_testing::tests(testcase!(|| {
        let cache: Cache<String, String> =
            Cache::new(CacheBuilder::new().max_weight(Weight::new(10_000)).build());
        cache.insert("k".into(), "v0".into());

        let (tx, rx) = mpsc::channel::<()>(4);
        let driver = drive_fn(async move |()| match tx.send(()).await {
            Ok(()) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        });

        let rx = Arc::new(AsyncMutex::new(rx));
        let factory = async move |()| {
            let ex = CapturingExecutor::default();
            let rx_clone = Arc::clone(&rx);
            let _ = cache.try_revalidate("k".to_string(), &ex, move |_| async move {
                let mut guard = rx_clone.lock().await;
                match guard.recv().await {
                    Some(()) => Ok::<String, InitError>("v".to_string()),
                    None => Err(InitError),
                }
            });
            if let Some(work) = ex.take_latest() {
                work.await;
            }
        };

        (driver, factory)
    }))
    .run();
}
