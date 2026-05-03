//! Run a single future on the current thread until it completes.
//!
//! The blocking driver is **pollster** logic inlined from **[pollster](https://github.com/zesterer/pollster)**
//! **0.4.0** (not a crates.io dependency). SPDX: **Apache-2.0 OR MIT**.
//!
//! # Correct use
//!
//! Intended for [`crate::Cache::get_with_blocking`] and similar call sites that have **no**
//! async executor or are already on a thread where blocking is acceptable.
//!
//! # Hazards
//!
//! - **Do not** call [`block_on`] from a future running on a work-stealing runtime (e.g. Tokio) if
//!   the inner future may need that same runtime to make progress — you risk **deadlock** or severe
//!   stalls.
//! - **Avoid** nesting `block_on` calls, or calling `block_on` from an initializer used with
//!   [`crate::Cache::get_with_blocking`], unless the inner future cannot wait on the outer one.

use std::future::IntoFuture;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};

// --- begin vendored pollster 0.4.0 (https://github.com/zesterer/pollster) ---

enum SignalState {
    Empty,
    Waiting,
    Notified,
}

struct Signal {
    state: Mutex<SignalState>,
    cond: Condvar,
}

impl Signal {
    fn new() -> Self {
        Self {
            state: Mutex::new(SignalState::Empty),
            cond: Condvar::new(),
        }
    }

    fn wait(&self) {
        let mut state = self.state.lock().unwrap();
        match *state {
            SignalState::Notified => *state = SignalState::Empty,
            SignalState::Waiting => {
                unreachable!("Multiple threads waiting on the same signal: Open a bug report!");
            }
            SignalState::Empty => {
                *state = SignalState::Waiting;
                while let SignalState::Waiting = *state {
                    state = self.cond.wait(state).unwrap();
                }
            }
        }
    }

    fn notify(&self) {
        let mut state = self.state.lock().unwrap();
        match *state {
            SignalState::Notified => {}
            SignalState::Empty => *state = SignalState::Notified,
            SignalState::Waiting => {
                *state = SignalState::Empty;
                self.cond.notify_one();
            }
        }
    }
}

impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.notify();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}

/// Runs a [`std::future::IntoFuture`] to completion on the current thread (pollster `block_on`).
///
/// Crate-private; see the [module documentation](self) for when this is safe to use.
pub(crate) fn block_on<F: IntoFuture>(fut: F) -> F::Output {
    let mut fut = std::pin::pin!(fut.into_future());

    let signal = Arc::new(Signal::new());

    let waker = Waker::from(Arc::clone(&signal));
    let mut context = Context::from_waker(&waker);

    loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Pending => signal.wait(),
            Poll::Ready(item) => break item,
        }
    }
}

// --- end vendored pollster ---

#[cfg(test)]
mod tests {
    use std::future::poll_fn;
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::Duration;

    use super::block_on;

    fn poll_once_pending_then_ready(
        waker_slot: Arc<Mutex<Option<std::task::Waker>>>,
    ) -> impl std::future::Future<Output = u8> {
        poll_fn(move |cx| {
            let mut g = waker_slot.lock().expect("waker slot mutex poisoned");
            if g.is_none() {
                *g = Some(cx.waker().clone());
                std::task::Poll::Pending
            } else {
                std::task::Poll::Ready(7_u8)
            }
        })
    }

    #[test]
    fn block_on_unparks_when_waker_wakes_by_ref_from_another_thread() {
        let waker_slot: Arc<Mutex<Option<std::task::Waker>>> = Arc::new(Mutex::new(None));
        let ws = Arc::clone(&waker_slot);
        let (done_tx, done_rx) = mpsc::channel();
        let slot_for_block = Arc::clone(&waker_slot);
        let blocker = std::thread::spawn(move || {
            let out = block_on(async { poll_once_pending_then_ready(slot_for_block).await });
            let _ = done_tx.send(out);
        });
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            let g = ws.lock().expect("waker slot mutex poisoned");
            if let Some(w) = g.as_ref() {
                w.wake_by_ref();
            }
        });

        let out = done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("block_on did not finish (wake_by_ref may be broken)");
        assert_eq!(out, 7);
        worker.join().expect("wake thread panicked");
        blocker.join().expect("block_on thread panicked");
    }

    #[test]
    #[allow(clippy::waker_clone_wake)]
    fn block_on_unparks_when_waker_wake_consumed_from_another_thread() {
        let waker_slot: Arc<Mutex<Option<std::task::Waker>>> = Arc::new(Mutex::new(None));
        let ws = Arc::clone(&waker_slot);
        let (done_tx, done_rx) = mpsc::channel();
        let slot_for_block = Arc::clone(&waker_slot);
        let blocker = std::thread::spawn(move || {
            let out = block_on(async { poll_once_pending_then_ready(slot_for_block).await });
            let _ = done_tx.send(out);
        });
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            let g = ws.lock().expect("waker slot mutex poisoned");
            if let Some(w) = g.as_ref() {
                w.clone().wake();
            }
        });

        let out = done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("block_on did not finish (wake may be broken)");
        assert_eq!(out, 7);
        worker.join().expect("wake thread panicked");
        blocker.join().expect("block_on thread panicked");
    }
}
