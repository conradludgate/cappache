//! Run a single future on the current thread (park / unpark).
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

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

/// Runs a [`Future`] to completion on the current thread (pollster-style).
///
/// Crate-private; see the [module documentation](self) for when this is safe to use.
pub(crate) fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    let mut pinned = pin!(future);
    let thread = thread::current();
    let waker = Waker::from(Arc::new(ParkerWake(thread)));
    let mut cx = Context::from_waker(&waker);

    loop {
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => thread::park(),
        }
    }
}

struct ParkerWake(Thread);

impl Wake for ParkerWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
