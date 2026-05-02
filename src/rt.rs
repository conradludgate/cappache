//! Hooks for driving asynchronous work with the caller’s runtime.
//!
//! [`Executor`] is shaped like [hyper’s `Executor`](https://docs.rs/hyper/latest/hyper/rt/trait.Executor.html):
//! a small abstraction so **`clache`** stays executor-agnostic (no Tokio dependency).
//!
//! Use it with [`Cache::revalidate`](crate::Cache::revalidate) / [`Cache::try_revalidate`](crate::Cache::try_revalidate)
//! to **spawn** background refresh jobs.
//! That path requires the scheduled future to be [`Send`] + `'static` for typical
//! multi-threaded executors; [`Cache::get_with`](crate::Cache::get_with) may use
//! **non-`Send`** futures when you **`.await`** it on a single-threaded context (e.g.
//! `tokio::task::LocalSet`).
//!
//! # Blanket `impl`s (same idea as hyper)
//!
//! Your executor type usually implements **`Executor<F>` for all `F`** that satisfy bounds, e.g.
//! `F: Future + Send + 'static` and `F::Output: Send + 'static` for `tokio::spawn`. The cache passes
//! a concrete job type [`crate::cache::RefreshWork`] (parameterized by your initializer future); that
//! `RefreshWork<Fut>` is what appears in **`Executor<RefreshWork<Fut>>`** at the call site—not your
//! raw `Fut` type alone.

/// Runs a future on some runtime (e.g. `tokio::spawn`, a local pool, or tests).
///
/// Prior art: [`hyper::rt::Executor`](https://docs.rs/hyper/latest/hyper/rt/trait.Executor.html).
///
/// If this name clashes with another `Executor` in scope, qualify as **`clache::rt::Executor`**.
pub trait Executor<Fut> {
    /// Schedules `fut` to run to completion on the underlying runtime.
    fn execute(&self, fut: Fut);
}

/// Whether [`Cache::revalidate`](crate::Cache::revalidate) or [`Cache::try_revalidate`](crate::Cache::try_revalidate)
/// submitted work to the [`Executor`].
///
/// **`SkippedInFlight`** means a refresh was already in flight for that key (per
/// [`Cache::try_revalidate`](crate::Cache::try_revalidate)’s `refresh_inflight` tracking) and a second
/// backend call was not started.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RevalidateScheduled {
    /// A refresh job was handed to the [`Executor`].
    Submitted,
    /// A refresh was already in flight for this key.
    SkippedInFlight,
}
