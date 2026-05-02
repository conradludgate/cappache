//! Concurrent cache handle: synchronous lookups and updates; async for insert-on-miss helpers.
//!
//! [`get`](Cache::get), [`insert`](Cache::insert), and [`remove`](Cache::remove) are **synchronous**.
//! On a miss, [`get_with`](Cache::get_with) / [`try_get_with`](Cache::try_get_with) run an initializer.
//! Use [`get_with_blocking`](Cache::get_with_blocking) to run [`get_with`](Cache::get_with) on the current thread via the
//! internal park driver (not a public API).
//!
//! # Caller-driven refresh
//!
//! [`Cache::revalidate`](Cache::revalidate) and [`try_revalidate`](Cache::try_revalidate) schedule a background
//! refresh using an injected [`crate::rt::Executor`] (same idea as [hyper’s `Executor`](https://docs.rs/hyper/latest/hyper/rt/trait.Executor.html)).
//! There is **no** `revalidate_blocking` — refresh is meant to be **spawned**, not park-driven.
//! Method bodies are still [`todo!()`]; semantics are documented for the future implementation.
//!
//! # `Send` on initializer futures
//!
//! - **`get_with` / `try_get_with`**: the initializer future does **not** need to be [`Send`]
//!   when you only `.await` on the current async context (e.g. `tokio::task::LocalSet`).
//! - **`revalidate` / `try_revalidate`**: once wired up, the job handed to [`crate::rt::Executor::execute`]
//!   will typically need to be [`Send`] + `'static` for runtimes like `tokio::spawn` (see [`crate::rt`]).
//!
//! # Fallible init (`E: Clone`)
//!
//! [`try_get_with`](Cache::try_get_with) and [`try_revalidate`](Cache::try_revalidate) use **`Result<V, E>`** with
//! **`E: Clone`** instead of `Arc<E>` so error types from newer crates can be duplicated cheaply. [`get_with`](Cache::get_with)
//! and [`revalidate`](Cache::revalidate) wrap the fallible APIs with [`std::convert::Infallible`] as the error type
//! (via [`OkIntoResult`] for the refresh path).
//!
//! # Blocking and async runtimes
//!
//! [`get_with_blocking`](Cache::get_with_blocking) uses thread **park** / **unpark**. Do not
//! call it from a future executing on a typical async worker pool if the initializer may need that
//! same pool to complete (risk of **deadlock** or long stalls). Prefer the async helpers under an
//! async runtime, or call blocking only from threads where blocking is safe (main thread, dedicated
//! blocking pool, etc.).

use std::convert::Infallible;
use std::future::Future;
use std::hash::Hash;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};

use pin_project_lite::pin_project;

use crate::blocking;
use crate::config::CacheConfig;
use crate::lifecycle::LifecycleListener;
use crate::policy::{Policy, Weight};
use crate::rt::{Executor, RevalidateScheduled};

pin_project! {
    /// Wraps `Fut: Future<Output = V>` as `Future<Output = Result<V, Infallible>>`.
    ///
    /// Used so infallible initializers share the same [`try_get_with`](Cache::try_get_with) /
    /// [`try_revalidate`](Cache::try_revalidate) machinery as fallible paths, with a **named** future type
    /// instead of an anonymous `async` block (so [`Executor`](crate::rt::Executor)`<RefreshWork<…>>` bounds line up).
    pub struct OkIntoResult<Fut> {
        #[pin]
        inner: Fut,
    }
}

impl<Fut> OkIntoResult<Fut> {
    #[must_use]
    pub fn new(inner: Fut) -> Self {
        Self { inner }
    }
}

impl<Fut, V> Future for OkIntoResult<Fut>
where
    Fut: Future<Output = V>,
{
    type Output = Result<V, Infallible>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.project().inner.poll(cx).map(Ok)
    }
}

/// Job future type for [`Cache::revalidate`](Cache::revalidate)’s [`crate::rt::Executor`] bound.
///
/// You do **not** construct this type. The cache will build it in a later phase: it will own the
/// per-key single-flight, run your `init(key)` future, install the new `V`, and wake waiters—same
/// gate as a **miss** on [`try_get_with`](Cache::try_get_with) / [`get_with`](Cache::get_with).
///
/// Today it exists so call sites type-check against **`Executor<RefreshWork<Fut>>`** (same
/// pattern as hyper’s `Executor` over concrete connection futures). [`poll`](Future::poll) is
/// [`todo!()`] until then.
pub struct RefreshWork<Fut> {
    pub(crate) _marker: PhantomData<Fut>,
}

impl<Fut> Default for RefreshWork<Fut> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<Fut> Future for RefreshWork<Fut> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}

/// A `Sync` concurrent key-value cache with weighted capacity and per-entry [`Policy`].
///
/// Requires `K: Eq + Hash + Clone + Send + Sync` and `V: Clone + Send + Sync` (same as
/// [`CacheBuilder`](crate::CacheBuilder)).
///
/// Weight and time limits come from [`CacheConfig::policy`] (set via [`crate::CacheBuilder::policy`]).
///
/// # `get_with` concurrency
///
/// When multiple threads race to insert the same missing key, **at most one** initializer runs to
/// completion for that key; others await the result (**single-writer** style insert path). Exact
/// blocking behavior is implementation-defined in later phases.
pub struct Cache<K, V>
where
    K: 'static,
    V: 'static,
{
    pub(crate) _inner: Arc<RwLock<()>>,
    pub(crate) config: CacheConfig<K, V>,
    pub(crate) _marker: PhantomData<(K, V)>,
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    #[must_use]
    pub fn new(config: CacheConfig<K, V>) -> Self {
        Self {
            _inner: Arc::new(RwLock::new(())),
            config,
            _marker: PhantomData,
        }
    }

    /// Returns a clone of the value for `key`, if present and still served per [`Self::policy_ref`]
    /// (eviction / policy — not “application staleness” for [`Self::revalidate`]).
    #[must_use]
    pub fn get(&self, _key: &K) -> Option<V> {
        todo!()
    }

    /// Inserts `value` for `key`, returning any previous value.
    ///
    /// Uses [`Self::policy_ref`] for weight and time limits when implemented.
    #[must_use]
    pub fn insert(&self, _key: K, _value: V) -> Option<V> {
        todo!()
    }

    /// Removes `key`, returning the previous value if it existed.
    #[must_use]
    pub fn remove(&self, _key: &K) -> Option<V> {
        todo!()
    }

    /// On a miss, runs `init` once (single-flight with other waiters); on failure returns a **clone**
    /// of the error (`E: Clone` instead of `Arc<E>`).
    ///
    /// The `init` future must not be awaited while holding the cache’s synchronous lock; the
    /// implementation will enforce that once the cache core exists—this stub does not.
    pub async fn try_get_with<F, Fut, E>(&self, _key: K, _init: F) -> Result<V, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<V, E>>,
        E: Clone,
    {
        todo!()
    }

    /// Returns the existing value for `key`, or on a miss runs `init` once (single-flight with other
    /// waiters) and stores the result per [`Self::policy_ref`] (same as [`Self::insert`]).
    ///
    /// Like [`Self::try_get_with`] with `Fut::Output = Result<V, Infallible>` — i.e. an infallible
    /// initializer — so this always resolves to `V` once implemented.
    ///
    /// `init` is **`FnOnce() -> Fut`** (no `K` argument) so you can close over `key` or `&key` from
    /// the outer call; [`Self::revalidate`] takes **`FnOnce(K) -> Fut`** so the scheduled job owns
    /// `K` inside the spawned future.
    ///
    /// The initializer future does **not** need to be [`Send`] when you only `.await` this method
    /// on the current async context (e.g. `tokio::task::LocalSet`).
    ///
    /// The `init` future must not be awaited while holding the cache’s synchronous lock; the
    /// implementation will enforce that once the cache core exists—this stub does not.
    pub async fn get_with<F, Fut>(&self, key: K, init: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = V>,
    {
        match self.try_get_with(key, || OkIntoResult::new(init())).await {
            Ok(v) => v,
            Err(e) => match e {},
        }
    }

    /// Blocking version of [`Self::get_with`].
    ///
    /// Uses the crate’s internal blocking driver (park / unpark). See the **Blocking and async runtimes**
    /// section in the [`cache`](crate::cache) module documentation for when this is safe (avoid from async worker threads if the
    /// initializer needs that same runtime).
    #[must_use]
    pub fn get_with_blocking<F, Fut>(&self, key: K, init: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = V>,
    {
        blocking::block_on(self.get_with(key, init))
    }

    /// Fallible background refresh: `init(key)` must resolve to `Result<V, E>` with **`E: Clone`**.
    ///
    /// Same [`crate::rt::Executor`]`<RefreshWork<Fut>>` story as [`Self::revalidate`] (there `Fut` is
    /// [`OkIntoResult`]`<…>`). When implemented,
    /// coalescing / error propagation will follow the same intent as [`Self::try_get_with`] (including
    /// that different `Fut` / `E` lanes may not dedupe with each other, mirroring moka’s `(key, TypeId)` pattern).
    ///
    /// **Phase A:** body is [`todo!()`].
    #[must_use]
    pub fn try_revalidate<Ex, F, Fut, E>(
        &self,
        _key: K,
        _executor: &Ex,
        _init: F,
    ) -> Result<RevalidateScheduled, E>
    where
        Ex: Executor<RefreshWork<Fut>>,
        F: FnOnce(K) -> Fut,
        Fut: Future<Output = Result<V, E>> + Send + 'static,
        E: Clone + Send + Sync + 'static,
    {
        todo!()
    }

    /// Schedules a background job that will run `init(key)` under the same per-key single-flight as a
    /// **miss** on [`Self::try_get_with`], without removing the existing entry first.
    ///
    /// Wrapper around [`Self::try_revalidate`]: `init(key)` is wrapped in [`OkIntoResult`] so the
    /// fallible path sees `Result<V, Infallible>` (always `Ok` once the inner future completes).
    ///
    /// The cache does **not** decide staleness — call only when **your** logic says to refresh.
    ///
    /// The executor bound is **[`crate::rt::Executor`]<[`RefreshWork`]<[`OkIntoResult`]<`Fut`>>>** — the
    /// same `Fut` you return from `init`, not `Executor<Fut>` alone: the cache will hand your runtime a
    /// concrete [`RefreshWork`] future (see [`crate::rt`] module docs).
    ///
    /// When implemented, if a refresh is already in flight for `key`, returns
    /// [`RevalidateScheduled::SkippedInFlight`] and does not spawn a second job. The job future passed
    /// to [`crate::rt::Executor::execute`] will typically need to be [`Send`] +
    /// `'static` for executors like `tokio::spawn`.
    ///
    /// Intended usage (syntax may use `async move` until async closures stabilize):
    ///
    /// ```ignore
    /// cache.revalidate(key, &executor, |key| async move { fetch(key).await });
    /// ```
    #[must_use]
    pub fn revalidate<Ex, F, Fut>(&self, key: K, executor: &Ex, init: F) -> RevalidateScheduled
    where
        Ex: Executor<RefreshWork<OkIntoResult<Fut>>>,
        F: FnOnce(K) -> Fut,
        Fut: Future<Output = V> + Send + 'static,
    {
        match self.try_revalidate(key, executor, |k| OkIntoResult::new(init(k))) {
            Ok(s) => s,
            Err(e) => match e {},
        }
    }

    #[must_use]
    pub fn max_weight(&self) -> Weight {
        self.config.max_weight
    }

    /// Clone of the shared [`Policy`] handle (increments the `Arc` refcount).
    #[must_use]
    pub fn policy(&self) -> Arc<dyn Policy<K, V> + Send + Sync> {
        Arc::clone(&self.config.policy)
    }

    /// Borrow the installed [`Policy`] without cloning the `Arc`.
    #[must_use]
    pub fn policy_ref(&self) -> &(dyn Policy<K, V> + Send + Sync) {
        self.config.policy.as_ref()
    }

    #[must_use]
    pub fn lifecycle_listener(&self) -> Option<&Arc<dyn LifecycleListener<K, V> + Send + Sync>> {
        self.config.lifecycle_listener.as_ref()
    }
}
