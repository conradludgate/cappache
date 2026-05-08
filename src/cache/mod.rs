//! In-memory concurrent cache: synchronous `get` / `insert` / `remove`, async miss path, and
//! executor-driven refresh.

mod clock;
mod expiry;
mod flight;
mod inner;
mod miss;
mod reclaim;
mod slab;

use std::borrow::Borrow;
use std::convert::Infallible;
use std::future::Future;
use std::hash::{BuildHasher, Hash};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use hashbrown::DefaultHashBuilder;
use pin_project_lite::pin_project;

use crate::Weight;
use crate::config::{CacheConfig, DynPolicy};
use crate::lifecycle::LifecycleListener;
use crate::policy::Policy;
use crate::rt::{Executor, RevalidateScheduled};

pin_project! {
    /// Wraps a `Future<Output = T>` as `Future<Output = Result<T, Infallible>>`
    /// so refresh executors stay homogenous over `Result<V, E>`.
    pub(crate) struct OkIntoResult<Fut> {
        #[pin]
        fut: Fut,
    }
}

impl<Fut> OkIntoResult<Fut> {
    pub(crate) fn new(fut: Fut) -> Self {
        Self { fut }
    }
}

impl<Fut: Future> Future for OkIntoResult<Fut> {
    type Output = Result<Fut::Output, Infallible>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.project().fut.poll(cx).map(Ok)
    }
}

/// Clears `refresh_inflight` if `try_revalidate` unwinds before
/// handing the flag off to the spawned [`RefreshWork`].
struct RefreshGuard<'a, K, V, P, S>
where
    K: Eq + Hash + 'static,
    V: 'static,
    P: 'static,
    S: BuildHasher + 'static,
{
    inner: &'a Arc<inner::Inner<K, V, P, S>>,
    key: &'a K,
    armed: bool,
}

impl<K, V, P, S> Drop for RefreshGuard<'_, K, V, P, S>
where
    K: Eq + Hash + 'static,
    V: 'static,
    P: 'static,
    S: BuildHasher + 'static,
{
    fn drop(&mut self) {
        if self.armed {
            self.inner.refresh_inflight_remove(self.key);
        }
    }
}

/// Job handed to [`Executor::execute`](crate::rt::Executor::execute) by
/// [`Cache::try_revalidate`]. `cleanup` clears `refresh_inflight` if
/// the executor drops the job without polling it to `Ready`.
pub struct RefreshWork<
    K: 'static,
    V: 'static,
    E: 'static,
    P: 'static = DynPolicy<K, V>,
    S: 'static = DefaultHashBuilder,
> {
    pub(crate) inner: Arc<inner::Inner<K, V, P, S>>,
    pub(crate) key: K,
    pub(crate) fut: Pin<Box<dyn Future<Output = Result<V, E>> + Send>>,
    pub(crate) cleanup: fn(&inner::Inner<K, V, P, S>, &K),
    pub(crate) cleaned: bool,
}

impl<K, V, E, P, S> Future for RefreshWork<K, V, E, P, S>
where
    K: Eq + Hash + Clone + Send + Sync + Unpin + 'static,
    V: Clone + Send + Sync + 'static,
    E: Send + 'static,
    P: Policy<K, V> + Send + Sync + 'static,
    S: BuildHasher + Clone + Send + Sync + 'static,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.fut.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(v)) => {
                this.inner.apply_refresh_ok(&this.key, v);
                this.cleaned = true;
                Poll::Ready(())
            }
            Poll::Ready(Err(_)) => {
                this.inner.refresh_inflight_remove(&this.key);
                this.cleaned = true;
                Poll::Ready(())
            }
        }
    }
}

impl<K, V, E, P, S> Drop for RefreshWork<K, V, E, P, S>
where
    K: 'static,
    V: 'static,
    E: 'static,
    P: 'static,
    S: 'static,
{
    fn drop(&mut self) {
        if !self.cleaned {
            (self.cleanup)(&self.inner, &self.key);
        }
    }
}

/// Concurrent in-memory cache.
///
/// `Cache` is cheap to clone: each handle is an `Arc` to the same shared
/// state, so call sites can hand out clones to spawned workers without
/// rebuilding configuration. The underlying state lives until the last
/// handle is dropped.
///
/// `P` is the policy type. The default ([`DynPolicy`]) keeps the cache
/// generic-erased; pass a concrete `P: Policy<K, V>` to inline policy
/// hooks and skip the vtable on every cache operation.
pub struct Cache<
    K: 'static,
    V: 'static,
    P: 'static = DynPolicy<K, V>,
    S: 'static = DefaultHashBuilder,
> {
    inner: Arc<inner::Inner<K, V, P, S>>,
}

impl<K: 'static, V: 'static, P: 'static, S: 'static> Clone for Cache<K, V, P, S> {
    /// Cloning a `Cache` is `Arc`-cheap: both handles refer to the same
    /// shared state. Mutations from one are visible from the other.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K, V, P, S> Cache<K, V, P, S>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    P: Policy<K, V> + Clone + Send + Sync + 'static,
    S: BuildHasher + Clone + Send + Sync + 'static,
{
    #[must_use]
    pub fn new(config: CacheConfig<K, V, P, S>) -> Self {
        Self {
            inner: Arc::new(inner::Inner::new(config)),
        }
    }

    #[must_use]
    pub fn max_weight(&self) -> Weight {
        self.inner.config.max_weight
    }

    /// Returns a clone of the policy. For the default `DynPolicy`
    /// (`Arc<dyn Policy>`) this is an `Arc::clone`; for a concrete `P` it
    /// invokes `P: Clone`.
    #[must_use]
    pub fn policy(&self) -> P {
        self.inner.config.policy.clone()
    }

    /// Returns a borrow of the policy. Cheap regardless of `P`.
    #[must_use]
    pub fn policy_ref(&self) -> &P {
        &self.inner.config.policy
    }

    #[must_use]
    pub fn lifecycle_listener(&self) -> Option<&Arc<dyn LifecycleListener<K, V> + Send + Sync>> {
        self.inner.config.lifecycle_listener.as_ref()
    }

    /// Returns the value for `key` if present and not expired. The key may
    /// be borrowed in any form `K: Borrow<Q>` accepts, so e.g. a
    /// `Cache<String, _>` accepts `&str` directly: `cache.get("foo")`.
    #[must_use]
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get_served(key)
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    /// Removes and returns the value for `key`. See [`Self::get`] for the
    /// `Borrow<Q>` lookup contract.
    #[must_use]
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.remove(key)
    }

    /// Returns whether the cache currently has a served value for `key`. Does
    /// not promote the entry in LRU order or evict expired entries.
    #[must_use]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.contains_key(key)
    }

    /// Number of `Ready` entries currently held. In-flight miss leaders are
    /// not counted.
    ///
    /// With more than one shard the per-shard locks are taken sequentially,
    /// so the returned count is **not** snapshot-consistent across shards
    /// against concurrent writers; treat it as a coarse gauge rather than a
    /// transactional read.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when no `Ready` entry is currently held.
    ///
    /// Short-circuits on the first non-empty shard, so it is cheaper than
    /// `self.len() == 0` when any shard has data; it shares the same
    /// non-snapshot caveat as [`Self::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Sum of [`Policy::weight`] over all `Ready` entries.
    ///
    /// With `shards == 1` the result is bounded by [`Self::max_weight`].
    /// With more shards each one rounds its local cap up, so the sum may
    /// exceed `max_weight` by up to `shards - 1` units of slack. Like
    /// [`Self::len`], the sum is not snapshot-consistent across shards.
    #[must_use]
    pub fn weight(&self) -> u64 {
        self.inner.weight()
    }

    /// Removes every `Ready` entry, fires `Removed` lifecycle events for each
    /// drained value, and cancels every in-flight miss lane (so any pending
    /// `try_get_with` follower retries instead of waiting forever on a
    /// flight that nobody will publish).
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Runs the miss initializer for `key` when absent or not yet served, coalescing concurrent waiters.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the initializer future resolves to `Err`; no value is installed for `key`.
    ///
    /// # Panics
    ///
    /// Panics only on internal bugs: if the miss single-flight `watch` sender is dropped before
    /// waiters are notified, if a stored miss error does not match the requested `E` when
    /// unwrapping waiters, or if the cache mutex is poisoned.
    pub async fn try_get_with<F, Fut, E>(&self, key: K, init: F) -> Result<V, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<V, E>>,
        E: Clone + Send + Sync + 'static,
    {
        inner::Inner::try_get_with(&self.inner, key, init).await
    }

    /// Miss path that returns `V` by using [`Infallible`] errors internally; see [`Self::try_get_with`].
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::try_get_with`].
    pub async fn get_with<F, Fut>(&self, key: K, init: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = V>,
    {
        match self
            .try_get_with(key, || {
                let f = init();
                async move { Ok::<_, Infallible>(f.await) }
            })
            .await
        {
            Ok(v) => v,
            Err(e) => match e {},
        }
    }

    /// Synchronous fallible miss-coalescing variant of
    /// [`Self::try_get_with`]: blocks the current thread on a Condvar
    /// until the leader publishes an outcome. Does **not** drive any
    /// async runtime, so it is safe to use outside (or inside) a Tokio
    /// context. The shard lock is released before `init` runs, so other
    /// keys remain responsive while the initializer is executing. `Err`
    /// outcomes are forwarded to followers but never installed in the
    /// slab — the next caller probes fresh.
    ///
    /// # Errors
    ///
    /// Returns the `Err` produced by `init`. Errors are not cached;
    /// every caller after a failed leader runs `init` again (or joins
    /// a fresh in-flight follower).
    ///
    /// # Panics
    ///
    /// Panics under the same internal-bug conditions as
    /// [`Self::try_get_with`].
    pub fn try_get_with_blocking<F, E>(&self, key: K, init: F) -> Result<V, E>
    where
        F: FnOnce(&K) -> Result<V, E>,
        E: Clone + Send + Sync + 'static,
    {
        inner::Inner::try_get_with_blocking(&self.inner, key, init)
    }

    /// Synchronous infallible miss-coalescing variant of
    /// [`Self::get_with`]: blocks the current thread on a Condvar until
    /// the leader publishes a value. Wraps [`Self::try_get_with_blocking`]
    /// with [`Infallible`] errors.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::try_get_with_blocking`].
    pub fn get_with_blocking<F>(&self, key: K, init: F) -> V
    where
        F: FnOnce(&K) -> V,
    {
        match self.try_get_with_blocking::<_, Infallible>(key, |k| Ok(init(k))) {
            Ok(v) => v,
            Err(e) => match e {},
        }
    }

    /// Schedules a refresh job on `executor` for `key`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the initializer resolves to `Err` on its first poll before the job is
    /// handed to [`Executor::execute`]. Errors that appear only after the job is scheduled are not
    /// returned here (the refresh future completes without updating the map).
    ///
    /// # Panics
    ///
    /// Panics if the cache mutex is poisoned.
    pub fn try_revalidate<E, Ex, F, Fut>(
        &self,
        key: K,
        executor: &Ex,
        init: F,
    ) -> Result<RevalidateScheduled, E>
    where
        E: Clone + Send + Sync + 'static,
        Ex: Executor<RefreshWork<K, V, E, P, S>> + ?Sized,
        F: FnOnce(&K) -> Fut,
        Fut: Future<Output = Result<V, E>> + Send + 'static,
        K: Unpin,
    {
        match self.inner.try_mark_refresh_inflight(&key) {
            inner::RefreshGate::Submit => {}
            inner::RefreshGate::SkipInFlight => {
                return Ok(RevalidateScheduled::SkippedInFlight);
            }
            inner::RefreshGate::AbsentKey => {
                return Ok(RevalidateScheduled::AbsentKey);
            }
        }

        let mut guard = RefreshGuard {
            inner: &self.inner,
            key: &key,
            armed: true,
        };

        let mut fut: Pin<Box<dyn Future<Output = Result<V, E>> + Send>> = Box::pin(init(&key));
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(Err(e)) => {
                return Err(e);
            }
            Poll::Ready(Ok(v)) => {
                let v2 = v.clone();
                fut = Box::pin(async move { Ok::<V, E>(v2) });
            }
            Poll::Pending => {}
        }

        guard.armed = false;
        drop(guard);
        executor.execute(RefreshWork {
            inner: Arc::clone(&self.inner),
            key,
            fut,
            cleanup: |inner, key| inner.refresh_inflight_remove(key),
            cleaned: false,
        });
        Ok(RevalidateScheduled::Submitted)
    }

    /// Infallible wrapper around [`Self::try_revalidate`].
    ///
    /// `revalidate` reuses the same job pipeline as [`Self::try_revalidate`]
    /// but parameterises the spawned [`RefreshWork`] over [`Infallible`]
    /// (the `init` future cannot fail). An executor that implements
    /// `Executor<RefreshWork<K, V, E, P>>` for every `E` — for example a
    /// blanket `impl<Fut: Future + Send + 'static> Executor<Fut> for
    /// MyExecutor` — therefore covers both `try_revalidate` *and*
    /// `revalidate` without any per-method specialisation. If you write an
    /// executor with a concrete `Fut` type instead, you need both
    /// `Executor<RefreshWork<K, V, MyError, P>>` and
    /// `Executor<RefreshWork<K, V, Infallible, P>>` impls to use both
    /// methods.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::try_revalidate`].
    pub fn revalidate<Ex, F, Fut>(&self, key: K, executor: &Ex, init: F) -> RevalidateScheduled
    where
        Ex: Executor<RefreshWork<K, V, Infallible, P, S>> + ?Sized,
        F: FnOnce(&K) -> Fut,
        Fut: Future<Output = V> + Send + 'static,
        K: Unpin,
    {
        match self.try_revalidate(key, executor, |k| OkIntoResult::new(init(k))) {
            Ok(scheduled) => scheduled,
            Err(e) => match e {},
        }
    }
}

#[cfg(feature = "typesize")]
impl<K, V, P, S> typesize::TypeSize for Cache<K, V, P, S>
where
    K: typesize::TypeSize + Eq + Hash + 'static,
    V: typesize::TypeSize + 'static,
    P: typesize::TypeSize + 'static,
    S: typesize::TypeSize + 'static,
{
    /// Walks every shard under a read lock and every slab entry — for
    /// diagnostics only.
    fn extra_size(&self) -> usize {
        inner::arc_inner_size(&self.inner)
    }
}
