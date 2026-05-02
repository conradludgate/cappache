//! In-memory concurrent cache: synchronous `get` / `insert` / `remove`, async miss path, and
//! executor-driven refresh.

mod inner;

use std::convert::Infallible;
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use pin_project_lite::pin_project;

use crate::config::CacheConfig;
use crate::lifecycle::LifecycleListener;
use crate::policy::Policy;
use crate::rt::{Executor, RevalidateScheduled};
use crate::{Weight, blocking};

pin_project! {
    /// Wraps a fallible miss future as `Result<_, Infallible>` for [`Executor`] bounds on refresh.
    pub struct OkIntoResult<Fut> {
        #[pin]
        fut: Fut,
    }
}

impl<Fut> OkIntoResult<Fut> {
    #[must_use]
    pub fn new(fut: Fut) -> Self {
        Self { fut }
    }
}

impl<Fut: Future> Future for OkIntoResult<Fut> {
    type Output = Result<Fut::Output, Infallible>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.project().fut.poll(cx).map(Ok)
    }
}

/// Job handed to [`Executor::execute`](crate::rt::Executor::execute) by [`Cache::try_revalidate`].
pub struct RefreshWork<K: 'static, V: 'static, E: 'static> {
    pub(crate) inner: Arc<inner::Inner<K, V>>,
    pub(crate) key: K,
    pub(crate) fut: Pin<Box<dyn Future<Output = Result<V, E>> + Send>>,
}

impl<K, V, E> Future for RefreshWork<K, V, E>
where
    K: Eq + Hash + Clone + Send + Sync + Unpin + 'static,
    V: Clone + Send + Sync + 'static,
    E: Send + 'static,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.fut.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(v)) => {
                this.inner.apply_refresh_ok(&this.key, v);
                this.inner.refresh_inflight_remove(&this.key);
                Poll::Ready(())
            }
            Poll::Ready(Err(_)) => {
                this.inner.refresh_inflight_remove(&this.key);
                Poll::Ready(())
            }
        }
    }
}

/// Concurrent in-memory cache.
pub struct Cache<K: 'static, V: 'static> {
    inner: Arc<inner::Inner<K, V>>,
}

impl<K: 'static, V: 'static> Clone for Cache<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    #[must_use]
    pub fn new(config: CacheConfig<K, V>) -> Self {
        Self {
            inner: Arc::new(inner::Inner::new(config)),
        }
    }

    #[must_use]
    pub fn max_weight(&self) -> Weight {
        self.inner.config.max_weight
    }

    #[must_use]
    pub fn policy(&self) -> Arc<dyn Policy<K, V> + Send + Sync> {
        Arc::clone(&self.inner.config.policy)
    }

    #[must_use]
    pub fn policy_ref(&self) -> &(dyn Policy<K, V> + Send + Sync) {
        self.inner.config.policy.as_ref()
    }

    #[must_use]
    pub fn lifecycle_listener(&self) -> Option<&Arc<dyn LifecycleListener<K, V> + Send + Sync>> {
        self.inner.config.lifecycle_listener.as_ref()
    }

    #[must_use]
    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.get_served(key)
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    #[must_use]
    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.remove(key)
    }

    pub async fn try_get_with<F, Fut, E>(&self, key: K, init: F) -> Result<V, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<V, E>>,
        E: Clone + Send + Sync + 'static,
    {
        inner::Inner::try_get_with(&self.inner, key, init).await
    }

    pub async fn get_with<F, Fut>(&self, key: K, init: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = V>,
    {
        self.try_get_with(key, || {
            let f = init();
            async move { Ok::<_, Infallible>(f.await) }
        })
        .await
        .unwrap_or_else(|e| match e {})
    }

    pub fn get_with_blocking<F, Fut>(&self, key: K, init: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = V>,
    {
        let inner = Arc::clone(&self.inner);
        blocking::block_on(async move {
            inner::Inner::try_get_with(&inner, key, || {
                let f = init();
                async move { Ok::<_, Infallible>(f.await) }
            })
            .await
            .unwrap_or_else(|e| match e {})
        })
    }

    pub fn try_revalidate<E, Ex, F, Fut>(
        &self,
        key: K,
        executor: &Ex,
        init: F,
    ) -> Result<RevalidateScheduled, E>
    where
        E: Clone + Send + Sync + 'static,
        Ex: Executor<RefreshWork<K, V, E>> + ?Sized,
        F: FnOnce(&K) -> Fut,
        Fut: Future<Output = Result<V, E>> + Send + 'static,
        K: Unpin,
    {
        {
            let mut st = self.inner.state.lock().expect("cache lock");
            if st.refresh_inflight.contains(&key) {
                return Ok(RevalidateScheduled::SkippedInFlight);
            }
            st.refresh_inflight.insert(key.clone());
        }

        let mut fut: Pin<Box<dyn Future<Output = Result<V, E>> + Send>> = Box::pin(init(&key));
        let waker = Waker::noop();
        let mut cx = Context::from_waker(&waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(Err(e)) => {
                self.inner.refresh_inflight_remove(&key);
                return Err(e);
            }
            Poll::Ready(Ok(v)) => {
                let v2 = v.clone();
                fut = Box::pin(async move { Ok::<V, E>(v2) });
            }
            Poll::Pending => {}
        }

        executor.execute(RefreshWork {
            inner: Arc::clone(&self.inner),
            key,
            fut,
        });
        Ok(RevalidateScheduled::Submitted)
    }

    pub fn revalidate<Ex, F, Fut>(&self, key: K, executor: &Ex, init: F) -> RevalidateScheduled
    where
        Ex: Executor<RefreshWork<K, V, Infallible>> + ?Sized,
        F: FnOnce(&K) -> Fut,
        Fut: Future<Output = V> + Send + 'static,
        K: Unpin,
    {
        match self.try_revalidate(key, executor, |k| OkIntoResult::new(init(k))) {
            Ok(s) => s,
            Err(e) => match e {},
        }
    }
}
