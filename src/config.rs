//! Builder-time configuration: capacity budget and optional listeners.

use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::lifecycle::LifecycleListener;
use crate::policy::{DefaultPolicy, Policy, Weight};

/// Immutable options fixed at cache construction.
///
/// [`CacheBuilder`] only produces configs for `K` / `V` that match [`Cache::new`](crate::Cache::new).
/// If you construct this type manually, use the same `K` and `V` bounds as [`Cache`](crate::Cache) and
/// [`Cache::new`](crate::Cache::new).
#[derive(Clone)]
pub struct CacheConfig<K, V>
where
    K: 'static,
    V: 'static,
{
    pub max_weight: Weight,
    pub policy: Arc<dyn Policy<K, V> + Send + Sync>,
    pub(crate) lifecycle_listener: Option<Arc<dyn LifecycleListener<K, V> + Send + Sync>>,
    pub(crate) _marker: PhantomData<(K, V)>,
}

/// Constructs a [`crate::Cache`].
///
/// `K` and `V` bounds match [`crate::Cache`]: any config from [`Self::build`] is accepted by
/// [`Cache::new`](crate::Cache::new).
#[derive(Clone)]
pub struct CacheBuilder<K, V>
where
    K: 'static,
    V: 'static,
{
    max_weight: Weight,
    policy: Arc<dyn Policy<K, V> + Send + Sync>,
    lifecycle_listener: Option<Arc<dyn LifecycleListener<K, V> + Send + Sync>>,
    _marker: PhantomData<(K, V)>,
}

impl<K, V> Default for CacheBuilder<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            max_weight: Weight::ZERO,
            policy: Arc::new(DefaultPolicy),
            lifecycle_listener: None,
            _marker: PhantomData,
        }
    }
}

impl<K, V> CacheBuilder<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn max_weight(mut self, weight: Weight) -> Self {
        self.max_weight = weight;
        self
    }

    #[must_use]
    pub fn policy(mut self, policy: Arc<dyn Policy<K, V> + Send + Sync>) -> Self {
        self.policy = policy;
        self
    }

    #[must_use]
    pub fn lifecycle_listener(
        mut self,
        listener: Arc<dyn LifecycleListener<K, V> + Send + Sync>,
    ) -> Self {
        self.lifecycle_listener = Some(listener);
        self
    }

    #[must_use]
    pub fn build(self) -> CacheConfig<K, V> {
        CacheConfig {
            max_weight: self.max_weight,
            policy: self.policy,
            lifecycle_listener: self.lifecycle_listener,
            _marker: PhantomData,
        }
    }
}
