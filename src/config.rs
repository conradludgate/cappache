//! Builder-time configuration: capacity budget and optional listeners.

use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::lifecycle::LifecycleListener;
use crate::policy::{DefaultPolicy, Policy, Weight};

/// Default policy slot type for [`CacheConfig`] / [`CacheBuilder`] / [`crate::Cache`].
///
/// Type-erased so callers who do not care about monomorphisation get the
/// same one-line ergonomics they always had. Pick a concrete `P` if you
/// want vtable-free hot paths.
pub type DynPolicy<K, V> = Arc<dyn Policy<K, V> + Send + Sync>;

/// Immutable options fixed at cache construction.
///
/// `P` is the policy type. By default, [`CacheBuilder::new`] gives you
/// `P = Arc<dyn Policy<K, V> + Send + Sync>` (= [`DynPolicy`]) so the cache
/// stays object-safe; pass any concrete `Policy<K, V>` impl to
/// [`CacheBuilder::policy`] to monomorphise the hot path and skip the
/// vtable.
#[derive(Clone)]
pub struct CacheConfig<K, V, P = DynPolicy<K, V>>
where
    K: 'static,
    V: 'static,
{
    pub max_weight: Weight,
    pub policy: P,
    pub(crate) lifecycle_listener: Option<Arc<dyn LifecycleListener<K, V> + Send + Sync>>,
    pub(crate) _marker: PhantomData<(K, V)>,
}

/// Constructs a [`crate::Cache`].
///
/// `K` and `V` bounds match [`crate::Cache`]; `P` follows
/// [`CacheConfig`]. [`Self::policy`] is generic so a `.policy(MyPolicy)`
/// chain switches the builder's `P` to the concrete type, allowing the
/// downstream cache to inline policy hooks.
#[derive(Clone)]
pub struct CacheBuilder<K, V, P = DynPolicy<K, V>>
where
    K: 'static,
    V: 'static,
{
    max_weight: Weight,
    policy: P,
    lifecycle_listener: Option<Arc<dyn LifecycleListener<K, V> + Send + Sync>>,
    _marker: PhantomData<(K, V)>,
}

impl<K, V> Default for CacheBuilder<K, V, DynPolicy<K, V>>
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

impl<K, V> CacheBuilder<K, V, DynPolicy<K, V>>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Returns a builder seeded with [`DefaultPolicy`] erased through
    /// [`DynPolicy`]. Switch to a monomorphic policy by chaining
    /// [`Self::policy`] with a concrete value.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V, P> CacheBuilder<K, V, P>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    P: Policy<K, V> + Clone + Send + Sync + 'static,
{
    #[must_use]
    pub fn max_weight(mut self, weight: Weight) -> Self {
        self.max_weight = weight;
        self
    }

    /// Replaces the policy with `policy`, switching the builder's `P` to
    /// the concrete (or trait-object) type passed in. Pass a concrete
    /// `MyPolicy` for vtable-free hot paths, or an
    /// `Arc<dyn Policy<K, V> + Send + Sync>` to keep type erasure.
    #[must_use]
    pub fn policy<P2>(self, policy: P2) -> CacheBuilder<K, V, P2>
    where
        P2: Policy<K, V> + Clone + Send + Sync + 'static,
    {
        CacheBuilder {
            max_weight: self.max_weight,
            policy,
            lifecycle_listener: self.lifecycle_listener,
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn lifecycle_listener(
        mut self,
        listener: Arc<dyn LifecycleListener<K, V> + Send + Sync>,
    ) -> Self {
        self.lifecycle_listener = Some(listener);
        self
    }

    /// Finalises the builder.
    ///
    /// # Panics
    ///
    /// Panics if [`Self::max_weight`] is left at the default `Weight::ZERO`.
    /// A zero-weight cache cannot hold any entry (every insert would exceed
    /// the budget and be evicted immediately), so this almost certainly
    /// indicates a forgotten `.max_weight(...)` call. Surface it loudly at
    /// build time rather than silently producing a useless cache.
    #[must_use]
    pub fn build(self) -> CacheConfig<K, V, P> {
        assert!(
            self.max_weight.get() > 0,
            "CacheBuilder::build: max_weight must be > 0; call .max_weight(Weight::new(...)) before .build()"
        );
        CacheConfig {
            max_weight: self.max_weight,
            policy: self.policy,
            lifecycle_listener: self.lifecycle_listener,
            _marker: PhantomData,
        }
    }
}
