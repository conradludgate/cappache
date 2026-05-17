//! Builder-time configuration: capacity budget and optional listeners.

use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::sync::Arc;

use hashbrown::DefaultHashBuilder;

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
///
/// `S` is the [`BuildHasher`]. The default ([`DefaultHashBuilder`] from
/// `hashbrown`, currently `foldhash::fast::RandomState`) is fast and
/// DoS-resistant. Pass `ahash::RandomState`, `xxhash_rust::xxh3::Xxh3Builder`,
/// or any custom `BuildHasher` via [`CacheBuilder::hasher`] to swap.
#[derive(Clone)]
pub struct CacheConfig<K, V, P = DynPolicy<K, V>, S = DefaultHashBuilder>
where
    K: 'static,
    V: 'static,
{
    pub max_weight: Weight,
    pub policy: P,
    pub hasher: S,
    pub(crate) shards: usize,
    pub(crate) lifecycle_listener: Option<Arc<dyn LifecycleListener<K, V> + Send + Sync>>,
    pub(crate) _marker: PhantomData<(K, V)>,
}

/// Constructs a [`crate::Cache`].
///
/// `K` and `V` bounds match [`crate::Cache`]; `P` and `S` follow
/// [`CacheConfig`]. [`Self::policy`] and [`Self::hasher`] are generic so
/// a `.policy(MyPolicy)` or `.hasher(MyHasher)` chain switches the
/// builder's `P` / `S` to the concrete type, allowing the downstream
/// cache to inline both the policy hooks and the `BuildHasher` calls.
#[derive(Clone)]
pub struct CacheBuilder<K, V, P = DynPolicy<K, V>, S = DefaultHashBuilder>
where
    K: 'static,
    V: 'static,
{
    max_weight: Weight,
    policy: P,
    hasher: S,
    shards: usize,
    lifecycle_listener: Option<Arc<dyn LifecycleListener<K, V> + Send + Sync>>,
    _marker: PhantomData<(K, V)>,
}

impl<K, V> Default for CacheBuilder<K, V, DynPolicy<K, V>, DefaultHashBuilder>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            max_weight: Weight::ZERO,
            policy: Arc::new(DefaultPolicy),
            hasher: DefaultHashBuilder::default(),
            shards: default_shard_count(),
            lifecycle_listener: None,
            _marker: PhantomData,
        }
    }
}

/// next-pow2(available parallelism), min 2. Functionally any value
/// `>= 2` works; only contention behaviour changes. Min 2 matches
/// `clashcore::ClashCollection`'s precondition.
fn default_shard_count() -> usize {
    std::thread::available_parallelism()
        .map_or(2, usize::from)
        .next_power_of_two()
        .max(2)
}

impl<K, V> CacheBuilder<K, V, DynPolicy<K, V>, DefaultHashBuilder>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Returns a builder seeded with [`DefaultPolicy`] erased through
    /// [`DynPolicy`] and [`DefaultHashBuilder`] (foldhash). Switch to a
    /// monomorphic policy by chaining [`Self::policy`] with a concrete
    /// value, or to a different hasher with [`Self::hasher`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V, P, S> CacheBuilder<K, V, P, S>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    P: Policy<K, V> + Send + Sync + 'static,
    S: BuildHasher + Clone + Send + Sync + 'static,
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
    pub fn policy<P2>(self, policy: P2) -> CacheBuilder<K, V, P2, S>
    where
        P2: Policy<K, V> + Send + Sync + 'static,
    {
        CacheBuilder {
            max_weight: self.max_weight,
            policy,
            hasher: self.hasher,
            shards: self.shards,
            lifecycle_listener: self.lifecycle_listener,
            _marker: PhantomData,
        }
    }

    /// Replaces the [`BuildHasher`] used for shard selection and entry
    /// lookup. Switching the hasher changes the builder's `S` so the
    /// downstream cache calls the new `BuildHasher` directly without a
    /// vtable. Common choices: `ahash::RandomState`,
    /// `xxhash_rust::xxh3::Xxh3Builder`, `foldhash::fast::RandomState`
    /// (the default).
    #[must_use]
    pub fn hasher<S2>(self, hasher: S2) -> CacheBuilder<K, V, P, S2>
    where
        S2: BuildHasher + Clone + Send + Sync + 'static,
    {
        CacheBuilder {
            max_weight: self.max_weight,
            policy: self.policy,
            hasher,
            shards: self.shards,
            lifecycle_listener: self.lifecycle_listener,
            _marker: PhantomData,
        }
    }

    /// Sets the requested shard count. Clamped to `>= 2` and rounded up
    /// to the next power of two — both invariants are required by
    /// [`clashcore::ClashCollection`], which backs the per-shard storage.
    /// Each shard enforces a rounded-up share of [`Self::max_weight`],
    /// so the observed total weight may overshoot by up to
    /// `shards - 1` units of rounding slack.
    #[must_use]
    pub fn shards(mut self, shards: usize) -> Self {
        self.shards = shards.max(2).next_power_of_two();
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
    pub fn build(self) -> CacheConfig<K, V, P, S> {
        assert!(
            self.max_weight.get() > 0,
            "CacheBuilder::build: max_weight must be > 0; call .max_weight(Weight::new(...)) before .build()"
        );
        CacheConfig {
            max_weight: self.max_weight,
            policy: self.policy,
            hasher: self.hasher,
            shards: self.shards,
            lifecycle_listener: self.lifecycle_listener,
            _marker: PhantomData,
        }
    }
}
