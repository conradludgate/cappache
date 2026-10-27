#![deny(unsafe_code)]
#![warn(clippy::pedantic)]

//! Clache (Conrad Ludgate cache): a safe Rust concurrent cache API.
//!
//! This crate is **`#![deny(unsafe_code)]`**. The in-memory [`Cache`](crate::Cache) implements
//! synchronous `get` / `insert` / `remove`, async miss helpers, and executor-driven refresh.
//! Sync miss-coalescing lives on [`Cache::get_with_blocking`] /
//! [`Cache::try_get_with_blocking`] (Condvar-backed; no async runtime needed) for callers
//! outside Tokio.
//!
//! ## Overview
//!
//! - [`Cache`] is [`Sync`]: share it across threads behind an `std::sync::Arc` if needed.
//! - Configure capacity with [`Weight`] and per-entry behavior with [`Policy`] via
//!   [`CacheBuilder::policy`] (stored as [`CacheConfig::policy`]).
//! - Time-based eviction is **not** a separate cache feature: it comes from your [`Policy`]
//!   implementation (see [`policy::TtlFromWrite`], [`policy::IdleTimeout`], etc.).
//! - Observe removals and updates through [`lifecycle::LifecycleReason`] and [`lifecycle::LifecycleListener`].
//! - [`Cache::get`], [`Cache::insert`], and [`Cache::remove`] are **synchronous**; miss-path helpers are
//!   async [`Cache::try_get_with`] / [`Cache::get_with`], plus the sync
//!   [`Cache::try_get_with_blocking`] / [`Cache::get_with_blocking`] for callers without an
//!   async runtime.
//! - Caller-driven background refresh: [`Cache::try_revalidate`], [`Cache::revalidate`] with an [`Executor`]
//!   from the [`rt`] module.
//! - Fallible miss path uses **`E: Clone`** on [`Cache::try_get_with`] / [`Cache::try_revalidate`] instead of `Arc<E>`;
//!   [`Cache::get_with`] / [`Cache::revalidate`] use [`Infallible`](std::convert::Infallible) as the error type.
//! - [`Cache`] is `Clone`: cloning yields a handle that shares the underlying state
//!   (cheap `Arc`-style increment), so spawning workers around the same cache is the
//!   intended pattern. Drop a clone independently; the cache is dropped when the last
//!   handle is dropped.

pub mod cache;
pub mod config;
pub mod lifecycle;
pub mod policy;
pub mod rt;

pub use cache::Cache;
pub use config::{CacheBuilder, CacheConfig, DynPolicy};
pub use lifecycle::{LifecycleListener, LifecycleReason};
pub use policy::{DefaultPolicy, IdleTimeout, Policy, TIME_UNBOUNDED, TtlFromWrite, Weight};
pub use rt::{Executor, RevalidateScheduled};

#[cfg(test)]
mod compile_tests {
    use std::sync::Arc;

    use super::*;

    fn assert_sync<T: Sync>() {}

    #[derive(Clone)]
    struct CustomPolicy;

    impl Policy<String, String> for CustomPolicy {
        fn weight(&self, _: &String, _: &String) -> Weight {
            Weight::new(1)
        }

        fn expire_after_read(
            &self,
            _: &String,
            _: &String,
            _: std::time::Instant,
            _: std::time::Instant,
            _: std::time::Duration,
        ) -> std::time::Duration {
            TIME_UNBOUNDED
        }

        fn expire_after_write(
            &self,
            _: &String,
            _: &String,
            _: std::time::Instant,
        ) -> std::time::Duration {
            TIME_UNBOUNDED
        }
    }

    #[test]
    fn cache_is_sync() {
        assert_sync::<Cache<String, String>>();
    }

    #[test]
    fn builder_accepts_custom_policy() {
        let _ = CacheBuilder::<String, String>::new().policy(Arc::new(CustomPolicy));
    }

    /// Vtable-free monomorphic policy path. Round-trips an insert and
    /// a get so the policy's `weight` / `expire_after_write` /
    /// `expire_after_read` hooks are actually invoked, not just type-
    /// checked.
    #[test]
    fn cache_accepts_concrete_policy_without_arc() {
        let cfg = CacheBuilder::<String, String>::new()
            .max_weight(Weight::new(1))
            .policy(CustomPolicy)
            .build();
        let cache: Cache<String, String, CustomPolicy> = Cache::new(cfg);
        cache.insert("k".into(), "v".into());
        assert_eq!(cache.get("k"), Some("v".into()));
    }

    #[test]
    fn ttl_and_idle_wrap_inner_policy() {
        let _ttl: TtlFromWrite<DefaultPolicy> =
            TtlFromWrite::new(std::time::Duration::from_mins(1), DefaultPolicy);
        let _idle: IdleTimeout<DefaultPolicy> =
            IdleTimeout::new(std::time::Duration::from_secs(30), DefaultPolicy);
        let _: Arc<dyn Policy<String, String> + Send + Sync> = Arc::new(TtlFromWrite::new(
            std::time::Duration::from_mins(2),
            DefaultPolicy,
        ));
    }

    #[test]
    fn typical_async_revalidate_init_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'static>>>(
        );
    }

    /// `Cache::new` and `CacheBuilder::policy` accept a non-`Clone` `P`.
    /// Only `Cache::policy()` requires `P: Clone`; `policy_ref()` and the
    /// hot paths do not.
    #[test]
    fn cache_accepts_non_clone_policy() {
        struct NotClonePolicy;

        impl Policy<String, String> for NotClonePolicy {
            fn weight(&self, _: &String, _: &String) -> Weight {
                Weight::new(1)
            }

            fn expire_after_read(
                &self,
                _: &String,
                _: &String,
                _: std::time::Instant,
                _: std::time::Instant,
                _: std::time::Duration,
            ) -> std::time::Duration {
                TIME_UNBOUNDED
            }

            fn expire_after_write(
                &self,
                _: &String,
                _: &String,
                _: std::time::Instant,
            ) -> std::time::Duration {
                TIME_UNBOUNDED
            }
        }

        let cache: Cache<String, String, NotClonePolicy> = Cache::new(
            CacheBuilder::<String, String>::new()
                .max_weight(Weight::new(1))
                .policy(NotClonePolicy)
                .build(),
        );
        cache.insert("k".into(), "v".into());
        assert_eq!(cache.get("k"), Some("v".into()));
        assert_eq!(
            cache.policy_ref().weight(&"k".into(), &"v".into()),
            Weight::new(1)
        );
    }

    /// Confirms `S` threads through builder → config → cache without
    /// erasure.
    #[test]
    fn cache_accepts_custom_hasher() {
        use std::hash::{BuildHasherDefault, Hasher};

        // Pathological for distribution; chosen here for its zero deps
        // (`Clone + Default + Send + Sync`).
        #[derive(Default)]
        struct ZeroHasher(u64);
        impl Hasher for ZeroHasher {
            fn write(&mut self, _: &[u8]) {}
            fn finish(&self) -> u64 {
                self.0
            }
        }
        type Zero = BuildHasherDefault<ZeroHasher>;

        let cache: Cache<String, String, _, Zero> = Cache::new(
            CacheBuilder::<String, String>::new()
                .max_weight(Weight::new(8))
                .hasher(Zero::default())
                .build(),
        );
        cache.insert("k".into(), "v".into());
        assert_eq!(cache.get("k"), Some("v".into()));
    }
}
