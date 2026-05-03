#![deny(unsafe_code)]
#![warn(clippy::pedantic)]

//! Clache (Conrad Ludgate cache): a safe Rust concurrent cache API.
//!
//! This crate is **`#![deny(unsafe_code)]`**. The in-memory [`Cache`](crate::Cache) implements
//! synchronous `get` / `insert` / `remove`, async miss helpers, and executor-driven refresh;
//! [`Cache::get_with_blocking`](crate::Cache::get_with_blocking) uses **[pollster](https://github.com/zesterer/pollster)**
//! **0.4.0** logic inlined in [`blocking`](crate::blocking) (not part of the public API; see `third-party/pollster/`).
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
//!   async [`Cache::try_get_with`] / [`Cache::get_with`], with [`Cache::get_with_blocking`] for callers without an async runtime.
//! - Caller-driven background refresh: [`Cache::try_revalidate`], [`Cache::revalidate`] with an [`Executor`]
//!   from the [`rt`] module.
//! - Fallible miss path uses **`E: Clone`** on [`Cache::try_get_with`] / [`Cache::try_revalidate`] instead of `Arc<E>`;
//!   [`Cache::get_with`] / [`Cache::revalidate`] use [`Infallible`](std::convert::Infallible) as the error type ([`OkIntoResult`] bridges `V` → `Result<V, Infallible>` for executor bounds on `revalidate`).
//!
//! ## Design notes
//!
//! Non-normative implementation intent (O(1) targets, lazy maintenance, deferred drops) lives in
//! [`DESIGN.md`](../docs/DESIGN.md) in the `docs/` folder.

pub(crate) mod blocking;
pub mod cache;
pub mod config;
pub mod lifecycle;
pub mod policy;
pub mod rt;

pub use cache::{Cache, OkIntoResult};
pub use config::{CacheBuilder, CacheConfig};
pub use lifecycle::{LifecycleListener, LifecycleReason};
pub use policy::{DefaultPolicy, IdleTimeout, Policy, TIME_UNBOUNDED, TtlFromWrite, Weight};
pub use rt::{Executor, RevalidateScheduled};

#[cfg(test)]
mod compile_tests {
    use std::sync::Arc;

    use super::*;

    fn assert_sync<T: Sync>() {}

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
}
