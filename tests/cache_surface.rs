//! Integration tests for cache **construction and introspection**.
//! Cache-driven lifecycle behaviour is covered in `tests/cache_ops.rs` (`lifecycle` module).

use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use clache::{
    Cache, CacheBuilder, DefaultPolicy, IdleTimeout, LifecycleListener, LifecycleReason, Policy,
    TIME_UNBOUNDED, TtlFromWrite, Weight,
};

struct SurfacePolicy;

impl Policy<String, String> for SurfacePolicy {
    fn weight(&self, _: &String, _: &String) -> Weight {
        Weight::new(7)
    }

    fn expire_after_read(
        &self,
        _: &String,
        _: &String,
        _: Instant,
        _: Instant,
        _: std::time::Duration,
    ) -> std::time::Duration {
        TIME_UNBOUNDED
    }

    fn expire_after_write(&self, _: &String, _: &String, _: Instant) -> std::time::Duration {
        TIME_UNBOUNDED
    }
}

struct NoopListener;

impl LifecycleListener<String, String> for NoopListener {
    fn on_event(&self, _reason: LifecycleReason, _key: String, _value: Option<String>) {}
}

struct CountListener(Arc<AtomicUsize>);

impl LifecycleListener<String, String> for CountListener {
    fn on_event(&self, _: LifecycleReason, _: String, _: Option<String>) {
        self.0.fetch_add(1, AtomicOrdering::Relaxed);
    }
}

fn string_cache() -> Cache<String, String> {
    Cache::new(CacheBuilder::new().build())
}

#[test]
fn default_max_weight_is_zero() {
    let cache = string_cache();
    assert_eq!(cache.max_weight(), Weight::ZERO);
}

#[test]
fn weight_orders_by_inner_u64() {
    assert!(Weight::ZERO < Weight::new(1));
    assert!(Weight::new(1) < Weight::new(2));
    assert_eq!(Weight::new(7).cmp(&Weight::new(7)), Ordering::Equal);
}

#[test]
fn max_weight_chain_preserved() {
    let w = Weight::new(42);
    let cache = Cache::new(CacheBuilder::<String, String>::new().max_weight(w).build());
    assert_eq!(cache.max_weight().get(), 42);
}

#[test]
fn policy_ref_matches_default_policy() {
    let cache = string_cache();
    let k = "k".to_string();
    let v = "v".to_string();
    let now = Instant::now();
    assert_eq!(cache.policy_ref().weight(&k, &v), Weight::new(1));
    assert_eq!(
        cache
            .policy_ref()
            .expire_after_read(&k, &v, now, now, std::time::Duration::ZERO),
        TIME_UNBOUNDED
    );
    assert_eq!(
        cache.policy_ref().expire_after_write(&k, &v, now),
        TIME_UNBOUNDED
    );
}

#[test]
fn custom_policy_via_builder() {
    let cache = Cache::new(
        CacheBuilder::<String, String>::new()
            .policy(Arc::new(SurfacePolicy))
            .build(),
    );
    let k = "k".to_string();
    let v = "v".to_string();
    assert_eq!(cache.policy_ref().weight(&k, &v), Weight::new(7));
}

#[test]
fn ttl_from_write_caps_write_expiry() {
    let ttl = std::time::Duration::from_secs(30);
    let cache = Cache::new(
        CacheBuilder::<String, String>::new()
            .policy(Arc::new(TtlFromWrite::new(ttl, DefaultPolicy)))
            .build(),
    );
    let k = "k".to_string();
    let v = "v".to_string();
    let now = Instant::now();
    assert_eq!(cache.policy_ref().weight(&k, &v), Weight::new(1));
    assert_eq!(cache.policy_ref().expire_after_write(&k, &v, now), ttl);
    let read_cap =
        cache
            .policy_ref()
            .expire_after_read(&k, &v, now, now, std::time::Duration::from_mins(1));
    assert_eq!(read_cap, std::time::Duration::from_mins(1));
}

#[test]
fn ttl_from_write_zero_clamps_write_expiry_to_zero() {
    let p = TtlFromWrite::new(std::time::Duration::ZERO, DefaultPolicy);
    let k = "k".to_string();
    let v = "v".to_string();
    let now = Instant::now();
    assert_eq!(p.expire_after_write(&k, &v, now), std::time::Duration::ZERO);
}

#[test]
fn idle_timeout_caps_hooks() {
    let idle = std::time::Duration::from_secs(15);
    let cache = Cache::new(
        CacheBuilder::<String, String>::new()
            .policy(Arc::new(IdleTimeout::new(idle, DefaultPolicy)))
            .build(),
    );
    let k = "k".to_string();
    let v = "v".to_string();
    let now = Instant::now();
    assert_eq!(cache.policy_ref().expire_after_write(&k, &v, now), idle);
    let read_cap = cache.policy_ref().expire_after_read(
        &k,
        &v,
        now,
        now,
        std::time::Duration::from_secs(9999),
    );
    assert_eq!(read_cap, idle);
}

#[test]
fn idle_timeout_weight_matches_inner_default_policy() {
    let idle = std::time::Duration::from_secs(30);
    let idle_pol = IdleTimeout::new(idle, DefaultPolicy);
    let k = "k".to_string();
    let v = "hello".to_string();
    assert_eq!(
        Policy::weight(&idle_pol, &k, &v),
        Policy::weight(&DefaultPolicy, &k, &v)
    );
}

#[test]
fn policy_clone_does_not_change_policy_ref_semantics() {
    let cache = string_cache();
    let k = "k".to_string();
    let v = "v".to_string();
    let w0 = cache.policy_ref().weight(&k, &v);
    {
        let _cloned_arc = cache.policy();
    }
    assert_eq!(cache.policy_ref().weight(&k, &v), w0);
}

#[test]
fn policy_strong_count_reflects_cache_and_clones() {
    let cache = string_cache();
    let p = cache.policy();
    assert_eq!(Arc::strong_count(&p), 2, "cache config + returned clone");
    drop(p);
    let p2 = cache.policy();
    assert_eq!(Arc::strong_count(&p2), 2);
}

#[test]
fn lifecycle_listener_none_by_default() {
    let cache = string_cache();
    assert!(cache.lifecycle_listener().is_none());
}

#[test]
fn lifecycle_listener_some_matches_installed() {
    let listener: Arc<dyn LifecycleListener<String, String> + Send + Sync> = Arc::new(NoopListener);
    let ptr = Arc::as_ptr(&listener);
    let cache = Cache::new(
        CacheBuilder::<String, String>::new()
            .lifecycle_listener(listener.clone())
            .build(),
    );
    let opt = cache.lifecycle_listener().expect("listener");
    assert!(Arc::ptr_eq(opt, &listener));
    assert_eq!(Arc::as_ptr(opt), ptr);
}

#[test]
fn lifecycle_reason_variants_have_distinct_debug() {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for r in [
        LifecycleReason::EvictedSize,
        LifecycleReason::TimeExpired,
        LifecycleReason::Removed,
        LifecycleReason::Updated,
    ] {
        assert!(seen.insert(format!("{r:?}")), "duplicate Debug: {r:?}");
    }
}

#[test]
fn lifecycle_listener_receives_each_reason() {
    let hits = Arc::new(AtomicUsize::new(0));
    let listener: Arc<dyn LifecycleListener<String, String> + Send + Sync> =
        Arc::new(CountListener(Arc::clone(&hits)));
    for reason in [
        LifecycleReason::EvictedSize,
        LifecycleReason::TimeExpired,
        LifecycleReason::Removed,
        LifecycleReason::Updated,
    ] {
        listener.on_event(reason, "k".into(), None);
    }
    assert_eq!(hits.load(AtomicOrdering::Relaxed), 4);
}

#[test]
fn cloned_builder_same_max_weight() {
    let b = CacheBuilder::<String, String>::new().max_weight(Weight::new(99));
    let b2 = b.clone();
    assert_eq!(Cache::new(b.build()).max_weight().get(), 99);
    assert_eq!(Cache::new(b2.build()).max_weight().get(), 99);
}

#[test]
fn cache_string_string_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Cache<String, String>>();
}

#[test]
fn cache_clone_shares_state() {
    let cache = Cache::<String, String>::new(
        CacheBuilder::new()
            .max_weight(Weight::new(8))
            .policy(Arc::new(DefaultPolicy))
            .build(),
    );
    let handle = cache.clone();
    cache.insert("k".into(), "v".into());
    assert_eq!(handle.get("k"), Some("v".to_string()));
    handle.insert("k".into(), "v2".into());
    assert_eq!(cache.get("k"), Some("v2".to_string()));
}

#[test]
fn len_is_empty_weight_track_inserts_and_removes() {
    let cache = Cache::<String, String>::new(
        CacheBuilder::new()
            .max_weight(Weight::new(100))
            .policy(Arc::new(DefaultPolicy))
            .build(),
    );
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.weight(), 0);

    cache.insert("a".into(), "v".into());
    cache.insert("b".into(), "v".into());
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 2);
    // DefaultPolicy weights every entry as 1.
    assert_eq!(cache.weight(), 2);

    let _ = cache.remove("a");
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.weight(), 1);
}

#[test]
fn clear_drops_entries_and_fires_removed_for_each() {
    use std::sync::Mutex;

    type Events = Arc<Mutex<Vec<(LifecycleReason, String, Option<String>)>>>;
    struct Recorder(Events);
    impl LifecycleListener<String, String> for Recorder {
        fn on_event(&self, r: LifecycleReason, k: String, v: Option<String>) {
            self.0.lock().unwrap().push((r, k, v));
        }
    }

    let events: Events = Arc::new(Mutex::new(Vec::new()));
    let listener: Arc<dyn LifecycleListener<String, String> + Send + Sync> =
        Arc::new(Recorder(Arc::clone(&events)));
    let cache = Cache::<String, String>::new(
        CacheBuilder::new()
            .max_weight(Weight::new(100))
            .policy(Arc::new(DefaultPolicy))
            .lifecycle_listener(listener)
            .build(),
    );
    cache.insert("a".into(), "va".into());
    cache.insert("b".into(), "vb".into());

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.weight(), 0);
    assert!(cache.get("a").is_none());

    let log = events.lock().unwrap().clone();
    let removed: Vec<_> = log
        .into_iter()
        .filter(|(r, _, _)| *r == LifecycleReason::Removed)
        .map(|(_, k, _)| k)
        .collect();
    assert!(removed.contains(&"a".to_string()));
    assert!(removed.contains(&"b".to_string()));
    assert_eq!(removed.len(), 2);
}
