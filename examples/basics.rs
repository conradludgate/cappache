//! Synchronous use of `Cache`: builder, insert / get / remove, and a
//! `LifecycleListener` that prints every removal reason as it fires.
//!
//! Run with: `cargo run --example basics`

use std::sync::Arc;

use clache::{Cache, CacheBuilder, LifecycleListener, LifecycleReason, Weight};

struct StdoutListener;

impl LifecycleListener<String, String> for StdoutListener {
    fn on_event(&self, reason: LifecycleReason, key: String, value: Option<String>) {
        println!("  [{reason:?}] {key} -> {value:?}");
    }
}

fn main() {
    // The builder is fluent and produces a `CacheConfig`; `Cache::new`
    // takes ownership of the config. `max_weight` is required: the
    // builder panics on `.build()` if it is left at the default zero.
    // `shards(1)` enforces the global weight budget exactly. The
    // builder defaults to many shards so per-shard contention scales
    // with cores; in production-sized caches that overshoot is
    // negligible, but at `max_weight = 3` it would scatter the four
    // keys across shards with no eviction. See the `shards` builder
    // doc for the multi-shard rounding rule.
    let cache: Cache<String, String> = Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(3))
            .shards(1)
            .lifecycle_listener(Arc::new(StdoutListener))
            .build(),
    );

    cache.insert("a".into(), "alpha".into());
    cache.insert("b".into(), "bravo".into());
    cache.insert("c".into(), "charlie".into());
    println!(
        "after 3 inserts: len = {}, weight = {}",
        cache.len(),
        cache.weight()
    );

    // Inserting a fourth key with `max_weight = 3` forces an
    // `EvictedSize` event for one existing entry. The S3-FIFO policy
    // picks a victim from the Small queue first; the default policy
    // gives every entry weight 1, so exactly one victim leaves.
    println!("\ninserting `d` (forces eviction):");
    cache.insert("d".into(), "delta".into());

    // Replacing an existing key fires `Updated` with the prior value.
    println!("\nupdating `d`:");
    cache.insert("d".into(), "DELTA".into());
    assert_eq!(cache.get("d"), Some("DELTA".to_string()));

    // Explicit removal fires `Removed`.
    println!("\nremoving `d`:");
    let _ = cache.remove("d");
}
