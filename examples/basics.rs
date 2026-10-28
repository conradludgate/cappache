//! Synchronous use of `Cache`: builder, insert / get / remove, and a
//! `LifecycleListener` that prints every removal reason as it fires.
//!
//! Run with: `cargo run --example basics`

use std::sync::Arc;

use cappache::{Cache, CacheBuilder, LifecycleListener, LifecycleReason, Weight};

struct StdoutListener;

impl LifecycleListener<String, String> for StdoutListener {
    fn on_event(&self, reason: LifecycleReason, key: String, value: Option<String>) {
        println!("  [{reason:?}] {key} -> {value:?}");
    }
}

fn main() {
    // The builder is fluent and produces a `CacheConfig`; `Cache::new`
    // takes ownership of the config. `max_weight` is required — the
    // builder panics on `.build()` if it is left at the default zero.
    // The cache shards internally for cross-core scaling: each shard
    // enforces a rounded-up share of `max_weight`, so the observed
    // total weight may overshoot by up to `shards - 1` units of
    // rounding slack (see the `shards` builder doc).
    let cache: Cache<String, String> = Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(4))
            .lifecycle_listener(Arc::new(StdoutListener))
            .build(),
    );

    for k in ["a", "b", "c"] {
        cache.insert(k.into(), format!("value-{k}"));
    }
    println!(
        "after 3 inserts: len = {}, weight = {}",
        cache.len(),
        cache.weight()
    );

    // Inserting more keys than `max_weight` allows forces `EvictedSize`
    // events for at least some entries. The S3-FIFO policy picks
    // victims from the Small queue first; with `max_weight = 4` the
    // run below evicts a few keys (exact identities depend on which
    // shard each one hashes into).
    println!("\ninserting more keys (forces evictions):");
    for k in ["d", "e", "f", "g", "h"] {
        cache.insert(k.into(), format!("value-{k}"));
    }

    // Replacing an existing key fires `Updated` with the prior value.
    println!("\nupdating `h`:");
    cache.insert("h".into(), "value-H".into());
    assert_eq!(cache.get("h"), Some("value-H".to_string()));

    // Explicit removal fires `Removed`.
    println!("\nremoving `h`:");
    let _ = cache.remove("h");
}
