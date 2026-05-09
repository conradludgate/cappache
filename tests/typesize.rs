#![cfg(feature = "typesize")]
//! Sanity checks for the `typesize` integration: a populated cache reports
//! a strictly larger deep size than an empty one, and we delegate to the
//! `Inner`'s walk under the read lock.

use std::hash::BuildHasher;

use clache::policy::DefaultPolicy;
use clache::{Cache, CacheBuilder, Weight};
use typesize::TypeSize;

#[derive(Clone)]
struct TestHasher;

impl BuildHasher for TestHasher {
    type Hasher = std::collections::hash_map::DefaultHasher;
    fn build_hasher(&self) -> Self::Hasher {
        std::collections::hash_map::DefaultHasher::new()
    }
}

impl TypeSize for TestHasher {}

fn build_cache(max_weight: u64) -> Cache<String, String, DefaultPolicy, TestHasher> {
    let cfg = CacheBuilder::new()
        .max_weight(Weight::new(max_weight))
        // `shards(2)` is the cache's documented minimum. `TestHasher`
        // is deterministic across runs, so the per-shard distribution
        // (and therefore the deep-size snapshot below) stays stable.
        .shards(2)
        .policy(DefaultPolicy)
        .hasher(TestHasher)
        .build();
    Cache::new(cfg)
}

#[test]
fn populated_cache_reports_larger_size_than_empty() {
    let empty = build_cache(64);
    let empty_size = empty.get_size();

    let populated = build_cache(64);
    for i in 0..16u32 {
        populated.insert(format!("key-{i}"), format!("value-{i}"));
    }
    let populated_size = populated.get_size();

    assert!(
        populated_size > empty_size,
        "expected populated > empty, got {populated_size} vs {empty_size}"
    );

    let raw_payload: usize = (0..16u32)
        .map(|i| format!("key-{i}").capacity() + format!("value-{i}").capacity())
        .sum();
    assert!(
        populated_size >= empty_size + raw_payload,
        "populated should at least cover key+value heap bytes; \
         empty={empty_size} populated={populated_size} payload={raw_payload}"
    );
}

#[test]
fn empty_cache_has_nonzero_extra_size() {
    let cache = build_cache(64);
    assert!(cache.extra_size() > 0, "shards / config carry heap bytes");
}

/// Snapshot test: pinning the deep size of a deterministic cache so any
/// drift in our `TypeSize` accounting (or in `hashbrown` / `slotmap` /
/// internal struct layout) shows up as a one-line failure.
///
/// `TestHasher` always seeds with `DefaultHasher::new()` (fixed initial
/// state), the cache uses `shards(2)` (cache's minimum), `max_weight =
/// u64::MAX` (so `Policy::weight` doesn't gate anything), and inserts
/// are strictly `format!("k-{i:04}") -> format!("v-{i:04}")` for
/// `i in 0..1023` — one below `slotmap`'s doubling boundary so capacity
/// exactly matches `len` per shard once keys distribute. Gated to
/// 64-bit targets because the constant below assumes 8-byte pointers.
#[cfg(target_pointer_width = "64")]
#[test]
fn snapshot_size_is_stable_with_fixed_seed_hasher() {
    let cache = build_cache(u64::MAX);
    for i in 0..1023u32 {
        cache.insert(format!("k-{i:04}"), format!("v-{i:04}"));
    }

    // Captured on x86_64 / aarch64 with the pinned dependency graph
    // (hashbrown 0.17, slotmap 1, typesize 0.1.14). If you legitimately
    // change layout, re-run and update — but pause first to confirm
    // the new number reflects an intentional shift, not a regression.
    const EXPECTED_DEEP_SIZE: usize = 207_256;
    assert_eq!(
        cache.get_size(),
        EXPECTED_DEEP_SIZE,
        "snapshot drift: deep size changed; review and update if intentional"
    );
}
