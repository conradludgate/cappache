//! Three-way comparative microbenchmarks across clache / moka / quick_cache.
//!
//! Identical workloads on identical configs so Criterion can diff the impls
//! directly. Each scenario uses `BenchmarkId::new("<impl>", <param>)` inside
//! one group so `target/criterion/<scenario>/<impl>/<param>` lays the impls
//! out side by side.

use std::hint::black_box;

use clache::{Cache, CacheBuilder, Weight};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Working set used for `get_hit` and the warm path of `insert_churn`.
/// Picked small enough to fit `INSERT_CHURN_CAPACITY` without eviction yet
/// large enough to exercise the bucket distribution for hash-table-backed
/// implementations.
const GET_HIT_CAPACITY: u64 = 4096;
const INSERT_CHURN_CAPACITY: u64 = 4096;

fn build_clache(capacity: u64) -> Cache<u64, u64> {
    Cache::new(
        CacheBuilder::<u64, u64>::new()
            .max_weight(Weight::new(capacity))
            .build(),
    )
}

fn build_moka(capacity: u64) -> moka::sync::Cache<u64, u64> {
    moka::sync::Cache::builder().max_capacity(capacity).build()
}

fn build_quick_cache(capacity: u64) -> quick_cache::sync::Cache<u64, u64> {
    quick_cache::sync::Cache::new(capacity as usize)
}

fn fill_clache(cache: &Cache<u64, u64>, n: u64) {
    for k in 0..n {
        cache.insert(k, k);
    }
}

fn fill_moka(cache: &moka::sync::Cache<u64, u64>, n: u64) {
    for k in 0..n {
        cache.insert(k, k);
    }
    cache.run_pending_tasks();
}

fn fill_quick_cache(cache: &quick_cache::sync::Cache<u64, u64>, n: u64) {
    for k in 0..n {
        cache.insert(k, k);
    }
}

/// Repeated `get` of one warm key on a fully-populated cache.
fn bench_get_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_hit");
    group.throughput(Throughput::Elements(1));

    let probe_key: u64 = GET_HIT_CAPACITY / 2;

    let clache_cache = build_clache(GET_HIT_CAPACITY);
    fill_clache(&clache_cache, GET_HIT_CAPACITY);
    group.bench_function(BenchmarkId::new("clache", GET_HIT_CAPACITY), |b| {
        b.iter(|| black_box(clache_cache.get(black_box(&probe_key))));
    });

    let moka_cache = build_moka(GET_HIT_CAPACITY);
    fill_moka(&moka_cache, GET_HIT_CAPACITY);
    group.bench_function(BenchmarkId::new("moka", GET_HIT_CAPACITY), |b| {
        b.iter(|| black_box(moka_cache.get(black_box(&probe_key))));
    });

    let quick = build_quick_cache(GET_HIT_CAPACITY);
    fill_quick_cache(&quick, GET_HIT_CAPACITY);
    group.bench_function(BenchmarkId::new("quick_cache", GET_HIT_CAPACITY), |b| {
        b.iter(|| black_box(quick.get(black_box(&probe_key))));
    });

    group.finish();
}

/// Stream of `n` distinct inserts into a fixed-capacity cache; once `n` exceeds
/// capacity each insert forces eviction. Varying `n` lets reviewers see the
/// per-insert cost scale (or fail to scale) with working-set size.
fn bench_insert_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_churn");
    let cap = INSERT_CHURN_CAPACITY;

    for &n in &[1_024u64, 4_096, 16_384, 65_536] {
        group.throughput(Throughput::Elements(n));

        group.bench_with_input(BenchmarkId::new("clache", n), &n, |b, &n| {
            let cache = build_clache(cap);
            fill_clache(&cache, cap);
            let mut counter: u64 = cap;
            b.iter(|| {
                for _ in 0..n {
                    cache.insert(black_box(counter), counter);
                    counter = counter.wrapping_add(1);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("moka", n), &n, |b, &n| {
            let cache = build_moka(cap);
            fill_moka(&cache, cap);
            let mut counter: u64 = cap;
            b.iter(|| {
                for _ in 0..n {
                    cache.insert(black_box(counter), counter);
                    counter = counter.wrapping_add(1);
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("quick_cache", n), &n, |b, &n| {
            let cache = build_quick_cache(cap);
            fill_quick_cache(&cache, cap);
            let mut counter: u64 = cap;
            b.iter(|| {
                for _ in 0..n {
                    cache.insert(black_box(counter), counter);
                    counter = counter.wrapping_add(1);
                }
            });
        });
    }

    group.finish();
}

/// Insert `n` distinct keys into a freshly built cache sized so no eviction
/// happens. Isolates structural insert cost from eviction work.
fn bench_insert_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_cold");

    for &n in &[1_024u64, 4_096, 16_384] {
        group.throughput(Throughput::Elements(n));
        let capacity = n * 2;

        group.bench_with_input(BenchmarkId::new("clache", n), &n, |b, &n| {
            b.iter_batched(
                || build_clache(capacity),
                |cache| {
                    for k in 0..n {
                        cache.insert(black_box(k), k);
                    }
                    cache
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("moka", n), &n, |b, &n| {
            b.iter_batched(
                || build_moka(capacity),
                |cache| {
                    for k in 0..n {
                        cache.insert(black_box(k), k);
                    }
                    cache
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("quick_cache", n), &n, |b, &n| {
            b.iter_batched(
                || build_quick_cache(capacity),
                |cache| {
                    for k in 0..n {
                        cache.insert(black_box(k), k);
                    }
                    cache
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_get_hit,
    bench_insert_churn,
    bench_insert_cold
);
criterion_main!(benches);
