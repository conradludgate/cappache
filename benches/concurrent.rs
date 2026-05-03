//! Multi-threaded comparative benchmarks across clache / moka / quick_cache.
//!
//! Bench crates here use `Arc<Cache<...>>` and a small barrier-driven worker
//! pool so all worker threads start the timed loop together; this lets
//! Criterion measure steady-state throughput on a sharded cache instead of
//! the cold start of each thread. Layout mirrors `compare_cache.rs`:
//! `BenchmarkId::new("<impl>", "<workers>")` keeps the criterion tree
//! grouped per scenario.

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;

use clache::{Cache, CacheBuilder, Weight};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Working set sized to keep things in-cache for the read-heavy workloads.
const CAPACITY: u64 = 4096;
/// Per-thread work budget per Criterion iteration. Big enough to amortise
/// thread sync but small enough that Criterion's outer sampling stays
/// responsive on machines with few cores.
const OPS_PER_THREAD: u64 = 4_096;
/// Default worker count for the headline scenarios. Picked to give the
/// 4-shard default cache one shard per worker without overwhelming
/// laptop-sized hosts that run the bench locally.
const WORKERS: usize = 8;

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

/// Spawn `workers` threads, each calling `op(thread_idx)` `OPS_PER_THREAD`
/// times. The barrier ensures the timed window starts after every worker
/// has reached the loop, so the measurement reflects steady-state
/// contention rather than spawn overhead.
fn run_workers<F>(workers: usize, op: F)
where
    F: Fn(usize, u64) + Send + Sync + 'static,
{
    let op = Arc::new(op);
    let barrier = Arc::new(Barrier::new(workers));
    let mut handles = Vec::with_capacity(workers);
    for tid in 0..workers {
        let op = Arc::clone(&op);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..OPS_PER_THREAD {
                op(tid, i);
            }
        }));
    }
    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

/// All workers hammer the same warm key; pure read fan-out, so it is the
/// canonical "lock contention vs sharding" stress test.
fn bench_get_hit_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_hit_concurrent");
    let workers = WORKERS;
    let total_ops = OPS_PER_THREAD * workers as u64;
    group.throughput(Throughput::Elements(total_ops));

    let probe_key: u64 = CAPACITY / 2;

    let clache_cache = Arc::new(build_clache(CAPACITY));
    fill_clache(&clache_cache, CAPACITY);
    group.bench_function(BenchmarkId::new("clache", workers), |b| {
        b.iter(|| {
            let cache = Arc::clone(&clache_cache);
            run_workers(workers, move |_tid, _i| {
                black_box(cache.get(black_box(&probe_key)));
            });
        });
    });

    let moka_cache = Arc::new(build_moka(CAPACITY));
    fill_moka(&moka_cache, CAPACITY);
    group.bench_function(BenchmarkId::new("moka", workers), |b| {
        b.iter(|| {
            let cache = Arc::clone(&moka_cache);
            run_workers(workers, move |_tid, _i| {
                black_box(cache.get(black_box(&probe_key)));
            });
        });
    });

    let quick = Arc::new(build_quick_cache(CAPACITY));
    fill_quick_cache(&quick, CAPACITY);
    group.bench_function(BenchmarkId::new("quick_cache", workers), |b| {
        b.iter(|| {
            let cache = Arc::clone(&quick);
            run_workers(workers, move |_tid, _i| {
                black_box(cache.get(black_box(&probe_key)));
            });
        });
    });

    group.finish();
}

/// Read/write mix on a shared hot working set. 90% reads, 10% writes; the
/// write key rotates per op so each writer touches different shards over
/// time, exercising both the read fast path and the eviction path under
/// contention.
fn bench_mixed_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_90r10w_concurrent");
    let workers = WORKERS;
    let total_ops = OPS_PER_THREAD * workers as u64;
    group.throughput(Throughput::Elements(total_ops));

    let clache_cache = Arc::new(build_clache(CAPACITY));
    fill_clache(&clache_cache, CAPACITY);
    group.bench_function(BenchmarkId::new("clache", workers), |b| {
        b.iter(|| {
            let cache = Arc::clone(&clache_cache);
            run_workers(workers, move |tid, i| {
                if i % 10 == 0 {
                    let k = (tid as u64).wrapping_mul(1_000_003).wrapping_add(i);
                    cache.insert(black_box(k), k);
                } else {
                    let k = i % CAPACITY;
                    black_box(cache.get(black_box(&k)));
                }
            });
        });
    });

    let moka_cache = Arc::new(build_moka(CAPACITY));
    fill_moka(&moka_cache, CAPACITY);
    group.bench_function(BenchmarkId::new("moka", workers), |b| {
        b.iter(|| {
            let cache = Arc::clone(&moka_cache);
            run_workers(workers, move |tid, i| {
                if i % 10 == 0 {
                    let k = (tid as u64).wrapping_mul(1_000_003).wrapping_add(i);
                    cache.insert(black_box(k), k);
                } else {
                    let k = i % CAPACITY;
                    black_box(cache.get(black_box(&k)));
                }
            });
        });
    });

    let quick = Arc::new(build_quick_cache(CAPACITY));
    fill_quick_cache(&quick, CAPACITY);
    group.bench_function(BenchmarkId::new("quick_cache", workers), |b| {
        b.iter(|| {
            let cache = Arc::clone(&quick);
            run_workers(workers, move |tid, i| {
                if i % 10 == 0 {
                    let k = (tid as u64).wrapping_mul(1_000_003).wrapping_add(i);
                    cache.insert(black_box(k), k);
                } else {
                    let k = i % CAPACITY;
                    black_box(cache.get(black_box(&k)));
                }
            });
        });
    });

    group.finish();
}

criterion_group!(benches, bench_get_hit_concurrent, bench_mixed_concurrent);
criterion_main!(benches);
