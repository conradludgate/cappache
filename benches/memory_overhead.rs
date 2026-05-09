//! Memory-overhead benchmark: clache vs moka vs quick_cache.
//!
//! Runs a deterministic mokabench-style trace (1M ops over 100K keys,
//! ~80/20 read/write split, Pareto-skewed access) on each cache and
//! reports the alloc/dealloc counts, net resting bytes, and peak
//! resident bytes during the workload phase.
//!
//! TTL is `carry_only`: every cache pays the cost of tracking a 1h
//! deadline but no entry expires during the run, so we measure the
//! steady-state overhead of the deadline machinery. `quick_cache` lacks
//! a native TTL, so its value is wrapped as `(Instant, String)` to pay
//! an equivalent footprint.

use std::time::{Duration, Instant};

use clache::{Cache as Clache, CacheBuilder, DefaultPolicy, TtlFromWrite, Weight};
use moka::sync::Cache as MokaCache;
use peak_alloc::PeakAlloc;
use quick_cache::sync::Cache as QuickCache;
use stats_alloc::{Region, StatsAlloc};

#[global_allocator]
static GLOBAL: StatsAlloc<PeakAlloc> = StatsAlloc::new(PeakAlloc);

/// Reads PEAK/CURRENT atomics from the inner allocator. Any `PeakAlloc`
/// instance reads the same crate-internal globals, so this works even
/// though `StatsAlloc` does not expose its inner allocator.
const PEAK: PeakAlloc = PeakAlloc;

const KEYSPACE: usize = 100_000;
const OPS: usize = 1_000_000;
const KEY_LEN: usize = 32;
const VAL_LEN: usize = 256;
const TTL: Duration = Duration::from_secs(3600);
const SEED: u64 = 0xC1ACE;

/// Pareto-style: ~80% of ops touch the first 20% of the keyspace.
const HOT_KEY_FRAC: f64 = 0.20;
const HOT_OP_FRAC: f64 = 0.80;
/// Mokabench-style: 80/20 reads/writes.
const WRITE_FRAC: f64 = 0.20;

#[derive(Clone, Copy)]
enum Op {
    Get(u32),
    Put(u32),
}

fn build_keyspace() -> Vec<String> {
    (0..KEYSPACE)
        .map(|i| format!("k{i:0>width$}", width = KEY_LEN - 1))
        .collect()
}

fn canonical_value() -> String {
    "x".repeat(VAL_LEN)
}

fn build_trace() -> Vec<Op> {
    let mut rng = fastrand::Rng::with_seed(SEED);
    let hot_cutoff = ((KEYSPACE as f64) * HOT_KEY_FRAC) as usize;
    let mut trace = Vec::with_capacity(OPS);
    for _ in 0..OPS {
        let raw_idx = if rng.f64() < HOT_OP_FRAC {
            rng.usize(0..hot_cutoff)
        } else {
            rng.usize(hot_cutoff..KEYSPACE)
        };
        let idx = raw_idx as u32;
        trace.push(if rng.f64() < WRITE_FRAC {
            Op::Put(idx)
        } else {
            Op::Get(idx)
        });
    }
    trace
}

trait CacheUnderTest {
    fn name(&self) -> &'static str;
    fn get_or_insert(&self, key: &str, value: &str);
    fn put(&self, key: &str, value: &str);
    fn entry_count(&self) -> usize;
    fn settle(&self);
}

struct ClacheAdapter(Clache<String, String, TtlFromWrite<DefaultPolicy>>);

impl ClacheAdapter {
    fn new() -> Self {
        // Capacity 4x the keyspace so the workload never triggers
        // size-driven eviction; we want to measure the resting cost,
        // not S3-FIFO recirculation.
        let cfg = CacheBuilder::<String, String>::new()
            .max_weight(Weight::new(KEYSPACE as u64 * 4))
            .policy(TtlFromWrite::new(TTL, DefaultPolicy))
            .build();
        Self(Clache::new(cfg))
    }
}

impl CacheUnderTest for ClacheAdapter {
    fn name(&self) -> &'static str {
        "clache"
    }
    fn get_or_insert(&self, key: &str, value: &str) {
        if self.0.get(key).is_some() {
            return;
        }
        let _ = self.0.insert(key.to_owned(), value.to_owned());
    }
    fn put(&self, key: &str, value: &str) {
        let _ = self.0.insert(key.to_owned(), value.to_owned());
    }
    fn entry_count(&self) -> usize {
        self.0.len()
    }
    fn settle(&self) {}
}

struct MokaAdapter(MokaCache<String, String>);

impl MokaAdapter {
    fn new() -> Self {
        Self(
            MokaCache::builder()
                .max_capacity((KEYSPACE as u64) * 4)
                .time_to_live(TTL)
                .build(),
        )
    }
}

impl CacheUnderTest for MokaAdapter {
    fn name(&self) -> &'static str {
        "moka"
    }
    fn get_or_insert(&self, key: &str, value: &str) {
        if self.0.get(key).is_some() {
            return;
        }
        self.0.insert(key.to_owned(), value.to_owned());
    }
    fn put(&self, key: &str, value: &str) {
        self.0.insert(key.to_owned(), value.to_owned());
    }
    fn entry_count(&self) -> usize {
        self.0.entry_count() as usize
    }
    fn settle(&self) {
        self.0.run_pending_tasks();
    }
}

struct QuickCacheAdapter(QuickCache<String, (Instant, String)>);

impl QuickCacheAdapter {
    fn new() -> Self {
        Self(QuickCache::new(KEYSPACE * 4))
    }
}

impl CacheUnderTest for QuickCacheAdapter {
    fn name(&self) -> &'static str {
        "quick_cache"
    }
    fn get_or_insert(&self, key: &str, value: &str) {
        if self.0.get(key).is_some() {
            return;
        }
        self.0
            .insert(key.to_owned(), (Instant::now() + TTL, value.to_owned()));
    }
    fn put(&self, key: &str, value: &str) {
        self.0
            .insert(key.to_owned(), (Instant::now() + TTL, value.to_owned()));
    }
    fn entry_count(&self) -> usize {
        self.0.len()
    }
    fn settle(&self) {}
}

fn run_phase(cache: &dyn CacheUnderTest, keyspace: &[String], trace: &[Op], value: &str) {
    for &op in trace {
        match op {
            Op::Get(idx) => cache.get_or_insert(&keyspace[idx as usize], value),
            Op::Put(idx) => cache.put(&keyspace[idx as usize], value),
        }
    }
}

struct PhaseResult {
    name: &'static str,
    allocs: u64,
    deallocs: u64,
    bytes_alloc: u64,
    bytes_dealloc: u64,
    peak_bytes: usize,
    entries: usize,
}

fn measure(
    cache: &dyn CacheUnderTest,
    keyspace: &[String],
    trace: &[Op],
    value: &str,
) -> PhaseResult {
    cache.settle();
    PEAK.reset_peak_usage();
    let region = Region::new(&GLOBAL);
    run_phase(cache, keyspace, trace, value);
    cache.settle();
    let stats = region.change();
    let peak = PEAK.peak_usage();
    PhaseResult {
        name: cache.name(),
        allocs: stats.allocations as u64,
        deallocs: stats.deallocations as u64,
        bytes_alloc: stats.bytes_allocated as u64,
        bytes_dealloc: stats.bytes_deallocated as u64,
        peak_bytes: peak,
        entries: cache.entry_count(),
    }
}

fn measure_clache(keyspace: &[String], trace: &[Op], value: &str) -> PhaseResult {
    let cache = ClacheAdapter::new();
    measure(&cache, keyspace, trace, value)
}

fn measure_moka(keyspace: &[String], trace: &[Op], value: &str) -> PhaseResult {
    let cache = MokaAdapter::new();
    measure(&cache, keyspace, trace, value)
}

fn measure_quick_cache(keyspace: &[String], trace: &[Op], value: &str) -> PhaseResult {
    let cache = QuickCacheAdapter::new();
    measure(&cache, keyspace, trace, value)
}

fn print_header() {
    println!(
        "{:<13} {:>10} {:>10} {:>14} {:>14} {:>14} {:>10}",
        "cache", "allocs", "deallocs", "bytes_alloc", "bytes_freed", "peak_bytes", "entries"
    );
    println!("{}", "-".repeat(91));
}

fn print_row(r: &PhaseResult) {
    println!(
        "{:<13} {:>10} {:>10} {:>14} {:>14} {:>14} {:>10}",
        r.name, r.allocs, r.deallocs, r.bytes_alloc, r.bytes_dealloc, r.peak_bytes, r.entries
    );
}

fn main() {
    println!(
        "memory_overhead: keyspace={KEYSPACE}, ops={OPS}, key={KEY_LEN}B, value={VAL_LEN}B, ttl={TTL:?} (carry_only)"
    );
    let keyspace = build_keyspace();
    let value = canonical_value();
    let trace = build_trace();
    let payload_floor = (KEY_LEN + VAL_LEN) * KEYSPACE;
    println!("payload floor (key + value bytes per slot * keyspace) = {payload_floor} bytes");
    println!();
    print_header();
    let results = [
        measure_clache(&keyspace, &trace, &value),
        measure_moka(&keyspace, &trace, &value),
        measure_quick_cache(&keyspace, &trace, &value),
    ];
    for r in &results {
        print_row(r);
    }
}
