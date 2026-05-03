# post-clashcore performance baseline

Captured immediately after swapping the per-shard `std::sync::Mutex<State>`
for `clashcore::CachePadded<clashcore::lock::RwLock<State>>` (a single-CAS
uncontended `RawRwLock` on `parking_lot_core`, with each shard pinned to
its own cache line). Pairs with `bench-baselines/post-shards/` so the
"primitive swap only" delta is recoverable.

## Provenance

- clache repo SHA            : `59bb6ea6287328b071e00bcc175a2fe00bd0ece5`
- clashcore branch           : `phase-2/hashbrown-0.17` (commit `71cf4eaa`)
- rustc                      : `1.95.0 (59807616e 2026-04-14)` aarch64-apple-darwin
- host OS                    : macOS 26.4
- captured                   : 2026-05-03

## Layout

| Path                   | Contents                                                       |
| ---------------------- | -------------------------------------------------------------- |
| `criterion/`           | Mirror of `target/criterion/` after the swap. Holds `post_clashcore`, `new`, and per-scenario `change` directories so future runs can diff either way. |

## Concurrent deltas vs `post_shards` (8 worker threads)

clache only — moka and quick_cache rows in this baseline run drifted
~1–8% within run-to-run noise; the comparison numbers below are the
clache rows, which is where the swap was meant to land.

| Scenario                          | Time delta              | Throughput delta      | Notes                              |
| --------------------------------- | ----------------------- | --------------------- | ---------------------------------- |
| `get_hit_concurrent/clache/8`     | −12.1%                  | +13.8%                | Cache padding + RwLock fast path   |
| `mixed_90r10w_concurrent/clache/8`| −44.0%                  | **+78.5%**            | False-sharing elimination dominates|

Hot-key fan-out (`get_hit_concurrent`) gains ~14% from removing the
single-cache-line bounce on the shared `Mutex<State>`. The mixed
read/write workload — where each writer touches different shards over
time and the readers fan out to all four — sees the full payoff: the
shard cache lines no longer ping between cores on every cross-shard
op, and the RwLock's single-CAS exclusive path edges out `Mutex::lock`.

## Single-thread deltas vs `post_shards` (`compare_cache.rs`)

| Scenario                       | Time delta | Throughput delta | Notes                            |
| ------------------------------ | ---------- | ---------------- | -------------------------------- |
| `get_hit/clache/4096`          | −3.3%      | +3.4%            | RwLock write fast path           |
| `insert_cold/clache/1024`      | −5.0%      | +5.3%            | "                                |
| `insert_cold/clache/4096`      | −4.6%      | +4.8%            | "                                |
| `insert_cold/clache/16384`     | −6.0%      | +6.4%            | "                                |
| `insert_churn/clache/1024`     | +4.7%      | −4.5%            | RwLock write path slightly heavier than `Mutex` for tight loops |
| `insert_churn/clache/4096`     | −1.2%      | +1.2%            | within noise                     |
| `insert_churn/clache/16384`    | +0.7%      | −0.7%            | within noise                     |
| `insert_churn/clache/65536`    | +2.1%      | −2.1%            | small regression at large N      |

Net single-thread is positive: `get_hit` and `insert_cold` gain a few
percent, while `insert_churn` is mixed (small regression at the
smallest and largest working sets, likely the RwLock state machine
costing a few extra cycles per repeated write under no contention).
Acceptable trade for the multi-threaded wins above.

## How to compare against this baseline

```sh
cp -R bench-baselines/post-clashcore/criterion/* target/criterion/

# Concurrent harness:
cargo bench --bench concurrent -- --baseline post_clashcore

# Single-thread microbenches:
cargo bench --bench compare_cache -- --baseline post_clashcore
```

## Open follow-ups

- Split `get_served` so the truly read-only path (no LRU promote, no
  `expire_after_read` hook) can take `RwLock::read()` instead of
  `write()`; clashcore's `RawRwLock` is read-fair and would let
  multiple readers progress in parallel inside one shard. Today every
  shard access still calls `.write()`.
- Re-run the trace-driven `mokabench` ARC OLTP / S3 traces against
  this baseline; the deltas above are microbench-only.
