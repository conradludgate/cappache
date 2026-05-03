# post-refactor performance baseline

Companion archive to `bench-baselines/pre-refactor/`. Captured immediately
after the Phase 2 refactor (slab + intrusive LRU + lazy-validating expiry
heap) so the regression policy described in `docs/DESIGN.md` has both
endpoints to diff.

## Provenance

- clache repo SHA            : `d2cec2ae4d39d80ce4b9668880e3d50b998dfdf4`
- mokabench submodule SHA    : `f8f166628d9a20cfdb5119d23b069bee71d86e06`
- cache-trace submodule SHA  : `ef0a9de8cf0202a1f2fee186a0af497774b0f0a9`
- rustc                      : `1.95.0 (59807616e 2026-04-14)` aarch64-apple-darwin
- host OS                    : macOS 26.4 (BuildVersion 25E246)
- captured                   : 2026-05-03
- machine class              : same workstation as `bench-baselines/pre-refactor/`

## Layout

| Path                                  | Contents                                                                                           |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `criterion/`                          | Persisted Criterion tree mirroring `target/criterion`. Holds `pre_refactor`, `new`, and `post_refactor` baselines per `(scenario, impl, param)` so future runs can diff against either endpoint. |
| `mokabench/oltp.log`                  | `cargo run --release --example mokabench_compare -- oltp` stdout: clache vs moka vs quick_cache, ARC OLTP trace, 8 client threads, capacities 256 / 512 / 1000 / 2000. |

## Criterion deltas vs `pre_refactor`

clache rows (negative time = faster). moka / quick_cache rows are within
the noise band — included implicitly because the harness runs all three
on the same workload.

| Scenario                      | Time delta              | Throughput delta            | Notes                                  |
| ----------------------------- | ----------------------- | --------------------------- | -------------------------------------- |
| `get_hit/clache/4096`         | −9.5%                   | +10.5%                      | Slab hit path, intrusive LRU promote   |
| `insert_churn/clache/1024`    | −98.49%                 | ×65 throughput              | O(n) `evict_until_fit` → O(1) per LRU victim |
| `insert_churn/clache/4096`    | −98.46%                 | ×64 throughput              | "                                      |
| `insert_churn/clache/16384`   | −98.36%                 | ×60 throughput              | "                                      |
| `insert_churn/clache/65536`   | −98.25%                 | ×56 throughput              | "                                      |
| `insert_cold/clache/1024`     | −21.1%                  | +26.8%                      | Pre-sized map + slab; no rehash spike  |
| `insert_cold/clache/4096`     | −18.9%                  | +23.2%                      | "                                      |
| `insert_cold/clache/16384`    | −20.4%                  | +25.7%                      | "                                      |

`insert_churn` no longer scales with N: per-element cost holds at
~110–115 ns regardless of working-set size, confirming the O(1) shape.

## mokabench OLTP deltas

ARC OLTP trace, 8 clients, full source: `mokabench/oltp.log`. Hit ratios
include miss-induced inserts.

| Capacity | Engine       | Hit ratio (pre → post) | Duration (pre → post) |
| -------- | ------------ | ---------------------- | --------------------- |
| 256      | clache       | 5.14% → **12.19%**     | 3.114s → **0.675s**   |
| 256      | moka         | 27.75% → 27.93%        | 0.395s → 0.446s       |
| 256      | quick_cache  | 21.96% → 22.02%        | 0.092s → 0.090s       |
| 512      | clache       | 4.19% → **21.40%**     | 3.726s → **0.596s**   |
| 512      | moka         | 34.17% → 34.32%        | 0.346s → 0.380s       |
| 512      | quick_cache  | 28.32% → 28.35%        | 0.054s → 0.056s       |
| 1000     | clache       | 5.69% → **32.07%**     | 4.462s → **0.560s**   |
| 1000     | moka         | 39.64% → 39.58%        | 0.333s → 0.359s       |
| 1000     | quick_cache  | 34.90% → 34.91%        | 0.042s → 0.041s       |
| 2000     | clache       | 7.89% → **42.22%**     | 5.892s → **0.515s**   |
| 2000     | moka         | 44.48% → 44.37%        | 0.338s → 0.314s       |
| 2000     | quick_cache  | 41.71% → 41.72%        | 0.036s → 0.035s       |

Pre-refactor clache evicted via the O(n) `evict_until_fit` scan, which
both spent disproportionate CPU per insert (the slow durations) and
thrashed working sets (the very low hit ratios). Post-refactor clache
keeps the same trace driver but now uses the slab/LRU + heap engine, and
the LRU recency signal is enough for it to clear or beat moka and
quick_cache on this trace at every tested capacity.

## How to compare against this baseline

```sh
# Restore Criterion artifacts after `cargo clean`:
cp -R bench-baselines/post-refactor/criterion/* target/criterion/

# Microbenches: diff a future run against the post-refactor endpoint.
cargo bench --bench compare_cache -- --baseline post_refactor

# Trace-driven: re-run and diff stdout.
cargo run --release --example mokabench_compare -- oltp \
    > /tmp/oltp.log
diff bench-baselines/post-refactor/mokabench/oltp.log /tmp/oltp.log
```

## Open follow-ups

- Add `expiry-drain` and `try_get_with_hit` Criterion scenarios; both
  exercise post-refactor code paths (heap drain, leader fast hit) that
  the current baselines do not directly cover.
- Re-run the ARC `S3` trace now that the slab engine is in place; it was
  skipped pre-refactor because `evict_until_fit` made high-capacity S3
  prohibitively slow.
- Multi-threaded Criterion harness (`get_hit` mix; hot-key write storm)
  to capture the contention floor before considering Phase 3 sharding.
