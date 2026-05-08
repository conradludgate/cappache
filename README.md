# clache

Concurrent cache for Rust (API-first development).

**MSRV:** Rust **1.95** (`rust-version` in `Cargo.toml`).

- Crate is `#![deny(unsafe_code)]`.
- The cache implements synchronous `**get` / `insert` / `remove`**, async `**try_get_with**` / `**get_with**` (single-flight miss coalescing), and `**try_revalidate**` / `**revalidate**` (per-key refresh in flight, `Executor`-driven jobs).
- `**get_or_insert_with**` is the sync miss-coalescing variant for callers without an async runtime; it uses an internal `(Mutex, Condvar)` flight per key (no async runtime needed) and releases the per-shard lock before invoking `init`, so other keys remain responsive.
- `**try_revalidate`** / `**revalidate**` schedule work via `**Executor<RefreshWork<_>>**` (`revalidate` is the `Infallible`-error specialisation; the conversion is internal). `**RefreshWork**` is a `[Future](std::future::Future)` that polls the inner initializer and updates the map on success.

Integration tests: `**cargo test --test cache_surface**` covers builder and introspection; `**cargo test --test cache_ops**` covers data paths, concurrency, and refresh scheduling.

## License

TBD