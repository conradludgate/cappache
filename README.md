# clache

Concurrent cache for Rust (API-first development).

- Crate is `#![deny(unsafe_code)]`.
- The cache implements synchronous **`get` / `insert` / `remove`**, async **`try_get_with`** / **`get_with`** (single-flight miss coalescing), and **`try_revalidate`** / **`revalidate`** (per-key refresh in flight, `Executor`-driven jobs).
- **`get_with_blocking`** is for callers without an async runtime (internal park-based driver, not exported)—**do not** use it from async worker threads if the initializer needs that same runtime.
- **`try_revalidate`** / **`revalidate`** schedule work via **`Executor<RefreshWork<_>>`** (`revalidate` wraps the initializer with **`OkIntoResult<Fut>`**). **`RefreshWork`** is a [`Future`](std::future::Future) that polls the inner initializer and updates the map on success.

Integration tests: **`cargo test --test cache_surface`** covers builder and introspection; **`cargo test --test cache_ops`** covers data paths, concurrency, and refresh scheduling.

Design intent (O(1) targets, deferred drops, `Policy`-driven time, etc.) is in [docs/DESIGN.md](docs/DESIGN.md).

## License

TBD
