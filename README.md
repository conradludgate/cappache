# clache

Concurrent cache for Rust (API-first development).

- Crate is `#![deny(unsafe_code)]`.
- Public API is in place; cache operation bodies are still `todo!()` while behavior is implemented.
- **`get` / `insert` / `remove`** are synchronous; **`try_get_with`** / **`get_with`** are async, with **`get_with_blocking`** for callers without an executor (uses an internal park-based driver, not exported)—**do not** use blocking from async worker threads if the initializer needs that same runtime.
- **`try_revalidate`** / **`revalidate`** schedule background refresh via **`Executor<RefreshWork<_>>`** (`revalidate` uses **`OkIntoResult<Fut>`** in that `RefreshWork` type parameter; stubs for now).

Design intent (O(1) targets, deferred drops, `Policy`-driven time, etc.) is in [docs/DESIGN.md](docs/DESIGN.md).

## License

TBD
