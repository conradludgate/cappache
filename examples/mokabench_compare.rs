//! Trace-driven comparison: clache vs moka vs quick_cache, across the
//! `xxh3` / `ahash` / `foldhash` hashers for the implementations whose
//! mokabench drivers we own (clache + quick_cache). Moka stays pinned
//! to xxh3 because we don't own its driver here.
//!
//! Drives the in-tree fork at `third-party/mokabench` (with the `clache`
//! feature enabled). The fork's `TraceFile::path()` is rooted at
//! `./cache-trace`, so we `chdir` into the submodule before running. The
//! `cache-trace` submodule must be initialised:
//!
//! ```text
//! git submodule update --init --recursive third-party/mokabench
//! ```
//!
//! Usage:
//!
//! ```text
//! # all hashers, all three implementations, default (s3) trace:
//! cargo run --release --example mokabench_compare
//!
//! # specific traces, all hashers:
//! cargo run --release --example mokabench_compare -- s3 oltp
//!
//! # specific hashers (anywhere in the args), default trace set:
//! cargo run --release --example mokabench_compare -- --hasher=foldhash s3
//! cargo run --release --example mokabench_compare -- --hasher=xxh3,ahash ds1
//! ```

use std::env;
use std::path::PathBuf;

use mokabench::config::Config;
use mokabench::{
    HasherKind, Report, TraceFile, run_multi_threads_clache, run_multi_threads_moka_sync,
    run_multi_threads_quick_cache,
};

const NUM_CLIENTS: u16 = 8;

fn parse_hashers(arg: &str) -> anyhow::Result<Vec<HasherKind>> {
    arg.split(',')
        .map(|tok| match tok.trim() {
            "xxh3" => Ok(HasherKind::Xxh3),
            "ahash" => Ok(HasherKind::AHash),
            "foldhash" => Ok(HasherKind::Foldhash),
            other => Err(anyhow::anyhow!(
                "unknown hasher `{other}`; expected one of xxh3, ahash, foldhash",
            )),
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut hashers: Vec<HasherKind> =
        vec![HasherKind::Xxh3, HasherKind::AHash, HasherKind::Foldhash];
    let mut trace_names: Vec<String> = Vec::new();

    for arg in env::args().skip(1) {
        if let Some(rest) = arg.strip_prefix("--hasher=") {
            hashers = parse_hashers(rest)?;
        } else {
            trace_names.push(arg);
        }
    }
    if trace_names.is_empty() {
        trace_names.push("s3".into());
    }

    let mokabench_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third-party/mokabench");
    env::set_current_dir(&mokabench_dir)?;

    println!("{}", Report::cvs_header(/* eviction_listener */ false));

    for name in &trace_names {
        let trace = TraceFile::try_from(name.as_str())?;
        for &capacity in trace.default_capacities() {
            let config = Config::new(trace, None, None, None, None, None);

            for &hasher in &hashers {
                let report = run_multi_threads_clache(&config, capacity, NUM_CLIENTS, hasher)?;
                println!("{}", report.to_csv_record());
            }

            // Moka stays on xxh3; its driver isn't parameterised here.
            let report = run_multi_threads_moka_sync(&config, capacity, NUM_CLIENTS)?;
            println!("{}", report.to_csv_record());

            for &hasher in &hashers {
                let report = run_multi_threads_quick_cache(&config, capacity, NUM_CLIENTS, hasher)?;
                println!("{}", report.to_csv_record());
            }
        }
    }

    Ok(())
}
