//! Trace-driven three-way comparison: clache vs moka vs quick_cache.
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
//! cargo run --release --example mokabench_compare -- s3
//! cargo run --release --example mokabench_compare -- s3 oltp
//! ```
//!
//! Without args, runs the `s3` ARC trace.

use std::env;
use std::path::PathBuf;

use mokabench::config::Config;
use mokabench::{
    Report, TraceFile, run_multi_threads_clache, run_multi_threads_moka_sync,
    run_multi_threads_quick_cache,
};

const NUM_CLIENTS: u16 = 8;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let trace_names: Vec<&str> = if args.is_empty() {
        vec!["s3"]
    } else {
        args.iter().map(String::as_str).collect()
    };

    let mokabench_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third-party/mokabench");
    env::set_current_dir(&mokabench_dir)?;

    println!("{}", Report::cvs_header(/* eviction_listener */ false));

    for name in trace_names {
        let trace = TraceFile::try_from(name)?;
        for &capacity in trace.default_capacities() {
            let config = Config::new(trace, None, None, None, None, None);

            let report = run_multi_threads_clache(&config, capacity, NUM_CLIENTS)?;
            println!("{}", report.to_csv_record());

            let report = run_multi_threads_moka_sync(&config, capacity, NUM_CLIENTS)?;
            println!("{}", report.to_csv_record());

            let report = run_multi_threads_quick_cache(&config, capacity, NUM_CLIENTS)?;
            println!("{}", report.to_csv_record());
        }
    }

    Ok(())
}
