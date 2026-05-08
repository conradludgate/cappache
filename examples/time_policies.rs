//! Time-based eviction via `Policy`: `TtlFromWrite`, `IdleTimeout`,
//! their composition, and a custom `Policy` that uses string length
//! as the entry's weight.
//!
//! Time-based eviction is opportunistic: an entry's deadline is
//! checked on the next access, and reclaim drains any past-deadline
//! entries before installing a new one. There is no background
//! sweeper thread.
//!
//! Run with: `cargo run --example time_policies`

use std::thread;
use std::time::{Duration, Instant};

use clache::policy::{IdleTimeout, Policy, TIME_UNBOUNDED, TtlFromWrite, Weight};
use clache::{Cache, CacheBuilder};

/// Custom `Policy`: weight equals the value's UTF-8 byte length.
/// Time hooks delegate to the cache's running deadline (no extra
/// shortening from this layer).
#[derive(Clone)]
struct LenWeight;

impl Policy<String, String> for LenWeight {
    fn weight(&self, _key: &String, value: &String) -> Weight {
        Weight::new(value.len() as u64)
    }

    fn expire_after_read(
        &self,
        _key: &String,
        _value: &String,
        _read_at: Instant,
        _last_modified_at: Instant,
        time_remaining: Duration,
    ) -> Duration {
        time_remaining
    }

    fn expire_after_write(&self, _key: &String, _value: &String, _written_at: Instant) -> Duration {
        TIME_UNBOUNDED
    }
}

fn main() {
    // Compose: 200 ms idle timeout wrapping a 500 ms TTL wrapping a
    // custom weight policy. Each layer applies `.min()` to the inner
    // deadline, so the strictest rule wins.
    let policy = IdleTimeout::new(
        Duration::from_millis(200),
        TtlFromWrite::new(Duration::from_millis(500), LenWeight),
    );

    let cache = Cache::new(
        CacheBuilder::<String, String>::new()
            .max_weight(Weight::new(64))
            .policy(policy)
            .build(),
    );

    cache.insert("greeting".into(), "hello".into());
    println!("inserted (cache weight = {} bytes)", cache.weight());

    // Read inside the idle window: hit, and the deadline extends to
    // about now + 200 ms.
    thread::sleep(Duration::from_millis(150));
    assert_eq!(cache.get("greeting"), Some("hello".to_string()));
    println!("after 150 ms idle + read: still present");

    // Sit idle past the 200 ms idle window: expires lazily on next
    // access.
    thread::sleep(Duration::from_millis(250));
    assert_eq!(cache.get("greeting"), None);
    println!("after 250 ms idle without reads: expired");

    // TTL is anchored at the last *write*. Reads inside the idle
    // window keep the entry alive until the absolute TTL fires:
    cache.insert("note".into(), "from-write".into());
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(600) {
        let _ = cache.get("note");
        thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(cache.get("note"), None, "TTL caps absolute lifetime");
    println!("after 600 ms with periodic reads: TTL still expired the entry");
}
