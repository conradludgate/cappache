use std::hash::{BuildHasherDefault, Hasher};
use std::time::{Duration, Instant};

use cappache::{Cache, CacheBuilder, Policy, TIME_UNBOUNDED, Weight};

// Place u64 keys 0 and 1 in different shards deterministically.
#[derive(Default)]
struct SplitHasher(u64);

impl Hasher for SplitHasher {
    fn finish(&self) -> u64 {
        self.0 << 56
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = self.0.wrapping_mul(31).wrapping_add(u64::from(*byte));
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

struct MaximumWeight;

impl Policy<u64, ()> for MaximumWeight {
    fn weight(&self, _key: &u64, _value: &()) -> Weight {
        Weight::new(u64::MAX)
    }

    fn expire_after_read(
        &self,
        _key: &u64,
        _value: &(),
        _read_at: Instant,
        _last_modified_at: Instant,
        _time_remaining: Duration,
    ) -> Duration {
        TIME_UNBOUNDED
    }

    fn expire_after_write(&self, _key: &u64, _value: &(), _written_at: Instant) -> Duration {
        TIME_UNBOUNDED
    }
}

#[test]
fn total_weight_saturates_and_recovers_after_removal() {
    let cache = Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(u64::MAX))
            .shards(2)
            .hasher(BuildHasherDefault::<SplitHasher>::default())
            .policy(MaximumWeight)
            .build(),
    );
    cache.insert(0, ());
    cache.insert(1, ());
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&0), Some(()));
    assert_eq!(cache.get(&1), Some(()));
    assert_eq!(cache.weight(), u64::MAX);

    assert_eq!(cache.remove(&0), Some(()));
    assert_eq!(cache.weight(), u64::MAX);
    assert_eq!(cache.remove(&1), Some(()));
    assert_eq!(cache.weight(), 0);
}
