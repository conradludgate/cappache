//! White-box tests over `Inner`. Black-box `Cache` behaviour lives in
//! `tests/cache_ops.rs`.

/// Pins every key to shard 0 (clears the top 8 bits used by
/// `(h << 7) >> 63`) while preserving lower-bit fingerprints so the
/// FIFO / ghost / heap invariants stay observable.
#[derive(Clone, Copy, Default)]
struct SingleShardHasher;

struct SingleShardHasherImpl(std::collections::hash_map::DefaultHasher);

impl std::hash::Hasher for SingleShardHasherImpl {
    fn finish(&self) -> u64 {
        self.0.finish() & ((1u64 << 56) - 1)
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

impl std::hash::BuildHasher for SingleShardHasher {
    type Hasher = SingleShardHasherImpl;
    fn build_hasher(&self) -> SingleShardHasherImpl {
        SingleShardHasherImpl(std::collections::hash_map::DefaultHasher::new())
    }
}

mod size_regressions {
    use super::super::*;

    /// Niche-packed: the slotmap version field absorbs the discriminant.
    #[test]
    fn option_slot_key_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<SlotKey>(), 8);
        assert_eq!(std::mem::size_of::<Option<SlotKey>>(), 8);
    }

    /// Upper bound (not exact byte count) so the test stays portable.
    /// Catches accidental field-width bumps and lost niches.
    #[test]
    fn entry_slot_size_within_budget() {
        const BUDGET: usize = 152;
        let observed = std::mem::size_of::<EntrySlot<String, String>>();
        assert!(
            observed <= BUDGET,
            "EntrySlot grew to {observed} bytes (budget {BUDGET}); \
             check field widths and Option<SlotKey> niche packing",
        );
    }
}

mod shard_tests {
    use std::sync::Arc;

    use super::super::*;
    use crate::config::{CacheBuilder, DynPolicy};
    use crate::policy::{DefaultPolicy, Weight};

    type K = String;
    type V = String;

    fn build(shards: usize, max_weight: u64) -> Arc<Inner<K, V, DynPolicy<K, V>>> {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        let cfg = CacheBuilder::<K, V>::new()
            .max_weight(Weight::new(max_weight))
            .shards(shards)
            .policy(policy)
            .build();
        Arc::new(Inner::new(cfg))
    }

    #[test]
    fn shard_for_is_stable_across_clones() {
        let inner = build(8, 64);
        let clone = Arc::clone(&inner);
        for k in ["alpha", "beta", "gamma", "delta", "omega"] {
            let lhs = std::ptr::from_ref(inner.shard_for(k));
            let rhs = std::ptr::from_ref(clone.shard_for(k));
            assert!(
                std::ptr::eq(lhs, rhs),
                "key {k} mapped to different shards across clones",
            );
        }
    }

    /// Total observed weight may overshoot `max_weight` by up to
    /// `shards - 1` (each shard rounds its share up). Lock the bound.
    #[test]
    fn multi_shard_total_weight_bounded_by_max_plus_slack() {
        let shards: usize = 8;
        let max_weight: u64 = 17;
        let inner = build(shards, max_weight);
        let upper = max_weight + shards as u64 - 1;
        for i in 0..1024 {
            inner.insert(format!("key-{i}"), "v".to_string());
            let w = inner.weight();
            assert!(
                w <= upper,
                "weight {w} exceeded documented upper bound {upper} after k{i}",
            );
        }
    }

    /// `IdleTimeout` pushes a fresh deadline tuple per read; without a
    /// rebuild trigger the heap grows without bound. The trigger fires
    /// once stale tuples exceed live slab size.
    #[test]
    fn heap_stays_bounded_under_read_extending_policy() {
        use std::time::Duration;

        use crate::policy::{IdleTimeout, Weight};

        let policy: DynPolicy<K, V> = Arc::new(IdleTimeout::new(
            Duration::from_mins(1),
            crate::policy::DefaultPolicy,
        ));
        // `shards=2` halves the per-shard cap; double max_weight so
        // shard 0 (where SingleShardHasher pins everything) caps at 8.
        let cfg = CacheBuilder::<K, V>::new()
            .max_weight(Weight::new(16))
            .shards(2)
            .hasher(super::SingleShardHasher)
            .policy(policy)
            .build();
        let inner = Arc::new(Inner::new(cfg));

        for i in 0..8 {
            inner.insert(format!("k{i}"), "v".to_string());
        }

        for _ in 0..1024 {
            for i in 0..8 {
                let _ = inner.get_served(&format!("k{i}"));
            }
        }

        let occupied = inner
            .shards
            .shards()
            .iter()
            .find(|s| !s.read().slab.is_empty())
            .expect("at least one shard holds the inserted keys");
        let st = occupied.write();
        let live = st.slab.len();
        let heap_len = st.expiry.len();
        assert!(
            heap_len <= 2 * live + 1,
            "expiry heap grew to {heap_len} for {live} live entries; rebuild trigger broken",
        );
    }

    /// Filling one shard must not evict entries from another.
    #[test]
    fn eviction_stays_within_the_owning_shard() {
        let inner = build(4, 16);
        let target_shard = std::ptr::from_ref(inner.shard_for(&"anchor".to_string()));

        let mut sentinels: Vec<String> = Vec::new();
        let mut tries = 0u32;
        while sentinels.len() < 3 && tries < 10_000 {
            let candidate = format!("sentinel-{tries}");
            if !std::ptr::eq(
                std::ptr::from_ref(inner.shard_for(&candidate)),
                target_shard,
            ) {
                inner.insert(candidate.clone(), "x".into());
                sentinels.push(candidate);
            }
            tries += 1;
        }
        assert_eq!(
            sentinels.len(),
            3,
            "could not find one sentinel per non-target shard; check hasher distribution"
        );

        let mut storm = 0u32;
        let mut placed = 0u32;
        while placed < 64 && storm < 100_000 {
            let candidate = format!("hot-{storm}");
            if std::ptr::eq(
                std::ptr::from_ref(inner.shard_for(&candidate)),
                target_shard,
            ) {
                inner.insert(candidate, "y".into());
                placed += 1;
            }
            storm += 1;
        }

        for s in &sentinels {
            assert!(
                inner.contains_key(s),
                "sentinel {s} was evicted by storm in another shard",
            );
        }
    }
}

mod s3_fifo_tests {
    //! Internal-state assertions on `EntrySlot::queue`, `State::small_weight`,
    //! `State::ghost`. Black-box behaviour lives in `tests/cache_ops.rs`.

    use std::sync::Arc;

    use super::super::*;
    use super::SingleShardHasher;
    use crate::config::{CacheBuilder, DynPolicy};
    use crate::policy::{DefaultPolicy, Weight};

    type K = String;
    type V = String;

    fn build(max_weight: u64) -> Arc<Inner<K, V, DynPolicy<K, V>, SingleShardHasher>> {
        let policy: DynPolicy<K, V> = Arc::new(DefaultPolicy);
        let cfg = CacheBuilder::<K, V>::new()
            .max_weight(Weight::new(max_weight))
            .shards(2)
            .hasher(SingleShardHasher)
            .policy(policy)
            .build();
        Arc::new(Inner::new(cfg))
    }

    fn queue_of<P>(inner: &Inner<K, V, P, SingleShardHasher>, key: &K) -> Option<Queue>
    where
        P: Policy<K, V>,
    {
        let st = inner.shard_for(key).read();
        let id = entries_find(&st.entries, &inner.hasher, &st.slab, key)?;
        st.slab.get(id).map(|es| es.queue)
    }

    /// Cold Small eviction seeds ghost; reinsert before it ages out
    /// routes the entry to Main.
    #[test]
    fn ghost_hit_lands_in_main() {
        // shards=2 halves the per-shard cap; max_weight=4 gives cap=2
        // inside shard 0.
        let inner = build(4);
        inner.insert("a".into(), "va".into());
        inner.insert("b".into(), "vb".into());
        // Tail=a(freq=0) → evict + seed ghost(fp_a).
        inner.insert("c".into(), "vc".into());
        assert!(
            inner.get_served("a").is_none(),
            "a should have been evicted to seed ghost",
        );
        assert_eq!(
            queue_of(&inner, &"b".to_string()),
            Some(Queue::Small),
            "b stays in Small (it was the head, not the evicted tail)",
        );
        inner.insert("a".into(), "va2".into());
        assert_eq!(
            queue_of(&inner, &"a".to_string()),
            Some(Queue::Main),
            "ghost-hit on insert must route the entry into Main",
        );
    }

    /// `clear` must reset ghost (else next reinsert mis-routes to Main)
    /// and `small_weight` (else next reclaim mis-balances).
    #[test]
    fn clear_resets_ghost_and_small_weight() {
        let inner = build(4);
        inner.insert("a".into(), "v".into());
        inner.insert("b".into(), "v".into());
        // Force a Small eviction so `a` enters ghost.
        inner.insert("c".into(), "v".into());
        {
            let st = inner.shards.shards()[0].read();
            assert!(st.small_weight > 0);
        }

        inner.clear();

        {
            let st = inner.shards.shards()[0].read();
            assert_eq!(st.small_weight, 0, "small_weight must reset on clear");
            assert!(st.small.tail().is_none(), "small must be empty after clear");
            assert!(st.main.tail().is_none(), "main must be empty after clear");
        }
        // Reinsert `a` must land in Small (would land in Main if its
        // ghost fingerprint had survived `clear`).
        inner.insert("a".into(), "v".into());
        assert_eq!(
            queue_of(&inner, &"a".to_string()),
            Some(Queue::Small),
            "ghost must have been cleared so reinsert routes to Small",
        );
    }
}
