//! White-box tests over `Inner` internals: niche-packed sizes, shard
//! invariants, and S3-FIFO routing decisions. Black-box behaviours
//! observable through the public `Cache` API live in `tests/cache_ops.rs`.

/// `BuildHasher` that pins every key to shard 0 *while preserving per-key
/// fingerprints*. Wraps `DefaultHasher` (fixed initial state) and clears
/// the top 8 bits of `finish()` so [`clashcore::ClashCollection::determine_shard`]'s
/// `(h << 7) >> 63` always picks shard 0, but lower bits stay derived
/// from the actual key bytes. Used by white-box tests that want
/// single-shard determinism for FIFO / ghost / heap invariants without
/// collapsing every fingerprint to a single value.
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

    /// Lock the niche-packed [`SlotKey`] size. `slotmap::KeyData` packs an
    /// index `u32` plus a non-zero version `u32`, so `SlotKey` is one
    /// machine word and `Option<SlotKey>` is the same size (the version's
    /// niche absorbs the discriminant). If this fires, either `slotmap`
    /// changed its key layout or a custom key wrapper lost the niche.
    #[test]
    fn option_slot_key_is_eight_bytes() {
        assert_eq!(std::mem::size_of::<SlotKey>(), 8);
        assert_eq!(std::mem::size_of::<Option<SlotKey>>(), 8);
    }

    /// Lock the `EntrySlot` shape for a representative `<String, String>`
    /// cache against an upper bound rather than an exact byte count, so
    /// the test stays useful across platforms (`Instant` differs in size
    /// between macOS arm64 and `x86_64` / Linux). The load-bearing claim
    /// is "we did not accidentally double-bump a field width or lose the
    /// niche on `Option<SlotKey>`".
    #[test]
    fn entry_slot_size_within_budget() {
        // Two `String`s (48) + `CacheEntry<String>` worst-case 64 +
        // `weight` 8 + two `Option<SlotKey>` 16 (niche) +
        // `queue` 1 + `freq` 1 + `refresh_inflight` 1 (rounded to
        // alignment) gives a platform-portable upper bound. The +8 vs
        // the pre-S3-FIFO 144 is the tail padding around the three new
        // small fields.
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

    /// `shard_for` derives the shard purely from `&self.hasher` and
    /// `&self.shards`, both of which live on `Inner` and are therefore
    /// shared by every clone. Lock identity is the simplest cross-clone
    /// stability check: same key on two `Arc<Inner>` clones must hand
    /// back the *same* `Mutex` reference.
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

    /// The plan documents that with `shards > 1` the observed total
    /// weight may exceed `max_weight` by up to `shards - 1` units of
    /// rounding slack (each shard rounds its share up). Lock the bound
    /// with a stress insert so a future change to per-shard accounting
    /// can't quietly blow past it.
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

    /// Read-extending policies (e.g. `IdleTimeout`) push a fresh
    /// deadline tuple per read but cannot remove the old one cheaply, so
    /// without a rebuild trigger the heap grows without bound under
    /// repeated reads. The trigger fires once stale tuples exceed live
    /// slab size, capping the heap at one tuple per live entry plus the
    /// transient batch since the last rebuild.
    #[test]
    fn heap_stays_bounded_under_read_extending_policy() {
        use std::time::Duration;

        use crate::policy::{IdleTimeout, Weight};

        let policy: DynPolicy<K, V> = Arc::new(IdleTimeout::new(
            Duration::from_mins(1),
            crate::policy::DefaultPolicy,
        ));
        // Pin every key to one shard with `SingleShardHasher` so the
        // post-rebuild bound is observable without crawling all shards.
        // `shards = 2` halves the effective shard cap, so double
        // `max_weight` to keep shard 0 at cap 8 (matches the original
        // single-shard fixture).
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

        // Without the rebuild trigger, each read pushes a fresh heap
        // tuple and leaves the previous one stale; we'd see the heap
        // grow without bound. With the trigger, the heap stays at most
        // a constant multiple of the live slab size.
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

    /// Per-shard eviction must not leak across shards: filling one shard
    /// hard should not knock entries out of another shard. We exercise
    /// this without naming shards by inserting many keys that all map
    /// to a single shard (via `shard_for`-driven probing) and asserting
    /// every other shard's slab is untouched.
    #[test]
    fn eviction_stays_within_the_owning_shard() {
        let inner = build(4, 16);
        let target_shard = std::ptr::from_ref(inner.shard_for(&"anchor".to_string()));

        // Seed every other shard with a sentinel so we can confirm they
        // survive the storm aimed at `target_shard`.
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
    //! White-box S3-FIFO invariants that need to peek at internal
    //! state (`EntrySlot::queue`, `State::small_weight`,
    //! `State::ghost`). The black-box behaviours observable through
    //! the public `Cache` API live in `tests/cache_ops.rs::s3_fifo`.

    use std::sync::Arc;

    use super::super::*;
    use super::SingleShardHasher;
    use crate::config::{CacheBuilder, DynPolicy};
    use crate::policy::{DefaultPolicy, Weight};

    type K = String;
    type V = String;

    /// Pin every key into shard 0 via [`SingleShardHasher`] so per-shard
    /// FIFO ordering and ghost membership are observable without us
    /// having to coax the default hasher into giving us colocated keys.
    /// Uses `shards = 2` (the minimum the cache supports) — only shard 0
    /// receives any entries.
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

    /// A key evicted from Small with `freq <= 1` lands in the ghost
    /// set; reinserting it before its fingerprint ages out routes the
    /// new slot directly into Main.
    #[test]
    fn ghost_hit_lands_in_main() {
        // `shards = 2` halves the per-shard cap, so feed `max_weight = 4`
        // to land at cap 2 inside shard 0 (where every key resolves
        // courtesy of `SingleShardHasher`).
        let inner = build(4);
        // Fill Small to capacity with two cold (freq=0) keys.
        inner.insert("a".into(), "va".into());
        inner.insert("b".into(), "vb".into());
        // Push `a` out: small_max=0 means the next over-budget insert
        // drains Small first; tail=a (freq=0) → evict + ghost.insert(fp_a).
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
        // Reinsert `a` while its fingerprint is still in ghost. The
        // install path consumes ghost membership and routes a into Main.
        inner.insert("a".into(), "va2".into());
        assert_eq!(
            queue_of(&inner, &"a".to_string()),
            Some(Queue::Main),
            "ghost-hit on insert must route the entry into Main",
        );
    }

    /// `Cache::clear` empties the slab, the FIFOs, the ghost set, and
    /// resets `small_weight` so the next install starts from a pristine
    /// Small-budget accounting baseline. Without the ghost reset, a
    /// fingerprint that survived clear would mis-route a new key into
    /// Main on the next reinsert; without the `small_weight` reset, the
    /// next reclaim would mis-balance Small vs Main.
    #[test]
    fn clear_resets_ghost_and_small_weight() {
        let inner = build(4);
        inner.insert("a".into(), "v".into());
        inner.insert("b".into(), "v".into());
        // Force a Small eviction so `a`'s fingerprint enters ghost and
        // `small_weight` advances.
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
        // Ghost-cleared sanity: reinserting `a` must land it in Small,
        // not Main (which would happen if its fingerprint had survived
        // clear).
        inner.insert("a".into(), "v".into());
        assert_eq!(
            queue_of(&inner, &"a".to_string()),
            Some(Queue::Small),
            "ghost must have been cleared so reinsert routes to Small",
        );
    }
}
