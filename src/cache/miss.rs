//! Miss-path orchestration for `Inner`.
//!
//! Hosts the per-iteration probe (`miss_probe`), the lock-protected
//! hit/expiry pre-check (`with_probe_for_hit`), the post-leader install
//! step (`finish_miss_ok`), and the public miss-coalescing entry points
//! `try_get_with` (async) and `try_get_with_blocking` (sync). The shard
//! lock is held only for the probe-and-install windows; user-supplied
//! init closures run with the lock released so the cache stays
//! responsive on other keys.
//!
//! Crate-private surface; consumed only from `super::Cache` via the
//! `Inner` impl methods exposed here.
use std::any::TypeId;
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;
use std::time::Instant;

use super::expiry::expiry_push_extend;
use super::flight::{
    Flight, LeaderGuard, MissFlight, MissStep, NewFlight, SyncMissFlight, install_or_join_lane,
    remove_lane,
};
use super::inner::{
    FREQ_CAP, Inner, Shard, State, deadline_after_write, entries_find, entry_is_served, evict_slot,
    install_ready, update_read_deadline,
};
use super::reclaim::{Reclaimed, reclaim};
use super::slab::CacheEntry;
use crate::lifecycle::LifecycleReason;
use crate::policy::Policy;

impl<K, V, P, S> Inner<K, V, P, S>
where
    K: Eq + Hash,
    P: Policy<K, V>,
    S: BuildHasher,
{
    /// Installs a fresh `Ready` slot for a leader's miss outcome.
    ///
    /// Returns the post-install reclaim victims plus the prior value (if a
    /// concurrent `Cache::insert` for the same key landed while the leader
    /// was running its `init`); callers fire `Updated` for that prior
    /// value once the shard lock is released, mirroring [`Self::insert`]
    /// and [`Self::apply_refresh_ok`].
    pub(super) fn finish_miss_ok(
        &self,
        st: &mut State<K, V>,
        key: &K,
        v: &V,
        now: Instant,
    ) -> (Reclaimed<K, V>, Option<V>)
    where
        K: Clone + 'static,
        V: Clone + 'static,
    {
        let deadline = deadline_after_write(&self.config.policy, key, v, self.start, now);
        let nw = self.config.policy.weight(key, v).get();
        let reclaimed = reclaim(self, st, nw, key, now);
        let displaced = entries_find(&st.entries, &self.hasher, &st.slab, key)
            .and_then(|id| evict_slot(st, &self.hasher, id).map(|(_, prev, _)| prev));
        install_ready(
            st,
            &self.hasher,
            key.clone(),
            CacheEntry {
                value: v.clone(),
                written_at: super::clock::floor_tick(self.start, now),
                deadline,
            },
            nw,
        );
        (reclaimed, displaced)
    }

    /// Acquires the shard write lock, drains an expired entry (if any),
    /// bumps the freq counter on a still-live entry + pushes a
    /// refreshed read-side expiry, then hands control to `then` with
    /// the held lock and the hit (if any). The closure runs in the
    /// same critical section so the install/join-lane decision sees no
    /// TOCTOU window between hit detection and lane install.
    ///
    /// Releases and retakes the shard lock around the `TimeExpired`
    /// lifecycle callback so observers may re-enter the cache. The
    /// closure pattern keeps the [`RwLockWriteGuard`] type private to
    /// this method so we do not need `lock_api` as a direct dep.
    fn with_probe_for_hit<R>(
        &self,
        shard: &Shard<K, V>,
        key: &K,
        then: impl FnOnce(&mut State<K, V>, Option<V>) -> R,
    ) -> R
    where
        K: Clone + 'static,
        V: Clone + 'static,
    {
        let mut st = shard.write();

        let now = Instant::now();
        let now_tick = super::clock::floor_tick(self.start, now);
        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let expired = st
                .slab
                .get(id)
                .is_some_and(|es| !entry_is_served(&es.entry, now_tick));
            if expired && let Some((_, v, _)) = evict_slot(&mut st, &self.hasher, id) {
                drop(st);
                self.notify_lifecycle(LifecycleReason::TimeExpired, key.clone(), Some(v));
                st = shard.write();
            }
        }

        let now = Instant::now();
        let now_tick = super::clock::floor_tick(self.start, now);
        let mut hit = None;
        if let Some(id) = entries_find(&st.entries, &self.hasher, &st.slab, key) {
            let still_served = st
                .slab
                .get(id)
                .is_some_and(|es| entry_is_served(&es.entry, now_tick));
            if still_served {
                let (value, update) = {
                    let es = st
                        .slab
                        .get_mut(id)
                        .expect("just confirmed Occupied above under our mutex");
                    es.freq = es.freq.saturating_add(1).min(FREQ_CAP);
                    let value = es.entry.value.clone();
                    let update = update_read_deadline(
                        key,
                        &mut es.entry,
                        &self.config.policy,
                        self.start,
                        now,
                    );
                    (value, update)
                };
                if let Some(dl) = update.new_deadline {
                    expiry_push_extend(&mut st, dl, id, update.superseded_finite);
                }
                hit = Some(value);
            }
        }

        then(&mut st, hit)
    }

    /// Single critical section for one miss iteration on flight type `F`:
    /// served hit, in-flight miss wait, or install miss leader. Avoids
    /// TOCTOU between separate `get_served` / miss checks / leader insert.
    /// Used by both [`Self::try_get_with`] (`F = MissFlight<V, E>`) and
    /// [`Self::try_get_with_blocking`] (`F = SyncMissFlight<V, E>`).
    fn miss_probe<F>(&self, key: &K) -> MissStep<V, F>
    where
        K: Clone + 'static,
        V: Clone + Send + Sync + 'static,
        F: Flight + NewFlight,
    {
        let shard = self.shard_for(key);
        self.with_probe_for_hit(shard, key, |st, hit| {
            if let Some(v) = hit {
                return MissStep::Hit(v);
            }
            install_or_join_lane::<K, V, F>(st, key)
        })
    }

    pub(crate) async fn try_get_with<F, Fut, E>(self: &Arc<Self>, key: K, init: F) -> Result<V, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<V, E>>,
        E: Clone + Send + Sync + 'static,
        K: Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let mut init = Some(init);
        loop {
            match self.miss_probe::<MissFlight<V, E>>(&key) {
                MissStep::Hit(v) => return Ok(v),
                MissStep::Wait(f) => {
                    if let Some(res) = f.await_settled().await {
                        return res;
                    }
                    // Leader cancelled — its lane has been cleared. Loop and
                    // either hit a fresh value or take leadership ourselves.
                }
                MissStep::Leader { miss } => {
                    let init = init
                        .take()
                        .expect("init consumed: only the leader arm consumes it");
                    let mut guard = LeaderGuard::<MissFlight<V, E>, K, V, P, S> {
                        inner: self.as_ref(),
                        key: &key,
                        flight: miss,
                        armed: true,
                    };
                    let fut = init();
                    let res = fut.await;
                    let mut st = guard.inner.shard_for(&key).write();
                    remove_lane(
                        &mut st,
                        &key,
                        TypeId::of::<MissFlight<V, E>>(),
                        Arc::as_ptr(&guard.flight).cast::<()>(),
                    );

                    let finish = if let Ok(v) = &res {
                        let now = Instant::now();
                        Some(self.finish_miss_ok(&mut st, &key, v, now))
                    } else {
                        None
                    };

                    // Publish under the lock so a follower that wakes and
                    // immediately calls `get(key)` cannot observe a state
                    // where the value was reported settled but is not yet
                    // installed in `entries`.
                    guard.flight.publish_result(res.clone());
                    guard.armed = false;
                    drop(guard);
                    drop(st);

                    if let Some((rec, displaced)) = finish {
                        if let Some(prev) = displaced {
                            self.notify_lifecycle(
                                LifecycleReason::Updated,
                                key.clone(),
                                Some(prev),
                            );
                        }
                        self.fire_reclaim(rec);
                    }
                    return res;
                }
            }
        }
    }

    /// Sync fallible miss-coalescing entry point. The user closure runs
    /// **without** the shard lock held, so the cache stays responsive on
    /// other keys while the leader produces a value. `Err` outcomes are
    /// published to followers but never installed in the slab — the next
    /// caller probes fresh.
    ///
    /// `key` is taken by value to match the async [`Self::try_get_with`]
    /// signature; internally it is borrowed everywhere except the owned
    /// clone that goes into the `inflight` map and the slab.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn try_get_with_blocking<F, E>(self: &Arc<Self>, key: K, init: F) -> Result<V, E>
    where
        K: Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
        E: Clone + Send + Sync + 'static,
        F: FnOnce(&K) -> Result<V, E>,
    {
        let mut init = Some(init);
        loop {
            match self.miss_probe::<SyncMissFlight<V, E>>(&key) {
                MissStep::Hit(v) => return Ok(v),
                MissStep::Wait(f) => {
                    if let Some(res) = f.wait() {
                        return res;
                    }
                    // Leader cancelled — its lane is gone; loop and either
                    // observe a fresh value or take leadership ourselves.
                }
                MissStep::Leader { miss } => {
                    let init = init
                        .take()
                        .expect("init consumed: only the leader arm consumes it");
                    let mut guard = LeaderGuard::<SyncMissFlight<V, E>, K, V, P, S> {
                        inner: self.as_ref(),
                        key: &key,
                        flight: miss,
                        armed: true,
                    };
                    let res = init(&key);
                    let mut st = guard.inner.shard_for(&key).write();
                    remove_lane(
                        &mut st,
                        &key,
                        TypeId::of::<SyncMissFlight<V, E>>(),
                        Arc::as_ptr(&guard.flight).cast::<()>(),
                    );

                    let finish = if let Ok(v) = &res {
                        let now = Instant::now();
                        Some(self.finish_miss_ok(&mut st, &key, v, now))
                    } else {
                        None
                    };

                    // Publish under the lock so a follower that wakes and
                    // immediately calls `get(key)` cannot observe a state
                    // where the value was reported settled but is not yet
                    // installed in `entries`.
                    guard.flight.publish_result(res.clone());
                    guard.armed = false;
                    drop(guard);
                    drop(st);

                    if let Some((rec, displaced)) = finish {
                        if let Some(prev) = displaced {
                            self.notify_lifecycle(
                                LifecycleReason::Updated,
                                key.clone(),
                                Some(prev),
                            );
                        }
                        self.fire_reclaim(rec);
                    }
                    return res;
                }
            }
        }
    }
}
