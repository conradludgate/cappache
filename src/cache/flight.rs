//! Miss-coalescing primitives.
//!
//! Every `try_get_with` / `try_get_with_blocking` follower for the same
//! `(key, F)` joins the same in-flight lane so we run `init` exactly
//! once per coalesced burst. The flight types ([`MissFlight`] for the
//! async path, [`SyncMissFlight`] for the Condvar-backed sync path)
//! both implement the cache-side [`Flight`] trait so cancellation is
//! uniform; per-lane bookkeeping ([`InflightLane`], [`LaneKeyRef`],
//! [`LeaderGuard`]) lives here too.
//!
//! Crate-private surface: every public type / fn is `pub(super)` and
//! used only from `super::inner`. The orchestration that ties these
//! together lives in `Inner::miss_probe` and friends.
use std::any::{Any, TypeId};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::{Arc, Condvar, Mutex};

use hashbrown::Equivalent;
use tokio::sync::watch;

use super::inner::{Inner, State};

/// Outcome published on a miss flight's settle channel. `Cancelled` is sent by
/// [`LeaderGuard::drop`] when the leader's `init` future was dropped or
/// panicked before producing a value; followers translate it into a probe
/// retry. Each lane is monomorphic on `(V, E)` so no `Any` downcast is needed
/// to read the result.
#[derive(Clone)]
pub(super) enum LaneOutcome<V, E> {
    Settled(Result<V, E>),
    Cancelled,
}

/// Common cache-side surface for an in-flight miss lane. The cache only
/// needs to signal cancellation through this trait; the leader/follower
/// publish-and-wait API is concrete on the typed flight (`publish_result` /
/// `await_settled` for async, `publish` / `wait` for sync) so user-facing
/// code keeps full type information for `(V, E)` or `V`.
pub(super) trait Flight: Any + Send + Sync {
    fn cancel(&self);
}

/// Construction sibling of [`Flight`]: gives `install_or_join_lane` a way
/// to mint a fresh lane without naming the concrete type at the call
/// site. Both `MissFlight<V, E>` and `SyncMissFlight<V>` impl it.
pub(super) trait NewFlight: Flight + Sized {
    fn fresh() -> Self;
}

/// Per-(key, error type) coalescing point for `try_get_with`. Followers with
/// the same `E` subscribe to the same flight; followers with a different `E`
/// install their own lane and run their own `init`. Mirrors moka's typed miss
/// lanes — see `State::inflight`.
pub(super) struct MissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    settle: watch::Sender<Option<LaneOutcome<V, E>>>,
}

impl<V, E> MissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn new() -> Self {
        let (tx, _) = watch::channel(None);
        Self { settle: tx }
    }

    pub(super) fn publish_result(&self, r: Result<V, E>) {
        let _ = self.settle.send(Some(LaneOutcome::Settled(r)));
    }

    /// `None` when the leader cancelled (sender published `Cancelled`, or
    /// dropped before publishing); callers re-enter the probe loop.
    pub(super) async fn await_settled(&self) -> Option<Result<V, E>> {
        let mut rx = self.settle.subscribe();
        loop {
            {
                let cur = rx.borrow();
                if let Some(o) = cur.as_ref() {
                    return match o.clone() {
                        LaneOutcome::Settled(r) => Some(r),
                        LaneOutcome::Cancelled => None,
                    };
                }
            }
            if rx.changed().await.is_err() {
                return None;
            }
        }
    }
}

impl<V, E> Flight for MissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn cancel(&self) {
        let _ = self.settle.send(Some(LaneOutcome::Cancelled));
    }
}

impl<V, E> NewFlight for MissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn fresh() -> Self {
        Self::new()
    }
}

/// Per-key Condvar-backed coalescing point for the sync
/// `try_get_with_blocking` path. Lives alongside the async
/// [`MissFlight<V, E>`] entries in `State::inflight`, distinguished by the
/// concrete `TypeId` of the flight type carried in the map's `(K, TypeId)`
/// key — a sync follower's `(key, TypeId<SyncMissFlight<V, E>>)` lookup
/// can never collide with an async lane (and vice versa). Sync lanes are
/// also keyed on `E` so two sync followers with different error types
/// install their own lanes, mirroring the async [`MissFlight<V, E>`]
/// behaviour.
pub(super) struct SyncMissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    state: Mutex<SyncFlightState<V, E>>,
    cond: Condvar,
}

enum SyncFlightState<V, E> {
    Pending,
    Settled(Result<V, E>),
    Cancelled,
}

impl<V, E> SyncMissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn new() -> Self {
        Self {
            state: Mutex::new(SyncFlightState::Pending),
            cond: Condvar::new(),
        }
    }

    pub(super) fn publish_result(&self, r: Result<V, E>) {
        let mut s = self.state.lock().expect("sync flight mutex poisoned");
        *s = SyncFlightState::Settled(r);
        self.cond.notify_all();
    }

    /// Returns `Some(res)` once the leader publishes an outcome. Returns
    /// `None` when the leader cancelled (init closure panicked or leader
    /// guard fired); the caller is then expected to re-enter the probe
    /// loop to either become leader or join a fresh follower.
    pub(super) fn wait(&self) -> Option<Result<V, E>> {
        let mut s = self.state.lock().expect("sync flight mutex poisoned");
        loop {
            match &*s {
                SyncFlightState::Pending => {
                    s = self.cond.wait(s).expect("sync flight mutex poisoned");
                }
                SyncFlightState::Settled(r) => return Some(r.clone()),
                SyncFlightState::Cancelled => return None,
            }
        }
    }
}

impl<V, E> Flight for SyncMissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn cancel(&self) {
        let mut s = self.state.lock().expect("sync flight mutex poisoned");
        if matches!(*s, SyncFlightState::Pending) {
            *s = SyncFlightState::Cancelled;
            self.cond.notify_all();
        }
    }
}

impl<V, E> NewFlight for SyncMissFlight<V, E>
where
    V: Clone + Send + Sync + 'static,
    E: Clone + Send + Sync + 'static,
{
    fn fresh() -> Self {
        Self::new()
    }
}

/// One in-flight miss flight. Type-erased through [`Any`] so the
/// per-shard `inflight` map can mix sync and async lanes uniformly; the
/// concrete flight type lives in the map's `(K, TypeId)` key, so a
/// follower's `downcast::<F>` is infallible by invariant (the `Option`
/// return is preserved purely as defensive API ergonomics). The
/// [`Flight`] vtable lets cancel-all paths fire `cancel()` without
/// knowing the concrete type statically.
pub(super) struct InflightLane {
    flight: Arc<dyn Any + Send + Sync>,
    /// Captured at construction so cancel goes through the [`Flight`]
    /// vtable without the call site needing to name the concrete type.
    cancel_fn: fn(&Arc<dyn Any + Send + Sync>),
}

fn dispatch_cancel<F: Flight>(any: &Arc<dyn Any + Send + Sync>) {
    if let Some(f) = (**any).downcast_ref::<F>() {
        f.cancel();
    }
}

impl InflightLane {
    fn new<F: Flight>(flight: Arc<F>) -> Self {
        Self {
            flight: flight as Arc<dyn Any + Send + Sync>,
            cancel_fn: dispatch_cancel::<F>,
        }
    }

    fn downcast<F: Flight>(&self) -> Option<Arc<F>> {
        Arc::clone(&self.flight).downcast::<F>().ok()
    }

    /// Used by `Cache::clear` (and any other cancel-all paths) to signal
    /// every follower waiting on this lane that the leader is gone, so
    /// they re-enter the probe loop instead of blocking forever.
    pub(super) fn cancel(&self) {
        (self.cancel_fn)(&self.flight);
    }
}

/// Borrowed lookup adapter for [`State::inflight`]. Pairs `&K` with the
/// concrete flight `TypeId` so `inflight.get` / `inflight.remove` probe
/// the `(K, TypeId)` map without cloning the key. Tuple `Hash` is just
/// `K::hash` then `TypeId::hash` with no length prefix, so the byte
/// stream produced here is identical to what an owned `(K, TypeId)`
/// would produce, satisfying the [`Equivalent`] hash invariant.
struct LaneKeyRef<'a, K> {
    key: &'a K,
    type_id: TypeId,
}

impl<'a, K> LaneKeyRef<'a, K> {
    fn new(key: &'a K, type_id: TypeId) -> Self {
        Self { key, type_id }
    }
}

impl<K: Hash> Hash for LaneKeyRef<'_, K> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.key.hash(h);
        self.type_id.hash(h);
    }
}

impl<K: Eq> Equivalent<(K, TypeId)> for LaneKeyRef<'_, K> {
    fn equivalent(&self, key: &(K, TypeId)) -> bool {
        self.key == &key.0 && self.type_id == key.1
    }
}

/// Outcome of a single probe attempt on a miss-coalescing flight type `F`.
/// `F` is `MissFlight<V, E>` for the async paths and `SyncMissFlight<V, E>`
/// for the sync path; both share the Hit/Wait/Leader trichotomy.
pub(super) enum MissStep<V, F>
where
    F: Flight,
    V: Clone + Send + Sync + 'static,
{
    Hit(V),
    Wait(Arc<F>),
    Leader { miss: Arc<F> },
}

/// Identity-aware lane removal. The `(K, TypeId)` key locates the row
/// uniquely, but we still gate the actual removal on `Arc` pointer
/// identity so a fresh lane installed for the same `(key, F)` between
/// leader-takes-lock and leader-runs-cleanup (cancelled then
/// re-entered) is not disturbed.
pub(super) fn remove_lane<K, V>(
    st: &mut State<K, V>,
    key: &K,
    type_id: TypeId,
    flight_ptr: *const (),
) where
    K: Eq + Hash,
{
    let lookup = LaneKeyRef::new(key, type_id);
    let stored_matches = st
        .inflight
        .get(&lookup)
        .is_some_and(|lane| Arc::as_ptr(&lane.flight).cast::<()>() == flight_ptr);
    if stored_matches {
        st.inflight.remove(&lookup);
    }
}

/// Looks up the lane at `(key, TypeId<F>)`. Returns `Wait(lane)` if a
/// follower can join the existing leader, otherwise installs a fresh
/// `F` lane and returns `Leader { miss }`. Type discrimination is by
/// `TypeId<F>` carried in the map key, so an async
/// (`MissFlight<V, E>`) and sync (`SyncMissFlight<V>`) lane for the
/// same key occupy distinct rows and never join each other.
pub(super) fn install_or_join_lane<K, V, F>(st: &mut State<K, V>, key: &K) -> MissStep<V, F>
where
    K: Eq + Hash + Clone + 'static,
    V: Clone + Send + Sync + 'static,
    F: Flight + NewFlight,
{
    let type_id = TypeId::of::<F>();
    if let Some(lane) = st.inflight.get(&LaneKeyRef::new(key, type_id))
        && let Some(arc) = lane.downcast::<F>()
    {
        return MissStep::Wait(arc);
    }

    let miss = Arc::new(F::fresh());
    st.inflight
        .insert((key.clone(), type_id), InflightLane::new(Arc::clone(&miss)));
    MissStep::Leader { miss }
}

/// RAII cleanup for any miss-coalescing leader. On drop (cancel or panic),
/// removes the lane (identified by Arc identity, so we won't disturb a
/// follower-installed replacement) and dispatches `Flight::cancel` so any
/// follower waiting on this exact flight retries instead of blocking
/// forever on a sender / Condvar that will never publish.
///
/// One generic guard covers both `try_get_with` (`F = MissFlight<V, E>`)
/// and `try_get_with_blocking` (`F = SyncMissFlight<V, E>`); the only
/// per-flight behaviour difference is the cancellation channel, which is
/// already abstracted by `Flight::cancel`.
pub(super) struct LeaderGuard<'a, F, K, V, P, S>
where
    F: Flight,
    K: Eq + Hash + 'static,
    V: Clone + Send + Sync + 'static,
    P: 'static,
    S: BuildHasher + 'static,
{
    pub(super) inner: &'a Inner<K, V, P, S>,
    pub(super) key: &'a K,
    pub(super) flight: Arc<F>,
    pub(super) armed: bool,
}

impl<F, K, V, P, S> Drop for LeaderGuard<'_, F, K, V, P, S>
where
    F: Flight,
    K: Eq + Hash + 'static,
    V: Clone + Send + Sync + 'static,
    P: 'static,
    S: BuildHasher + 'static,
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // `clashcore::RwLock` does not poison, so a panicking leader leaves
        // the shard in its last consistent state — we can take the write
        // lock here and clean up without the std::sync::Mutex `into_inner`
        // dance.
        let mut st = self.inner.shard_for(self.key).write();
        let flight_ptr = Arc::as_ptr(&self.flight).cast::<()>();
        remove_lane(&mut st, self.key, TypeId::of::<F>(), flight_ptr);
        drop(st);
        self.flight.cancel();
    }
}
