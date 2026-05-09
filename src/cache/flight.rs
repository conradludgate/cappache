//! Miss-coalescing primitives.
//!
//! Followers for the same `(key, F)` join the same in-flight lane so
//! `init` runs exactly once per coalesced burst. The async path uses
//! [`MissFlight`] (tokio `watch`); the blocking path uses
//! [`SyncMissFlight`] (`Mutex` + `Condvar`). Both impl [`Flight`] so
//! cancel-all paths (e.g. `Cache::clear`) can fire `cancel()` without
//! knowing the concrete `(V, E)` specialisation.

use std::any::{Any, TypeId};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::{Arc, Condvar, Mutex};

use hashbrown::Equivalent;
use tokio::sync::watch;

use super::inner::{Inner, State};

/// `Cancelled` is sent when the leader's `init` was dropped or panicked
/// before producing a value; followers translate it into a probe retry.
#[derive(Clone)]
pub(super) enum LaneOutcome<V, E> {
    Settled(Result<V, E>),
    Cancelled,
}

pub(super) trait Flight: Any + Send + Sync {
    fn cancel(&self);
}

pub(super) trait NewFlight: Flight + Sized {
    fn fresh() -> Self;
}

/// Async miss-coalescing lane. Followers with a different `E` install
/// their own lane: the `(K, TypeId)` map key keys on the flight type.
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

    /// `None` means the leader cancelled; callers re-enter the probe loop.
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

/// Sync (`Condvar`-backed) counterpart to [`MissFlight`]. Distinguished
/// from the async lanes by `TypeId` in the `(K, TypeId)` map key, so
/// sync and async followers never join each other.
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

    /// `None` means the leader cancelled; caller re-enters the probe loop.
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

/// Type-erased lane stored in `State::inflight`. The concrete flight
/// type is recovered by `Any::downcast_ref`; `cancel_fn` lets the
/// cancel-all paths fire `Flight::cancel` without naming the type.
pub(super) struct InflightLane {
    flight: Arc<dyn Any + Send + Sync>,
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

    pub(super) fn cancel(&self) {
        (self.cancel_fn)(&self.flight);
    }
}

/// Borrowed `(K, TypeId)` lookup adapter for `State::inflight`. Tuple
/// `Hash` is `K::hash` then `TypeId::hash` (no length prefix), which
/// matches what an owned `(K, TypeId)` produces — required by
/// [`Equivalent`].
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

pub(super) enum MissStep<V, F>
where
    F: Flight,
    V: Clone + Send + Sync + 'static,
{
    Hit(V),
    Wait(Arc<F>),
    Leader { miss: Arc<F> },
}

/// Removes the lane only when the stored `Arc` matches `flight_ptr`. A
/// fresh lane installed for the same `(key, F)` between leader-cancel
/// and leader-cleanup must not be disturbed.
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

/// RAII cleanup for a miss-coalescing leader. On drop (cancel or
/// panic), removes the lane (Arc-identity check) and fires
/// `Flight::cancel` so followers retry instead of blocking forever.
pub(super) struct LeaderGuard<'a, F, K, V, P, S>
where
    F: Flight,
    K: Eq + Hash + Send + 'static,
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
    K: Eq + Hash + Send + 'static,
    V: Clone + Send + Sync + 'static,
    P: 'static,
    S: BuildHasher + 'static,
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // `clashcore::RwLock` doesn't poison, so we can take the write
        // lock here even after a panic.
        let mut st = self.inner.shard_for(self.key).write();
        let flight_ptr = Arc::as_ptr(&self.flight).cast::<()>();
        remove_lane(&mut st, self.key, TypeId::of::<F>(), flight_ptr);
        drop(st);
        self.flight.cancel();
    }
}

#[cfg(feature = "typesize")]
mod typesize_impls {
    use typesize::TypeSize;

    use super::InflightLane;

    // Shallow only: the `Arc<dyn Any>` payload is type-erased; we can't
    // deeply walk it without reconstructing every `(V, E)` flight type.
    impl TypeSize for InflightLane {
        fn extra_size(&self) -> usize {
            0
        }
    }
}
