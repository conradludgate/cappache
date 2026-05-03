use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::future::poll_fn;
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Instant;

use tokio::sync::watch;

use crate::config::CacheConfig;
use crate::lifecycle::LifecycleReason;
use crate::policy::{Policy, Weight};

#[derive(Clone)]
pub(crate) enum FlightMsg<V> {
    Ok(V),
    Err(Arc<dyn Any + Send + Sync>),
}

impl<V: Clone> FlightMsg<V> {
    pub(crate) fn from_result<E: Send + Sync + 'static>(r: Result<V, E>) -> Self {
        match r {
            Ok(v) => FlightMsg::Ok(v),
            Err(e) => FlightMsg::Err(Arc::new(e)),
        }
    }

    pub(crate) fn into_result<E: Clone + Send + Sync + 'static>(self) -> Result<V, E> {
        match self {
            FlightMsg::Ok(v) => Ok(v),
            FlightMsg::Err(arc) => Err(Arc::downcast::<E>(arc)
                .expect("miss flight error type mismatch: E must match the stored miss error type")
                .as_ref()
                .clone()),
        }
    }
}

pub(crate) struct MissFlight<V> {
    pub(crate) settle: watch::Sender<Option<FlightMsg<V>>>,
    pub(crate) _hold: watch::Receiver<Option<FlightMsg<V>>>,
}

pub(crate) struct Inner<K: 'static, V: 'static> {
    pub(crate) config: CacheConfig<K, V>,
    pub(crate) state: Mutex<State<K, V>>,
}

pub(crate) struct State<K, V> {
    pub(crate) entries: HashMap<K, MapSlot<V>>,
    pub(crate) refresh_inflight: HashSet<K>,
    /// Sum of [`Policy::weight`](crate::policy::Policy::weight) over [`MapSlot::Ready`] entries only.
    pub(crate) total_weight: u64,
}

pub(crate) enum MapSlot<V> {
    Ready(CacheEntry<V>),
    Miss(Arc<MissFlight<V>>),
}

pub(crate) struct CacheEntry<V> {
    pub(crate) value: V,
    pub(crate) written_at: Instant,
    pub(crate) deadline: Option<Instant>,
}

pub(crate) enum MissStep<V> {
    Wait(Arc<MissFlight<V>>),
    Retry,
    Leader { miss: Arc<MissFlight<V>> },
}

async fn await_miss_settled<V, E>(flight: &MissFlight<V>) -> Result<V, E>
where
    V: Clone,
    E: Clone + Send + Sync + 'static,
{
    let mut rx = flight.settle.subscribe();
    loop {
        {
            let cur = rx.borrow();
            if let Some(m) = cur.as_ref() {
                return m.clone().into_result::<E>();
            }
        }
        rx.changed().await.expect(
            "miss settle watch: sender dropped before flight completed; this is a bug in clache",
        );
    }
}

impl<K, V> Inner<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    pub(crate) fn new(config: CacheConfig<K, V>) -> Self {
        Self {
            config,
            state: Mutex::new(State {
                entries: HashMap::new(),
                refresh_inflight: HashSet::new(),
                total_weight: 0,
            }),
        }
    }

    fn ready_entry_weight(&self, key: &K, e: &CacheEntry<V>) -> u64
    where
        K: 'static,
        V: 'static,
    {
        self.config.policy.weight(key, &e.value).get()
    }

    pub(crate) fn get_served(&self, key: &K) -> Option<V>
    where
        K: 'static,
        V: 'static,
    {
        let now = Instant::now();
        let mut st = self.state.lock().expect("cache mutex poisoned");
        let expired = match st.entries.get(key) {
            Some(MapSlot::Ready(e)) => !entry_is_served(e, now),
            _ => false,
        };
        if expired {
            if let Some(MapSlot::Ready(e)) = st.entries.remove(key) {
                let w = self.ready_entry_weight(key, &e);
                st.total_weight = st.total_weight.saturating_sub(w);
                let v = e.value.clone();
                drop(st);
                self.notify_lifecycle(LifecycleReason::TimeExpired, key.clone(), Some(v));
            }
            return None;
        }
        let slot = st.entries.get_mut(key)?;
        let MapSlot::Ready(e) = slot else {
            return None;
        };
        update_read_deadline(key, e, self.config.policy.as_ref(), now);
        Some(e.value.clone())
    }

    pub(crate) fn insert(&self, key: K, value: V) -> Option<V>
    where
        K: 'static,
        V: 'static,
    {
        let now = Instant::now();
        let new_w = self.config.policy.weight(&key, &value).get();
        let key_for_updated = key.clone();
        let mut st = self.state.lock().expect("cache mutex poisoned");

        let prev = match st.entries.remove(&key) {
            Some(MapSlot::Ready(old)) => {
                st.total_weight = st
                    .total_weight
                    .saturating_sub(self.ready_entry_weight(&key, &old));
                Some(old.value)
            }
            Some(MapSlot::Miss(_)) | None => None,
        };

        let pending = self.evict_until_fit(&mut st, Weight::new(new_w), &key);

        let deadline = deadline_after_write(self.config.policy.as_ref(), &key, &value, now);
        st.total_weight = st.total_weight.saturating_add(new_w);
        st.entries.insert(
            key,
            MapSlot::Ready(CacheEntry {
                value,
                written_at: now,
                deadline,
            }),
        );
        drop(st);
        if let Some(ref v) = prev {
            self.notify_lifecycle(LifecycleReason::Updated, key_for_updated, Some(v.clone()));
        }
        for (ek, ev) in pending {
            self.notify_lifecycle(LifecycleReason::EvictedSize, ek, Some(ev));
        }
        prev
    }

    pub(crate) fn remove(&self, key: &K) -> Option<V>
    where
        K: 'static,
        V: 'static,
    {
        let mut st = self.state.lock().expect("cache mutex poisoned");
        match st.entries.remove(key)? {
            MapSlot::Ready(e) => {
                st.total_weight = st
                    .total_weight
                    .saturating_sub(self.ready_entry_weight(key, &e));
                let v = e.value.clone();
                drop(st);
                self.notify_lifecycle(LifecycleReason::Removed, key.clone(), Some(v));
                Some(e.value)
            }
            MapSlot::Miss(_) => None,
        }
    }

    fn evict_until_fit(&self, st: &mut State<K, V>, add_w: Weight, skip_key: &K) -> Vec<(K, V)>
    where
        K: Clone + Eq + Hash + 'static,
        V: Clone + 'static,
    {
        let max = self.config.max_weight.get();
        let mut pending = Vec::new();
        while st.total_weight.saturating_add(add_w.get()) > max {
            let keys: Vec<K> = st
                .entries
                .iter()
                .filter_map(|(k, s)| match s {
                    MapSlot::Ready(_) => Some(k.clone()),
                    MapSlot::Miss(_) => None,
                })
                .collect();
            if keys.is_empty() {
                break;
            }
            let victim = keys.iter().position(|k| k != skip_key).unwrap_or(0);
            let vk = keys[victim].clone();
            if let Some(MapSlot::Ready(e)) = st.entries.remove(&vk) {
                let w = self.ready_entry_weight(&vk, &e);
                st.total_weight = st.total_weight.saturating_sub(w);
                pending.push((vk, e.value));
            } else {
                break;
            }
        }
        pending
    }

    fn notify_lifecycle(&self, reason: LifecycleReason, key: K, value: Option<V>)
    where
        K: 'static,
        V: 'static,
    {
        if let Some(l) = self.config.lifecycle_listener.as_ref() {
            l.on_event(reason, key, value);
        }
    }

    fn finish_miss_ok(&self, st: &mut State<K, V>, key: &K, v: &V, now: Instant) -> Vec<(K, V)>
    where
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let deadline = deadline_after_write(self.config.policy.as_ref(), key, v, now);
        let nw = self.config.policy.weight(key, v).get();
        let pending = self.evict_until_fit(st, Weight::new(nw), key);
        st.total_weight = st.total_weight.saturating_add(nw);
        st.entries.insert(
            key.clone(),
            MapSlot::Ready(CacheEntry {
                value: v.clone(),
                written_at: now,
                deadline,
            }),
        );
        pending
    }

    pub(crate) async fn try_get_with<F, Fut, E>(self: &Arc<Self>, key: K, init: F) -> Result<V, E>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<V, E>>,
        E: Clone + Send + Sync + 'static,
        K: Eq + Hash + Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        loop {
            if let Some(v) = self.get_served(&key) {
                return Ok(v);
            }

            if let Some(flight) = {
                let st = self.state.lock().expect("cache mutex poisoned");
                match st.entries.get(&key) {
                    Some(MapSlot::Miss(f)) => Some(Arc::clone(f)),
                    _ => None,
                }
            } {
                return await_miss_settled(&flight).await;
            }

            let step = {
                let mut st = self.state.lock().expect("cache mutex poisoned");
                if let Some(MapSlot::Ready(e)) = st.entries.get(&key) {
                    let now = Instant::now();
                    if entry_is_served(e, now) {
                        return Ok(e.value.clone());
                    }
                }
                match st.entries.get(&key) {
                    Some(MapSlot::Miss(f)) => MissStep::Wait(Arc::clone(f)),
                    Some(MapSlot::Ready(_)) => MissStep::Retry,
                    None => {
                        let (settle_tx, settle_rx) = watch::channel::<Option<FlightMsg<V>>>(None);
                        let miss = Arc::new(MissFlight {
                            settle: settle_tx,
                            _hold: settle_rx,
                        });
                        st.entries
                            .insert(key.clone(), MapSlot::Miss(Arc::clone(&miss)));
                        MissStep::Leader { miss }
                    }
                }
            };

            match step {
                MissStep::Wait(f) => return await_miss_settled(&f).await,
                MissStep::Retry => {}
                MissStep::Leader { miss } => {
                    let mut first_poll = true;
                    poll_fn(|cx| {
                        if first_poll {
                            first_poll = false;
                            cx.waker().wake_by_ref();
                            Poll::Pending
                        } else {
                            Poll::Ready(())
                        }
                    })
                    .await;
                    let fut = init();
                    let res = fut.await;
                    let mut st = self.state.lock().expect("cache mutex poisoned");
                    st.entries.remove(&key);
                    let _ = miss.settle.send(Some(FlightMsg::from_result(res.clone())));

                    if let Ok(v) = &res {
                        let now = Instant::now();
                        let pending = self.finish_miss_ok(&mut st, &key, v, now);
                        drop(st);
                        for (ek, ev) in pending {
                            self.notify_lifecycle(LifecycleReason::EvictedSize, ek, Some(ev));
                        }
                        return res;
                    }
                    drop(st);
                    return res;
                }
            }
        }
    }

    pub(crate) fn refresh_inflight_remove(&self, key: &K)
    where
        K: Eq + Hash,
    {
        self.state
            .lock()
            .expect("cache mutex poisoned")
            .refresh_inflight
            .remove(key);
    }

    pub(crate) fn apply_refresh_ok(&self, key: &K, v: V)
    where
        K: Eq + Hash + Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        let mut st = self.state.lock().expect("cache mutex poisoned");
        let displaced = match st.entries.remove(key) {
            Some(MapSlot::Ready(e)) => {
                st.total_weight = st
                    .total_weight
                    .saturating_sub(self.ready_entry_weight(key, &e));
                Some(e.value)
            }
            Some(MapSlot::Miss(_)) | None => None,
        };
        let now = Instant::now();
        let deadline = deadline_after_write(self.config.policy.as_ref(), key, &v, now);
        let w = self.config.policy.weight(key, &v);
        let nw = w.get();
        let pending = self.evict_until_fit(&mut st, w, key);
        st.total_weight = st.total_weight.saturating_add(nw);
        st.entries.insert(
            key.clone(),
            MapSlot::Ready(CacheEntry {
                value: v,
                written_at: now,
                deadline,
            }),
        );
        drop(st);
        drop(displaced);
        for (ek, ev) in pending {
            self.notify_lifecycle(LifecycleReason::EvictedSize, ek, Some(ev));
        }
    }
}

fn deadline_after_write<K, V>(
    policy: &(dyn Policy<K, V> + Send + Sync),
    key: &K,
    value: &V,
    now: Instant,
) -> Option<Instant>
where
    K: 'static,
    V: 'static,
{
    let d = policy.expire_after_write(key, value, now);
    if d >= crate::policy::TIME_UNBOUNDED / 2 {
        None
    } else {
        now.checked_add(d)
    }
}

/// Whether the entry may be returned for a read at `now`, using only materialized [`CacheEntry::deadline`].
///
/// [`Policy::expire_after_read`](crate::policy::Policy::expire_after_read) runs only after a live hit
/// in [`update_read_deadline`] to compute the next deadline.
fn entry_is_served<V>(e: &CacheEntry<V>, now: Instant) -> bool {
    !matches!(e.deadline, Some(dl) if now >= dl)
}

fn update_read_deadline<K, V>(
    key: &K,
    e: &mut CacheEntry<V>,
    policy: &(dyn Policy<K, V> + Send + Sync),
    now: Instant,
) where
    K: 'static,
    V: 'static,
{
    let time_remaining = e.deadline.map_or(crate::policy::TIME_UNBOUNDED, |dl| {
        dl.saturating_duration_since(now)
    });
    let d = policy.expire_after_read(key, &e.value, now, e.written_at, time_remaining);
    if d >= crate::policy::TIME_UNBOUNDED / 2 {
        e.deadline = None;
    } else {
        e.deadline = now.checked_add(d);
    }
}
