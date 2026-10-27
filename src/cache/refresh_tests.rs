//! Refresh ownership across entry invalidation, using captured jobs and no sleeps.
use crate::{Cache, CacheBuilder, DynPolicy, Executor, RevalidateScheduled, Weight};
use std::{
    future::{Future, poll_fn},
    hash::{BuildHasher, Hasher},
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::Mutex,
    task::{Context, Poll, Waker},
};

type Work = Pin<Box<dyn Future<Output = ()> + Send>>;
#[derive(Default)]
struct Capture(Mutex<Vec<Work>>);
impl<F: Future<Output = ()> + Send + 'static> Executor<F> for Capture {
    fn execute(&self, f: F) {
        self.0.lock().unwrap().push(Box::pin(f));
    }
}
impl Capture {
    fn take(&self) -> Work {
        self.0.lock().unwrap().pop().unwrap()
    }
}
fn drive(work: &mut Work) {
    assert!(
        work.as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_ready()
    );
}

#[derive(Clone, Copy)]
struct SingleShard;
struct SingleShardHash(std::collections::hash_map::DefaultHasher);
impl Hasher for SingleShardHash {
    fn finish(&self) -> u64 {
        self.0.finish() & ((1_u64 << 56) - 1)
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}
impl BuildHasher for SingleShard {
    type Hasher = SingleShardHash;
    fn build_hasher(&self) -> Self::Hasher {
        SingleShardHash(std::collections::hash_map::DefaultHasher::new())
    }
}

type TestCache = Cache<String, String, DynPolicy<String, String>, SingleShard>;
#[derive(Clone, Copy, Debug)]
enum Invalidation {
    Remove,
    Insert,
    Clear,
    Expire,
    Evict,
}
#[derive(Clone, Copy, Debug)]
enum Finish {
    Ok,
    Err,
    Drop,
}

fn cache(how: Invalidation) -> TestCache {
    Cache::new(
        CacheBuilder::new()
            .max_weight(Weight::new(if matches!(how, Invalidation::Evict) {
                2
            } else {
                10_000
            }))
            .shards(2)
            .hasher(SingleShard)
            .build(),
    )
}
fn submit(
    cache: &TestCache,
    ex: &Capture,
    result: Result<&'static str, ()>,
) -> RevalidateScheduled {
    cache
        .try_revalidate("k".into(), ex, move |_| {
            let mut first = true;
            poll_fn(move |_| {
                if std::mem::take(&mut first) {
                    Poll::Pending
                } else {
                    Poll::Ready(result.map(str::to_owned))
                }
            })
        })
        .unwrap()
}
fn replace(cache: &TestCache, how: Invalidation) {
    match how {
        Invalidation::Remove => {
            assert_eq!(cache.remove("k").as_deref(), Some("original"));
        }
        Invalidation::Insert => {}
        Invalidation::Clear => cache.clear(),
        Invalidation::Expire => {
            // Materialize an expired deadline under the shard lock so this
            // exercises real get-time eviction without depending on clock ticks.
            let key = "k".to_owned();
            let mut state = cache.inner.shard_for(&key).write();
            let id =
                super::inner::entries_find(&state.entries, &cache.inner.hasher, &state.slab, &key)
                    .unwrap();
            let old = state.slab.get(id).unwrap().entry.deadline;
            state.expiry_reschedule(id, old, 0);
            state.slab.get_mut(id).unwrap().entry.deadline = 0;
            drop(state);
            assert_eq!(cache.get("k"), None);
        }
        Invalidation::Evict => {
            cache.insert("other".into(), "evictor".into());
            assert_eq!(cache.get("k"), None);
        }
    }
    cache.insert("k".into(), "replacement".into());
}

fn check_stale_job(how: Invalidation, finish: Finish) {
    let cache = cache(how);
    let ex = Capture::default();
    cache.insert("k".into(), "original".into());
    assert_eq!(
        submit(
            &cache,
            &ex,
            if matches!(finish, Finish::Err) {
                Err(())
            } else {
                Ok("stale")
            }
        ),
        RevalidateScheduled::Submitted
    );
    let mut old = ex.take();
    replace(&cache, how);
    assert_eq!(
        submit(&cache, &ex, Ok("fresh")),
        RevalidateScheduled::Submitted
    );
    let mut new = ex.take();
    if !matches!(finish, Finish::Drop) {
        drive(&mut old);
    }
    drop(old);
    assert_eq!(
        cache.get("k").as_deref(),
        Some("replacement"),
        "{how:?} / {finish:?}"
    );
    assert_eq!(
        submit(&cache, &ex, Ok("duplicate")),
        RevalidateScheduled::SkippedInFlight,
        "{how:?} / {finish:?}"
    );
    drive(&mut new);
    assert_eq!(
        cache.get("k").as_deref(),
        Some("fresh"),
        "{how:?} / {finish:?}"
    );
    drop(new);
    assert_eq!(
        submit(&cache, &ex, Ok("next")),
        RevalidateScheduled::Submitted
    );
    drive(&mut ex.take());
    assert_eq!(cache.get("k").as_deref(), Some("next"));
}

#[test]
fn stale_completion_cannot_replace_a_new_refresh() {
    for how in [
        Invalidation::Remove,
        Invalidation::Insert,
        Invalidation::Clear,
        Invalidation::Expire,
        Invalidation::Evict,
    ] {
        check_stale_job(how, Finish::Ok);
    }
}
#[test]
fn stale_error_cannot_clear_a_new_refresh() {
    for how in [
        Invalidation::Remove,
        Invalidation::Insert,
        Invalidation::Clear,
        Invalidation::Expire,
        Invalidation::Evict,
    ] {
        check_stale_job(how, Finish::Err);
    }
}
#[test]
fn stale_cancellation_cannot_clear_a_new_refresh() {
    for how in [
        Invalidation::Remove,
        Invalidation::Insert,
        Invalidation::Clear,
        Invalidation::Expire,
        Invalidation::Evict,
    ] {
        check_stale_job(how, Finish::Drop);
    }
}

#[test]
fn initializer_guard_cannot_clear_a_replacement_refresh() {
    for panic in [false, true] {
        let cache = cache(Invalidation::Remove);
        let ex = Capture::default();
        cache.insert("k".into(), "original".into());
        let result = catch_unwind(AssertUnwindSafe(|| {
            cache.try_revalidate("k".into(), &ex, |_| {
                replace(&cache, Invalidation::Remove);
                assert_eq!(
                    submit(&cache, &ex, Ok("fresh")),
                    RevalidateScheduled::Submitted
                );
                assert!(!panic, "initializer panic");
                async { Err::<String, ()>(()) }
            })
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(matches!(result, Ok(Err(()))));
        }
        assert_eq!(
            submit(&cache, &ex, Ok("duplicate")),
            RevalidateScheduled::SkippedInFlight
        );
        drive(&mut ex.take());
        assert_eq!(cache.get("k").as_deref(), Some("fresh"));
    }
}

#[test]
fn completed_error_job_cannot_reacquire_the_same_slot() {
    let cache = cache(Invalidation::Remove);
    let ex = Capture::default();
    cache.insert("k".into(), "original".into());
    cache
        .try_revalidate("k".into(), &ex, |_| {
            let mut polls = 0;
            poll_fn(move |_| {
                polls += 1;
                match polls {
                    1 => Poll::Pending,
                    2 => Poll::Ready(Err(())),
                    _ => Poll::Ready(Ok("stale".to_owned())),
                }
            })
        })
        .unwrap();
    let mut old = ex.take();
    drive(&mut old);
    assert_eq!(
        submit(&cache, &ex, Ok("fresh")),
        RevalidateScheduled::Submitted
    );
    let mut new = ex.take();
    drive(&mut old);
    drop(old);
    assert_eq!(cache.get("k").as_deref(), Some("original"));
    assert_eq!(
        submit(&cache, &ex, Ok("duplicate")),
        RevalidateScheduled::SkippedInFlight
    );
    drive(&mut new);
    assert_eq!(cache.get("k").as_deref(), Some("fresh"));
}
