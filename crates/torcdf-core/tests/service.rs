use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use torcdf_core::fixtures;
use torcdf_core::{CoreError, NcFile, RequestHandle, Slice, SliceRequest, SliceService};

const TIMEOUT: Duration = Duration::from_secs(10);

fn open_nc4() -> (tempfile::TempDir, Arc<NcFile>) {
    let d = tempfile::tempdir().unwrap();
    let f = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
    (d, Arc::new(f))
}

/// One time step of `/ocean/uo` (5 of 10 lats, starting at `lat0`).
fn step(t: usize, lat0: usize) -> SliceRequest {
    SliceRequest {
        var: "/ocean/uo".into(),
        start: vec![t, lat0, 0],
        count: vec![1, 5, 20],
        stride: vec![1, 1, 1],
    }
}

fn noop() -> Arc<dyn Fn() + Send + Sync> {
    Arc::new(|| {})
}

fn wait(h: &mut RequestHandle) -> Result<Arc<Slice>, CoreError> {
    let end = Instant::now() + TIMEOUT;
    loop {
        if let Some(r) = h.try_take() {
            return r;
        }
        assert!(Instant::now() < end, "timed out waiting for slice");
        std::thread::yield_now();
    }
}

/// An `on_ready` whose first call blocks until `release` is used. Gives the
/// test a worker that is provably busy, with no sleeps.
struct Gate {
    on_ready: Arc<dyn Fn() + Send + Sync>,
    entered: Receiver<()>,
    release: Sender<()>,
}

fn gate() -> Gate {
    let (entered_tx, entered) = bounded::<()>(1);
    let (release, release_rx) = unbounded::<()>();
    let first = Arc::new(AtomicBool::new(true));
    let on_ready: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
        if first.swap(false, Ordering::SeqCst) {
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(TIMEOUT).expect("gate never released");
        }
    });
    Gate { on_ready, entered, release }
}

#[test]
fn delivers_and_caches() {
    let (_d, f) = open_nc4();
    let svc = SliceService::new(1 << 20, 2, noop());
    let r = step(3, 0);
    let mut h = svc.request(&f, r.clone());
    assert_eq!(h.req(), &r);
    let s = wait(&mut h).unwrap();
    assert_eq!(s.shape, vec![1, 5, 20]);
    let c = svc.cached(&f, &r).expect("cached after delivery");
    assert!(Arc::ptr_eq(&s, &c));
    let reads = svc.reads_performed();
    let mut h2 = svc.request(&f, r);
    // A cache hit resolves on the very first poll.
    let s2 = h2.try_take().expect("hit resolves immediately").unwrap();
    assert!(Arc::ptr_eq(&s, &s2));
    assert_eq!(svc.reads_performed(), reads);
}

#[test]
fn on_ready_called_per_completed_job() {
    let (_d, f) = open_nc4();
    let (tx, rx) = unbounded::<()>();
    let svc = SliceService::new(1 << 20, 1, Arc::new(move || tx.send(()).unwrap()));
    let mut h = svc.request(&f, step(0, 0));
    rx.recv_timeout(TIMEOUT).expect("on_ready after job");
    assert!(h.try_take().is_some(), "result is available when on_ready fires");
}

#[test]
fn dropped_handle_is_cancelled() {
    let (_d, f) = open_nc4();
    let g = gate();
    let svc = SliceService::new(1 << 24, 1, g.on_ready.clone());
    // Occupy the only worker: it is blocked inside on_ready of this job.
    let mut first = svc.request(&f, step(0, 0));
    g.entered.recv_timeout(TIMEOUT).unwrap();
    let handles: Vec<_> = (0..24)
        .flat_map(|t| (0..3).map(move |l| (t, l)))
        .filter(|&(t, l)| (t, l) != (0, 0))
        .take(50)
        .map(|(t, l)| svc.request(&f, step(t, l)))
        .collect();
    assert_eq!(handles.len(), 50);
    drop(handles);
    g.release.send(()).unwrap();
    let start = Instant::now();
    let mut last = svc.request(&f, step(23, 4));
    wait(&mut last).unwrap();
    assert!(start.elapsed() < Duration::from_secs(1));
    wait(&mut first).unwrap();
    // The blocked job plus the final one; the 50 cancelled were skipped.
    assert_eq!(svc.reads_performed(), 2);
}

#[test]
fn stale_result_ignored() {
    let (_d, f) = open_nc4();
    let g = gate();
    let svc = SliceService::new(1 << 24, 1, g.on_ready.clone());
    let mut first = svc.request(&f, step(0, 0));
    g.entered.recv_timeout(TIMEOUT).unwrap();
    // App pattern: only the latest handle is held; older ones are replaced.
    let mut latest = svc.request(&f, step(1, 0));
    for t in 2..10 {
        latest = svc.request(&f, step(t, 0));
    }
    let want = step(9, 0);
    g.release.send(()).unwrap();
    let s = wait(&mut latest).unwrap();
    assert_eq!(latest.req(), &want);
    let direct = f.read_slice(&want).unwrap();
    assert_eq!(s.data.len(), direct.data.len());
    assert!(s.data.iter().zip(&direct.data).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
    wait(&mut first).unwrap();
}

#[test]
fn evicts_over_budget() {
    let (_d, f) = open_nc4();
    let one = 5 * 20 * 4;
    let svc = SliceService::new(2 * one, 1, noop());
    for t in 0..3 {
        wait(&mut svc.request(&f, step(t, 0))).unwrap();
    }
    assert!(svc.cached(&f, &step(0, 0)).is_none(), "oldest evicted");
    assert!(svc.cached(&f, &step(1, 0)).is_some());
    assert!(svc.cached(&f, &step(2, 0)).is_some());
}

#[test]
fn recently_used_entry_survives_eviction() {
    let (_d, f) = open_nc4();
    let one = 5 * 20 * 4;
    let svc = SliceService::new(2 * one, 1, noop());
    wait(&mut svc.request(&f, step(0, 0))).unwrap();
    wait(&mut svc.request(&f, step(1, 0))).unwrap();
    assert!(svc.cached(&f, &step(0, 0)).is_some()); // touch 0
    wait(&mut svc.request(&f, step(2, 0))).unwrap();
    assert!(svc.cached(&f, &step(0, 0)).is_some());
    assert!(svc.cached(&f, &step(1, 0)).is_none());
}

#[test]
fn oversized_slice_returned_but_not_cached() {
    let (_d, f) = open_nc4();
    let svc = SliceService::new(100, 1, noop());
    let r = step(0, 0);
    let s = wait(&mut svc.request(&f, r.clone())).unwrap();
    assert_eq!(s.data.len(), 100);
    assert!(svc.cached(&f, &r).is_none());
}

#[test]
fn read_error_is_delivered_not_panicked() {
    let (_d, f) = open_nc4();
    let svc = SliceService::new(1 << 20, 1, noop());
    let mut bad = step(0, 0);
    bad.start[0] = 1000;
    let r = wait(&mut svc.request(&f, bad.clone()));
    assert!(matches!(r, Err(CoreError::Read { .. })));
    assert!(svc.cached(&f, &bad).is_none());
    // The service still works afterwards.
    wait(&mut svc.request(&f, step(0, 0))).unwrap();
}

#[test]
fn dropping_service_stops_workers() {
    let (_d, f) = open_nc4();
    let svc = SliceService::new(1 << 20, 2, noop());
    wait(&mut svc.request(&f, step(0, 0))).unwrap();
    let (tx, rx) = bounded::<()>(1);
    std::thread::spawn(move || {
        drop(svc);
        tx.send(()).unwrap();
    });
    rx.recv_timeout(TIMEOUT).expect("service drop joins workers");
}

#[test]
fn same_request_on_two_files_is_cached_separately() {
    let (_d, f) = open_nc4();
    let d2 = tempfile::tempdir().unwrap();
    let f2 = Arc::new(NcFile::open(&fixtures::write_nc4(d2.path())).unwrap());
    let svc = SliceService::new(1 << 20, 1, noop());
    wait(&mut svc.request(&f, step(0, 0))).unwrap();
    assert!(svc.cached(&f2, &step(0, 0)).is_none());
}

#[test]
fn default_budget_is_a_quarter_of_ram() {
    let b = torcdf_core::default_cache_budget_bytes();
    assert!(b > 0);
    let mut sys = sysinfo::System::new().unwrap();
    sys.refresh_memory();
    assert_eq!(b as u64, sys.total_memory() / 4);
}
