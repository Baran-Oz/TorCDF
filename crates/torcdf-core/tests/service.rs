use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use torcdf_core::fixtures;
use torcdf_core::{
    CoreError, NcFile, RequestHandle, SeriesBlock, SeriesHandle, SeriesRequest, Slice, SliceRequest,
    SliceService, series_blocks,
};

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

fn write_values(path: &std::path::Path, base: f32) {
    let mut f = netcdf::create(path).unwrap();
    f.add_dimension("x", 4).unwrap();
    let mut v = f.add_variable::<f32>("v", &["x"]).unwrap();
    v.put_values(&[base, base + 1.0, base + 2.0, base + 3.0], ..).unwrap();
}

#[test]
fn reopening_a_rewritten_file_does_not_serve_stale_cache() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("same.nc");
    let req = SliceRequest { var: "/v".into(), start: vec![0], count: vec![4], stride: vec![1] };
    let svc = SliceService::new(1 << 20, 1, noop());

    write_values(&path, 1.0);
    let f = Arc::new(NcFile::open(&path).unwrap());
    let old = wait(&mut svc.request(&f, req.clone())).unwrap();
    assert_eq!(old.data, vec![1.0, 2.0, 3.0, 4.0]);
    drop(f);

    write_values(&path, 100.0);
    let f = Arc::new(NcFile::open(&path).unwrap());
    let new = wait(&mut svc.request(&f, req.clone())).unwrap();
    assert_eq!(new.data, vec![100.0, 101.0, 102.0, 103.0]);
}
/// Polls `h` until it finishes, returning every block in delivery order.
fn wait_series(h: &mut SeriesHandle) -> (Vec<SeriesBlock>, Result<(), CoreError>) {
    let end = Instant::now() + TIMEOUT;
    let mut blocks = Vec::new();
    loop {
        // Read `finished` first: blocks delivered before it are then all taken below.
        let done = h.finished();
        blocks.extend(h.take_blocks());
        if let Some(r) = done {
            return (blocks, r);
        }
        assert!(Instant::now() < end, "timed out waiting for series");
        std::thread::yield_now();
    }
}

/// The blocks' values joined in order of `start`.
fn joined(blocks: &[SeriesBlock]) -> Vec<f32> {
    let mut sorted: Vec<&SeriesBlock> = blocks.iter().collect();
    sorted.sort_by_key(|b| b.start);
    sorted.iter().flat_map(|b| b.values.iter().copied()).collect()
}

fn same_values(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

fn uo_series() -> SeriesRequest {
    SeriesRequest { var: "/ocean/uo".into(), point: vec![0, 4, 7], along: 0, centre: 10 }
}

#[test]
fn series_equals_slice_and_comes_centre_first() {
    let (_d, f) = open_nc4();
    assert!(f.fast_eligible("/ocean/uo"), "this test covers the parallel path");
    let svc = SliceService::new(1 << 20, 1, noop());
    let req = uo_series();
    let mut h = svc.request_series(&f, req.clone());
    let (blocks, done) = wait_series(&mut h);
    done.unwrap();
    let order: Vec<(usize, usize)> = blocks.iter().map(|b| (b.start, b.values.len())).collect();
    assert_eq!(order, series_blocks(24, 10, 1));
    let want = f.read_slice(&req.full_request(24)).unwrap().data;
    assert!(same_values(&joined(&blocks), &want));
    assert!(blocks.iter().all(|b| b.vector.is_none()));
    assert_eq!(h.progress(), (24, 24));
    assert!(matches!(h.finished(), Some(Ok(()))));
}

#[test]
fn series_on_library_path() {
    let d = tempfile::tempdir().unwrap();
    let f = Arc::new(NcFile::open(&fixtures::write_classic(d.path())).unwrap());
    assert!(!f.fast_eligible("/t"));
    let svc = SliceService::new(1 << 20, 1, noop());
    // Point (lat 0, lon 0) holds a fill value at time 0.
    let req = SeriesRequest { var: "/t".into(), point: vec![0, 0, 0], along: 0, centre: 1 };
    let mut h = svc.request_series(&f, req.clone());
    let (blocks, done) = wait_series(&mut h);
    done.unwrap();
    let order: Vec<(usize, usize)> = blocks.iter().map(|b| (b.start, b.values.len())).collect();
    assert_eq!(order, series_blocks(3, 1, 256));
    assert_eq!(order, vec![(0, 3)]);
    let want = f.read_slice(&req.full_request(3)).unwrap().data;
    assert!(want[0].is_nan());
    assert!(same_values(&joined(&blocks), &want));
    assert_eq!(h.progress(), (3, 3));
    assert!(matches!(h.finished(), Some(Ok(()))));
}

#[test]
fn finished_series_is_cached() {
    let (_d, f) = open_nc4();
    let svc = SliceService::new(1 << 20, 1, noop());
    let req = uo_series();
    let (first, done) = wait_series(&mut svc.request_series(&f, req.clone()));
    done.unwrap();
    assert!(svc.cached(&f, &req.full_request(24)).is_some());
    let reads = svc.reads_performed();
    assert_eq!(reads, 24);

    let mut h = svc.request_series(&f, req);
    // A cache hit resolves at once, as one block.
    assert!(matches!(h.finished(), Some(Ok(()))));
    let blocks = h.take_blocks();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].start, 0);
    assert!(same_values(&blocks[0].values, &joined(&first)));
    assert_eq!(h.progress(), (24, 24));
    assert_eq!(svc.reads_performed(), reads);
}

#[test]
fn dropping_series_handle_stops_reads() {
    let d = tempfile::tempdir().unwrap();
    let f = Arc::new(NcFile::open(&fixtures::write_long_series(d.path())).unwrap());
    assert!(!f.fast_eligible("/v"));
    // The coordinator blocks in on_ready after its first block.
    let g = gate();
    let svc = SliceService::new(1 << 20, 1, g.on_ready.clone());
    let req = SeriesRequest { var: "/v".into(), point: vec![0], along: 0, centre: 2048 };
    let mut h = svc.request_series(&f, req);
    g.entered.recv_timeout(TIMEOUT).unwrap();
    let blocks = h.take_blocks();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].start, 2048);
    assert_eq!(blocks[0].values, (2048..2304u16).map(f32::from).collect::<Vec<_>>());
    assert_eq!(h.progress(), (256, 4096));
    assert!(h.finished().is_none());
    drop(h);
    g.release.send(()).unwrap();
    assert!(svc.reads_performed() <= 2, "reads: {}", svc.reads_performed());

    let (tx, rx) = bounded::<()>(1);
    std::thread::spawn(move || {
        drop(svc);
        tx.send(()).unwrap();
    });
    rx.recv_timeout(TIMEOUT).expect("service drop joins the series coordinator");
}

#[test]
fn series_error_is_reported() {
    let (_d, f) = open_nc4();
    let svc = SliceService::new(1 << 20, 1, noop());
    let bad = SeriesRequest { point: vec![0, 4], ..uo_series() };
    let (blocks, done) = wait_series(&mut svc.request_series(&f, bad));
    assert!(blocks.is_empty());
    assert!(matches!(done, Err(CoreError::Read { .. })));
    // The service still works afterwards.
    wait_series(&mut svc.request_series(&f, uo_series())).1.unwrap();
}

#[test]
fn fast_series_blocks_follow_chunks_on_any_axis() {
    let d = tempfile::tempdir().unwrap();
    let f = Arc::new(NcFile::open(&fixtures::write_fast_matrix(d.path())).unwrap());
    assert!(f.fast_eligible("/sd_f32"));
    let svc = SliceService::new(1 << 20, 1, noop());
    // Chunks are [2, 5, 6] over (time 7, y 13, x 17): each axis ends in a partial chunk.
    for (along, len, block, centre) in [(0, 7, 2, 5), (1, 13, 5, 0), (2, 17, 6, 16)] {
        let req = SeriesRequest { var: "/sd_f32".into(), point: vec![3, 4, 5], along, centre };
        let (blocks, done) = wait_series(&mut svc.request_series(&f, req.clone()));
        done.unwrap();
        let order: Vec<(usize, usize)> = blocks.iter().map(|b| (b.start, b.values.len())).collect();
        assert_eq!(order, series_blocks(len, centre, block), "along {along}");
        let want = f.read_slice_via_library(&req.full_request(len)).unwrap().data;
        assert!(same_values(&joined(&blocks), &want), "along {along}");
    }
}
