//! Animation along one slider dimension, with read-ahead through the
//! `SliceService`. Playback never skips a frame: when the next one is not
//! read yet it waits (and says "loading") instead.
//!
//! Read-ahead frames are ordinary service requests, so they share the one
//! cache budget; their count is capped so they hold at most `budget / 8`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use torcdf_core::{NcFile, RequestHandle, Slice, SliceRequest, SliceService};

pub const DEFAULT_FPS: f32 = 4.0;
pub const FPS_RANGE: std::ops::RangeInclusive<f32> = 1.0..=30.0;
/// Seconds of playback kept requested ahead.
const READAHEAD_SECS: f32 = 2.0;
/// Shown frames the effective rate is measured over.
const RATE_WINDOW: usize = 8;

/// `fps` within [`FPS_RANGE`] (the default when not a number).
fn clamp_fps(fps: f32) -> f32 {
    if fps.is_finite() { fps.clamp(*FPS_RANGE.start(), *FPS_RANGE.end()) } else { DEFAULT_FPS }
}

/// Frames to keep requested ahead: ceil(2 s * fps), at least 1, and at most
/// (budget / 8) / frame_bytes (at least 1).
pub fn readahead_len(fps: f32, frame_bytes: usize, budget: usize) -> usize {
    let wanted = ((READAHEAD_SECS * clamp_fps(fps)).ceil() as usize).max(1);
    let cap = ((budget / 8) / frame_bytes.max(1)).max(1);
    wanted.min(cap)
}

/// Bytes one frame of `reqs` holds once read (`components` values per cell:
/// 3 for a vector pair, magnitude plus both components).
pub fn frame_bytes(reqs: &[SliceRequest], components: usize) -> usize {
    reqs.iter()
        .map(|r| {
            let cells: usize =
                r.count.iter().zip(&r.stride).map(|(&c, &s)| c.div_ceil(s.max(1))).product();
            cells.saturating_mul(components * std::mem::size_of::<f32>())
        })
        .fold(0, usize::saturating_add)
}

/// One frame requested ahead.
struct Frame {
    index: usize,
    reqs: Vec<SliceRequest>,
    /// Reads still running; dropping them cancels the unstarted ones.
    handles: Vec<RequestHandle>,
    /// Reads done, held so cache eviction cannot lose a ready frame.
    /// A failed read is simply done: the frame is shown anyway, and the
    /// view's own read of it shows the error.
    slices: Vec<(SliceRequest, Arc<Slice>)>,
}

impl Frame {
    fn new(index: usize, reqs: Vec<SliceRequest>, service: &SliceService, file: &Arc<NcFile>) -> Frame {
        let handles = reqs.iter().map(|r| service.request(file, r.clone())).collect();
        let mut f = Frame { index, reqs, handles, slices: Vec::new() };
        f.poll();
        f
    }

    fn poll(&mut self) {
        let slices = &mut self.slices;
        self.handles.retain_mut(|h| match h.try_take() {
            None => true,
            Some(Ok(s)) => {
                slices.push((h.req().clone(), s));
                false
            }
            Some(Err(_)) => false,
        });
    }

    fn ready(&self) -> bool {
        self.handles.is_empty()
    }
}

/// What the view should do after a [`Playback::tick`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tick {
    /// Nothing is due.
    Idle,
    /// The next frame is due but not read yet.
    Wait,
    /// Show this index (its data is read, or its read failed).
    Show(usize),
    /// Past the end without looping: playback stopped.
    Stop,
}

pub struct Playback {
    pub playing: bool,
    /// The dimension animated.
    pub dim: usize,
    pub fps: f32,
    pub looping: bool,
    /// The next frame's data is not ready.
    pub waiting: bool,
    /// Measured over the last 8 shown frames.
    pub effective_fps: Option<f32>,
    ahead: VecDeque<Frame>,
    next_due: Instant,
    shown_at: VecDeque<Instant>,
    /// The slices of the frame last shown, for the view to take.
    ready: Vec<(SliceRequest, Arc<Slice>)>,
    #[cfg(test)]
    held: Option<usize>,
}

impl Playback {
    pub fn new(dim: usize) -> Playback {
        Playback {
            playing: false,
            dim,
            fps: DEFAULT_FPS,
            looping: false,
            waiting: false,
            effective_fps: None,
            ahead: VecDeque::new(),
            next_due: Instant::now(),
            shown_at: VecDeque::new(),
            ready: Vec::new(),
            #[cfg(test)]
            held: None,
        }
    }

    fn interval(&self) -> Duration {
        Duration::from_secs_f32(1.0 / clamp_fps(self.fps))
    }

    /// Starts playing; the first step is due one interval from `now`.
    pub fn start(&mut self, now: Instant) {
        self.playing = true;
        self.waiting = false;
        self.effective_fps = None;
        self.shown_at.clear();
        self.next_due = now + self.interval();
    }

    /// Stops playing and drops every read-ahead handle (cancelling the
    /// reads no worker has started).
    pub fn stop(&mut self) {
        self.playing = false;
        self.waiting = false;
        self.ahead.clear();
        self.ready.clear();
    }

    /// Time until the next step is due (zero when it is due now).
    pub fn due_in(&self, now: Instant) -> Duration {
        self.next_due.saturating_duration_since(now)
    }

    /// The slices of the frame just shown (empty when none were read).
    pub fn take_ready(&mut self) -> Vec<(SliceRequest, Arc<Slice>)> {
        std::mem::take(&mut self.ready)
    }

    /// `requests(index)` gives the SliceRequests of one frame (overview +
    /// current detail plan; the vector pair path covers both components).
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        now: Instant,
        current: usize,
        len: usize,
        frame_bytes: usize,
        budget: usize,
        service: &SliceService,
        file: &Arc<NcFile>,
        requests: &dyn Fn(usize) -> Vec<SliceRequest>,
    ) -> Tick {
        if !self.playing {
            return Tick::Idle;
        }
        let next = if current + 1 < len {
            current + 1
        } else if self.looping && len > 1 {
            0
        } else {
            if now >= self.next_due {
                self.stop();
                return Tick::Stop;
            }
            self.ahead.clear();
            return Tick::Idle;
        };
        self.fill(current, len, readahead_len(self.fps, frame_bytes, budget), service, file, requests);
        for f in &mut self.ahead {
            f.poll();
        }
        if now < self.next_due {
            return Tick::Idle;
        }
        let ready = self.ahead.front().is_some_and(|f| f.index == next && f.ready());
        #[cfg(test)]
        let ready = ready && self.held != Some(next);
        if !ready {
            self.waiting = true;
            return Tick::Wait;
        }
        let frame = self.ahead.pop_front().expect("checked above");
        self.ready = frame.slices;
        self.waiting = false;
        let interval = self.interval();
        // Far behind (after a wait): the next step is one interval from now.
        self.next_due = if now.saturating_duration_since(self.next_due) > interval {
            now + interval
        } else {
            self.next_due + interval
        };
        self.shown_at.push_back(now);
        if self.shown_at.len() > RATE_WINDOW {
            self.shown_at.pop_front();
        }
        if let (Some(first), Some(last)) = (self.shown_at.front(), self.shown_at.back()) {
            let secs = last.duration_since(*first).as_secs_f32();
            let n = self.shown_at.len();
            self.effective_fps = (n >= 2 && secs > 0.0).then(|| (n - 1) as f32 / secs);
        }
        Tick::Show(frame.index)
    }

    /// Makes `ahead` the frames after `current`, in order, `n` at most
    /// (wrapping when looping). Frames whose requests are unchanged keep
    /// their reads; the rest are dropped (cancelled) or requested anew.
    fn fill(
        &mut self,
        current: usize,
        len: usize,
        n: usize,
        service: &SliceService,
        file: &Arc<NcFile>,
        requests: &dyn Fn(usize) -> Vec<SliceRequest>,
    ) {
        let n = n.min(len.saturating_sub(1));
        let looping = self.looping;
        let wanted: Vec<usize> = (1..=n)
            .map_while(|k| {
                let i = current + k;
                if i < len {
                    Some(i)
                } else {
                    looping.then_some(i % len)
                }
            })
            .collect();
        let mut old = std::mem::take(&mut self.ahead);
        for index in wanted {
            let reqs = requests(index);
            let kept = old
                .iter()
                .position(|f| f.index == index && f.reqs == reqs)
                .and_then(|i| old.remove(i));
            self.ahead.push_back(kept.unwrap_or_else(|| Frame::new(index, reqs, service, file)));
        }
    }
}

#[cfg(test)]
impl Playback {
    /// Keeps frame `index` unready, whatever its reads say.
    pub fn hold(&mut self, index: usize) {
        self.held = Some(index);
    }

    pub fn release(&mut self) {
        self.held = None;
    }

    pub fn frame_ready(&self, index: usize) -> bool {
        self.ahead.iter().any(|f| f.index == index && f.ready())
    }

    pub fn ahead_len(&self) -> usize {
        self.ahead.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{Receiver, Sender, channel};
    use std::time::Duration;
    use torcdf_core::{NcFile, SliceRequest, SliceService, fixtures};

    const TIMEOUT: Duration = Duration::from_secs(20);

    fn uo(i: usize) -> SliceRequest {
        SliceRequest { var: "/ocean/uo".into(), start: vec![i, 0, 0], count: vec![1, 10, 20], stride: vec![1; 3] }
    }

    fn nc4() -> (tempfile::TempDir, Arc<NcFile>, SliceService) {
        let d = tempfile::tempdir().unwrap();
        let f = Arc::new(NcFile::open(&fixtures::write_nc4(d.path())).unwrap());
        (d, f, SliceService::new(1 << 24, 2, Arc::new(|| {})))
    }

    const FRAME: usize = 10 * 20 * 4;

    /// Ticks at `at` (before the first due time) until `index` is ready.
    #[allow(clippy::too_many_arguments)]
    fn wait_ready(
        p: &mut Playback,
        at: Instant,
        current: usize,
        len: usize,
        svc: &SliceService,
        f: &Arc<NcFile>,
        reqs: &dyn Fn(usize) -> Vec<SliceRequest>,
        index: usize,
    ) {
        let t0 = Instant::now();
        loop {
            assert!(matches!(p.tick(at, current, len, FRAME, 1 << 24, svc, f, reqs), Tick::Idle));
            if p.frame_ready(index) {
                return;
            }
            assert!(t0.elapsed() < TIMEOUT, "frame {index} never became ready");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn readahead_len_caps() {
        assert_eq!(readahead_len(4.0, 1000, 1 << 30), 8);
        assert_eq!(readahead_len(30.0, 1 << 20, 16 << 20), 2);
        assert_eq!(readahead_len(1.0, usize::MAX, 0), 1);
    }

    #[test]
    fn never_skips_waits_instead() {
        let (_d, f, svc) = nc4();
        let reqs = |i: usize| vec![uo(i)];
        let mut p = Playback::new(0);
        let t0 = Instant::now();
        p.start(t0);
        p.hold(2);
        wait_ready(&mut p, t0, 1, 24, &svc, &f, &reqs, 3);
        let step = Duration::from_secs_f32(1.0 / p.fps);
        for k in 1..=5 {
            let at = t0 + step * k;
            let t = p.tick(at, 1, 24, FRAME, 1 << 24, &svc, &f, &reqs);
            assert!(matches!(t, Tick::Wait), "tick {k} did not wait");
            assert!(p.waiting);
            assert!(p.frame_ready(3), "frame 3 is ready all along");
        }
        p.release();
        let t = p.tick(t0 + step * 6, 1, 24, FRAME, 1 << 24, &svc, &f, &reqs);
        assert!(matches!(t, Tick::Show(2)));
        assert!(!p.waiting);
        // The shown frame's slices are handed over, so eviction cannot lose them.
        let ready = p.take_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, uo(2));
    }

    #[test]
    fn failed_frame_is_shown_not_skipped() {
        let (_d, f, svc) = nc4();
        let reqs = |i: usize| {
            if i == 2 {
                vec![SliceRequest { var: "/ocean/missing".into(), ..uo(2) }]
            } else {
                vec![uo(i)]
            }
        };
        let mut p = Playback::new(0);
        let t0 = Instant::now();
        p.start(t0);
        wait_ready(&mut p, t0, 1, 24, &svc, &f, &reqs, 2);
        wait_ready(&mut p, t0, 1, 24, &svc, &f, &reqs, 3);
        let at = t0 + Duration::from_secs(1);
        let t = p.tick(at, 1, 24, FRAME, 1 << 24, &svc, &f, &reqs);
        assert!(matches!(t, Tick::Show(2)), "the failed frame was skipped");
    }

    #[test]
    fn loops_and_stops() {
        let (_d, f, svc) = nc4();
        let reqs = |i: usize| vec![uo(i)];
        let t0 = Instant::now();
        let at = t0 + Duration::from_secs(1);
        let mut p = Playback::new(0);
        p.looping = true;
        p.start(t0);
        wait_ready(&mut p, t0, 2, 3, &svc, &f, &reqs, 0);
        assert!(matches!(p.tick(at, 2, 3, FRAME, 1 << 24, &svc, &f, &reqs), Tick::Show(0)));

        let mut p = Playback::new(0);
        p.start(t0);
        assert!(matches!(p.tick(at, 2, 3, FRAME, 1 << 24, &svc, &f, &reqs), Tick::Stop));
    }

    /// An `on_ready` whose first call blocks until released: the one worker
    /// is provably busy while the test stops playback.
    fn gate() -> (Arc<dyn Fn() + Send + Sync>, Receiver<()>, Sender<()>) {
        let (entered_tx, entered) = channel::<()>();
        let (release, release_rx) = channel::<()>();
        let release_rx = std::sync::Mutex::new(release_rx);
        let first = AtomicBool::new(true);
        let on_ready: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            if first.swap(false, Ordering::SeqCst) {
                let _ = entered_tx.send(());
                let rx = release_rx.lock().unwrap();
                rx.recv_timeout(TIMEOUT).expect("gate never released");
            }
        });
        (on_ready, entered, release)
    }

    #[test]
    fn stop_drops_readahead() {
        let d = tempfile::tempdir().unwrap();
        let f = Arc::new(NcFile::open(&fixtures::write_long_series(d.path())).unwrap());
        let frame = |i: usize| SliceRequest { var: "/v".into(), start: vec![i], count: vec![1], stride: vec![1] };
        let reqs = |i: usize| vec![frame(i)];
        let (on_ready, entered, release) = gate();
        let workers = 1;
        let svc = SliceService::new(1 << 24, workers, on_ready);
        let mut p = Playback::new(0);
        let t0 = Instant::now();
        p.start(t0);
        assert!(matches!(p.tick(t0, 0, 4096, 4, 1 << 24, &svc, &f, &reqs), Tick::Idle));
        assert_eq!(p.ahead_len(), 8, "read-ahead was requested");
        // The worker is inside the first frame's on_ready; the other 7 are queued.
        entered.recv_timeout(TIMEOUT).expect("worker started");
        p.stop();
        assert_eq!(p.ahead_len(), 0);
        release.send(()).unwrap();
        // One worker takes jobs in order: once this one is done, every
        // read-ahead job before it was either read or skipped.
        let mut last = svc.request(&f, frame(4000));
        let t1 = Instant::now();
        while last.try_take().is_none() {
            assert!(t1.elapsed() < TIMEOUT, "queue never drained");
            std::thread::sleep(Duration::from_millis(1));
        }
        // The frame in progress plus the final read; the 7 dropped were skipped.
        let reads = svc.reads_performed();
        assert_eq!(reads, 2, "reads: {reads}");
        assert!(reads <= 2 + workers);
        drop(svc);
    }
}
