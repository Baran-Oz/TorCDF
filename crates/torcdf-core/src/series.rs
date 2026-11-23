//! Point series: one variable read along one dimension at a fixed point,
//! delivered in blocks, the block around the slider position first.
//!
//! Each series runs on a coordinator thread owned by the
//! [`SliceService`](crate::SliceService). Fast-reader variables are read in
//! chunk-aligned blocks in parallel on the decode pool; everything else in
//! 256-step blocks on the coordinator itself. Either way blocks are handed
//! over strictly in [`series_blocks`] order.

use crate::service::Shared;
use crate::{CoreError, NcFile, Slice, SliceRequest, VectorParts, fast};
use crossbeam_channel::unbounded;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// Block length when the variable is read through netCDF-C.
const LIBRARY_BLOCK: usize = 256;

/// Fast-path blocks queued on the decode pool per pool thread: enough to
/// keep every thread busy, few enough that dropping the handle stops reads
/// promptly and slice reads are not queued behind a whole long series.
const IN_FLIGHT_PER_THREAD: usize = 2;

/// A point series: `var` along dimension `along`, at `point` elsewhere.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SeriesRequest {
    pub var: String,
    /// A full index into the variable; the entry at `along` is ignored.
    pub point: Vec<usize>,
    pub along: usize,
    /// Index along `along` to deliver first (the slider position).
    pub centre: usize,
}

impl SeriesRequest {
    /// The 1-D hyperslab of the whole series (count = len along `along`, 1
    /// elsewhere, stride 1).
    pub fn full_request(&self, len: usize) -> SliceRequest {
        let n = self.point.len();
        let (mut start, mut count) = (self.point.clone(), vec![1; n]);
        if self.along < n {
            start[self.along] = 0;
            count[self.along] = len;
        }
        SliceRequest { var: self.var.clone(), start, count, stride: vec![1; n] }
    }
}

/// Consecutive values of a series.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesBlock {
    /// Index along `along` of the first value.
    pub start: usize,
    /// `NaN` marks missing data.
    pub values: Vec<f32>,
    /// East/north components when the variable is a vector field.
    pub vector: Option<VectorParts>,
}

/// Blocks of `block` steps aligned to multiples of `block`, ordered: the
/// block holding `centre`, then alternately the next block after and the
/// next before, until both ends. Returns `(start, count)` pairs. A `centre`
/// past the end counts as the last step.
pub fn series_blocks(len: usize, centre: usize, block: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return Vec::new();
    }
    let block = block.max(1);
    let n = len.div_ceil(block);
    let span = |b: usize| (b * block, block.min(len - b * block));
    let c = centre.min(len - 1) / block;
    let mut out = Vec::with_capacity(n);
    out.push(span(c));
    let (mut after, mut before) = (c + 1, c);
    while after < n || before > 0 {
        if after < n {
            out.push(span(after));
            after += 1;
        }
        if before > 0 {
            before -= 1;
            out.push(span(before));
        }
    }
    out
}

/// What the coordinator has handed over so far.
#[derive(Default)]
struct Progress {
    /// Delivered and not yet taken.
    blocks: Vec<SeriesBlock>,
    delivered: usize,
    total: usize,
    done: Option<Result<(), CoreError>>,
}

type State = Arc<Mutex<Progress>>;

fn lock(state: &State) -> MutexGuard<'_, Progress> {
    state.lock().unwrap_or_else(|e| e.into_inner())
}

/// Pending result of [`SliceService::request_series`](crate::SliceService::request_series).
/// Dropping it stops reads that have not started; nothing is delivered
/// after that.
pub struct SeriesHandle {
    state: State,
    cancelled: Arc<AtomicBool>,
}

impl SeriesHandle {
    /// Blocks delivered since the last call, in [`series_blocks`] order.
    pub fn take_blocks(&mut self) -> Vec<SeriesBlock> {
        std::mem::take(&mut lock(&self.state).blocks)
    }

    /// (steps delivered, total steps).
    pub fn progress(&self) -> (usize, usize) {
        let p = lock(&self.state);
        (p.delivered, p.total)
    }

    /// `Some` once every block was delivered, or after the first error
    /// (later blocks are then not delivered).
    pub fn finished(&self) -> Option<Result<(), CoreError>> {
        lock(&self.state).done.clone()
    }
}

impl Drop for SeriesHandle {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

/// Starts `req`: resolves it at once from the cache or as an error, else
/// spawns its coordinator thread (returned for the service to join).
pub(crate) fn start(
    shared: &Arc<Shared>,
    file: &Arc<NcFile>,
    req: SeriesRequest,
) -> (SeriesHandle, Option<JoinHandle<()>>) {
    let state = State::default();
    let cancelled = Arc::new(AtomicBool::new(false));
    let handle = SeriesHandle { state: state.clone(), cancelled: cancelled.clone() };
    let full = match full_request_for(file, &req) {
        Ok(full) => full,
        Err(e) => {
            lock(&state).done = Some(Err(e));
            return (handle, None);
        }
    };
    let len = full.count[req.along];
    lock(&state).total = len;

    let key = (file.open_id(), full.clone());
    if let Some(hit) = shared.lock().get(&key) {
        let mut p = lock(&state);
        p.blocks.push(SeriesBlock { start: 0, values: hit.data.clone(), vector: hit.vector.clone() });
        p.delivered = len;
        p.done = Some(Ok(()));
        drop(p);
        return (handle, None);
    }

    let run = Arc::new(Run {
        shared: shared.clone(),
        file: file.clone(),
        full,
        along: req.along,
        cancelled,
        state: state.clone(),
    });
    let spawned = std::thread::Builder::new()
        .name("torcdf-series".into())
        .spawn(move || run.coordinate(req.centre));
    match spawned {
        Ok(t) => (handle, Some(t)),
        Err(e) => {
            lock(&state).done = Some(Err(CoreError::Read {
                var: req.var,
                msg: format!("cannot start the series read: {e}"),
            }));
            (handle, None)
        }
    }
}

/// The whole-series request for `req`, validated against the file's
/// metadata (no file access).
fn full_request_for(file: &NcFile, req: &SeriesRequest) -> Result<SliceRequest, CoreError> {
    let err = |msg: String| CoreError::Read { var: req.var.clone(), msg };
    // A variable, or a vector pair on its centre grid.
    let shape = match file.meta.var(&req.var) {
        Some(info) => info.shape.clone(),
        None => file
            .pair(&req.var)
            .map(|p| p.centre_shape.clone())
            .ok_or_else(|| err("no such variable".into()))?,
    };
    let len = shape.get(req.along).copied().ok_or_else(|| {
        err(format!("no dimension {} (the variable has {})", req.along, shape.len()))
    })?;
    let full = req.full_request(len);
    file.check_any_request(&full)?;
    Ok(full)
}

/// One series being read.
struct Run {
    shared: Arc<Shared>,
    file: Arc<NcFile>,
    full: SliceRequest,
    along: usize,
    cancelled: Arc<AtomicBool>,
    state: State,
}

/// The whole series assembled from delivered blocks (cached when complete).
struct Joined {
    values: Vec<f32>,
    /// Components, once a block carried them (vector pairs).
    vector: Option<VectorParts>,
}

/// A block's read: `None` when it was skipped because the series stopped.
type BlockRead = (usize, Option<Result<Slice, CoreError>>);

impl Run {
    /// The handle was dropped or the service is shutting down.
    fn stopped(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst) || self.shared.stopping.load(Ordering::SeqCst)
    }

    /// Reads `count` steps from `start` (one counted read).
    fn read_block(&self, (start, count): (usize, usize)) -> Result<Slice, CoreError> {
        let mut r = self.full.clone();
        r.start[self.along] = start;
        r.count[self.along] = count;
        self.shared.reads.fetch_add(1, Ordering::SeqCst);
        catch_unwind(AssertUnwindSafe(|| self.file.read_slice(&r))).unwrap_or_else(|_| {
            Err(CoreError::Read { var: r.var.clone(), msg: "the read failed unexpectedly".into() })
        })
    }

    /// The coordinator thread: reads every block, hands them over in order,
    /// and caches the completed series.
    fn coordinate(self: Arc<Self>, centre: usize) {
        let len = self.full.count[self.along];
        // Opening the fast reader touches the file, so it happens here.
        let chunk =
            self.file.fast_chunk(&self.full.var).and_then(|c| c.get(self.along).copied());
        let pool = chunk.and_then(|_| fast::pool());
        let blocks = series_blocks(len, centre, chunk.unwrap_or(LIBRARY_BLOCK));
        let window = pool.map_or(1, |p| p.current_num_threads().max(1) * IN_FLIGHT_PER_THREAD);

        let (tx, rx) = unbounded::<BlockRead>();
        // Finished reads waiting for an earlier block, by position in `blocks`.
        let mut ready: Vec<Option<Result<Slice, CoreError>>> =
            blocks.iter().map(|_| None).collect();
        let mut series = Joined { values: vec![f32::NAN; len], vector: None };
        let (mut queued, mut delivered, mut in_flight) = (0, 0, 0);
        let mut error = None;
        loop {
            // At most `window` blocks are read ahead of the next to deliver.
            let limit = blocks.len().min(delivered + window);
            while error.is_none() && !self.stopped() && queued < limit {
                let (run, tx, i, b) = (self.clone(), tx.clone(), queued, blocks[queued]);
                // Every task sends exactly one message, so `in_flight` drains.
                let task = move || {
                    let read = (!run.stopped()).then(|| run.read_block(b));
                    let _ = tx.send((i, read));
                };
                match pool {
                    Some(p) => p.spawn_fifo(task),
                    None => task(),
                }
                queued += 1;
                in_flight += 1;
            }
            if in_flight == 0 {
                break;
            }
            let Ok((i, read)) = rx.recv() else { break };
            in_flight -= 1;
            ready[i] = read;
            while error.is_none() && !self.stopped() {
                let Some(read) = ready.get_mut(delivered).and_then(Option::take) else { break };
                let last = delivered + 1 == blocks.len();
                match read.and_then(|s| self.deliver(blocks[delivered], s, &mut series, last)) {
                    Ok(()) => delivered += 1,
                    Err(e) => error = Some(e),
                }
            }
        }
        // A dropped handle needs nothing more; otherwise say why it ended early.
        if self.cancelled.load(Ordering::SeqCst) || delivered == blocks.len() {
            return;
        }
        let e = error.unwrap_or_else(|| CoreError::Read {
            var: self.full.var.clone(),
            msg: "the series read was abandoned".into(),
        });
        lock(&self.state).done = Some(Err(e));
        if !self.shared.stopping.load(Ordering::SeqCst) {
            self.shared.ready();
        }
    }

    /// Hands over one block, copying it into `series`. The last block also
    /// caches the whole series and finishes the handle.
    fn deliver(
        &self,
        (start, count): (usize, usize),
        slice: Slice,
        series: &mut Joined,
        last: bool,
    ) -> Result<(), CoreError> {
        let wrong = || CoreError::Read {
            var: self.full.var.clone(),
            msg: "a series block has the wrong length".into(),
        };
        let range = start..start + count;
        let dst = series
            .values
            .get_mut(range.clone())
            .filter(|d| d.len() == slice.data.len())
            .ok_or_else(wrong)?;
        dst.copy_from_slice(&slice.data);
        if let Some(parts) = &slice.vector {
            let len = series.values.len();
            let whole = series.vector.get_or_insert_with(|| VectorParts {
                east: vec![f32::NAN; len],
                north: vec![f32::NAN; len],
            });
            for (dst, src) in [(&mut whole.east, &parts.east), (&mut whole.north, &parts.north)] {
                dst.get_mut(range.clone())
                    .filter(|d| d.len() == src.len())
                    .ok_or_else(wrong)?
                    .copy_from_slice(src);
            }
        }
        if last {
            let whole = Slice {
                shape: self.full.count.clone(),
                valid_count: series.values.iter().filter(|v| !v.is_nan()).count(),
                data: series.values.clone(),
                vector: series.vector.clone(),
            };
            let key = (self.file.open_id(), self.full.clone());
            self.shared.lock().insert(key, Arc::new(whole));
        }
        {
            let mut p = lock(&self.state);
            p.blocks.push(SeriesBlock { start, values: slice.data, vector: slice.vector });
            p.delivered += count;
            if last {
                p.done = Some(Ok(()));
            }
        }
        self.shared.ready();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_blocks_order() {
        assert_eq!(series_blocks(10, 5, 2), vec![(4, 2), (6, 2), (2, 2), (8, 2), (0, 2)]);
        assert_eq!(series_blocks(10, 0, 4), vec![(0, 4), (4, 4), (8, 2)]);
        assert_eq!(series_blocks(1, 0, 256), vec![(0, 1)]);
    }

    #[test]
    fn series_blocks_edge_cases() {
        assert_eq!(series_blocks(0, 0, 4), vec![]);
        // A centre past the end counts as the last step.
        assert_eq!(series_blocks(5, 99, 2), vec![(4, 1), (2, 2), (0, 2)]);
        // A zero block length counts as 1.
        assert_eq!(series_blocks(2, 1, 0), vec![(1, 1), (0, 1)]);
    }

    #[test]
    fn full_request_spans_the_series_dimension() {
        let req = SeriesRequest { var: "/v".into(), point: vec![3, 4, 7], along: 1, centre: 2 };
        let full = req.full_request(10);
        assert_eq!(full.start, vec![3, 0, 7]);
        assert_eq!(full.count, vec![1, 10, 1]);
        assert_eq!(full.stride, vec![1, 1, 1]);
    }
}
