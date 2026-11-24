//! Background slice reads with an LRU cache and cancellation.
//!
//! The UI thread never reads files: it calls [`SliceService::request`], keeps
//! the returned [`RequestHandle`], and polls it with [`RequestHandle::try_take`]
//! when `on_ready` wakes it. Dropping a handle cancels its job if no worker has
//! started it yet.
//!
//! [`SliceService::request_series`] reads a point series progressively on a
//! coordinator thread of its own (see [`crate::series`]); it shares the cache,
//! the read count and `on_ready`.

use crate::series::{self, SeriesHandle, SeriesRequest};
use crate::{CoreError, NcFile, Slice, SliceRequest};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, unbounded};
use lru::LruCache;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// Worker count used when the caller has no preference. netCDF-C serialises
/// reads, but a second worker overlaps unpacking with the next read.
pub const DEFAULT_WORKERS: usize = 2;

/// Default cache budget: a quarter of physical RAM (1 GiB if RAM cannot be determined).
pub fn default_cache_budget_bytes() -> usize {
    const FALLBACK: u64 = 4 << 30;
    let total = sysinfo::System::new()
        .map(|mut s| {
            s.refresh_memory();
            s.total_memory()
        })
        .unwrap_or(0);
    let total = if total == 0 { FALLBACK } else { total };
    usize::try_from(total / 4).unwrap_or(usize::MAX)
}

/// (NcFile::open_id, request): the id, not the path, so a file rewritten and
/// reopened at the same path never hits stale entries.
pub(crate) type Key = (u64, SliceRequest);
type Outcome = Result<Arc<Slice>, CoreError>;

/// LRU map plus a running byte total.
pub(crate) struct Cache {
    lru: LruCache<Key, Arc<Slice>>,
    bytes: usize,
    budget: usize,
}

impl Cache {
    pub(crate) fn get(&mut self, key: &Key) -> Option<Arc<Slice>> {
        self.lru.get(key).cloned()
    }

    /// Removes every entry of the file with this open id.
    fn forget(&mut self, open_id: u64) {
        let keys: Vec<Key> = self.lru.iter().filter(|(k, _)| k.0 == open_id).map(|(k, _)| k.clone()).collect();
        for k in keys {
            if let Some(old) = self.lru.pop(&k) {
                self.bytes -= old.byte_size();
            }
        }
    }

    /// Inserts `slice`, evicting least-recently-used entries until the total
    /// is within budget. A slice larger than the budget is not stored.
    pub(crate) fn insert(&mut self, key: Key, slice: Arc<Slice>) {
        let size = slice.byte_size();
        if size > self.budget {
            return;
        }
        if let Some(old) = self.lru.put(key, slice) {
            self.bytes -= old.byte_size();
        }
        self.bytes += size;
        while self.bytes > self.budget {
            match self.lru.pop_lru() {
                Some((_, old)) => self.bytes -= old.byte_size(),
                None => break,
            }
        }
    }
}

struct Job {
    file: Arc<NcFile>,
    req: SliceRequest,
    cancelled: Arc<AtomicBool>,
    reply: Sender<Outcome>,
}

pub(crate) struct Shared {
    cache: Mutex<Cache>,
    pub(crate) reads: AtomicUsize,
    on_ready: Arc<dyn Fn() + Send + Sync>,
    /// Set when the service drops: series coordinators stop reading.
    pub(crate) stopping: AtomicBool,
}

impl Shared {
    /// A poisoned lock only means another thread panicked; the cache itself
    /// is always left consistent, so keep using it.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Calls `on_ready`. A panicking callback must not kill the caller.
    pub(crate) fn ready(&self) {
        let _ = catch_unwind(AssertUnwindSafe(|| (self.on_ready)()));
    }
}

pub struct SliceService {
    shared: Arc<Shared>,
    jobs: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
    /// Series coordinator threads; finished ones are pruned on each request.
    series: Mutex<Vec<JoinHandle<()>>>,
}

impl SliceService {
    pub fn new(
        cache_budget_bytes: usize,
        workers: usize,
        on_ready: Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let shared = Arc::new(Shared {
            cache: Mutex::new(Cache {
                lru: LruCache::unbounded(),
                bytes: 0,
                budget: cache_budget_bytes,
            }),
            reads: AtomicUsize::new(0),
            on_ready,
            stopping: AtomicBool::new(false),
        });
        let (tx, rx) = unbounded::<Job>();
        let workers = (0..workers.max(1))
            .map(|i| {
                let (shared, rx) = (shared.clone(), rx.clone());
                std::thread::Builder::new()
                    .name(format!("torcdf-slice-{i}"))
                    .spawn(move || worker(&shared, &rx))
                    .expect("spawn slice worker")
            })
            .collect();
        SliceService { shared, jobs: Some(tx), workers, series: Mutex::new(Vec::new()) }
    }

    /// Starts reading a point series in blocks, the block holding
    /// `req.centre` first. A cached series (or an invalid request) resolves
    /// the handle at once; otherwise `on_ready` is called after each block.
    /// Dropping the handle stops reads that have not started.
    pub fn request_series(&self, file: &Arc<NcFile>, req: SeriesRequest) -> SeriesHandle {
        let (handle, thread) = series::start(&self.shared, file, req);
        if let Some(t) = thread {
            let mut threads = self.series.lock().unwrap_or_else(|e| e.into_inner());
            // Dropping a finished thread's handle just detaches it.
            threads.retain(|t| !t.is_finished());
            threads.push(t);
        }
        handle
    }

    /// Queues a read. A cache hit resolves the handle at once, without
    /// involving a worker. Dropping the handle cancels the job if it has not
    /// started.
    pub fn request(&self, file: &Arc<NcFile>, req: SliceRequest) -> RequestHandle {
        let (reply, rx) = bounded::<Outcome>(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let key = (file.open_id(), req.clone());
        if let Some(hit) = self.shared.lock().get(&key) {
            let _ = reply.send(Ok(hit));
        } else if let Some(jobs) = &self.jobs {
            let job = Job { file: file.clone(), req: req.clone(), cancelled: cancelled.clone(), reply };
            let _ = jobs.send(job);
        }
        RequestHandle { req, cancelled, rx, finished: false }
    }

    /// The cached slice for `req`, if present (marks it recently used).
    pub fn cached(&self, file: &NcFile, req: &SliceRequest) -> Option<Arc<Slice>> {
        self.shared.lock().get(&(file.open_id(), req.clone()))
    }

    /// Drops every cached slice and series of `file` (after a calculated
    /// variable was edited, so no value of the old formula is shown).
    pub fn forget_file(&self, file: &NcFile) {
        self.shared.lock().forget(file.open_id());
    }

    /// Number of reads actually performed against files (cache hits and
    /// cancelled jobs excluded).
    pub fn reads_performed(&self) -> usize {
        self.shared.reads.load(Ordering::SeqCst)
    }
}

impl Drop for SliceService {
    fn drop(&mut self) {
        // Series stop after their current blocks.
        self.shared.stopping.store(true, Ordering::SeqCst);
        let threads = std::mem::take(self.series.get_mut().unwrap_or_else(|e| e.into_inner()));
        for t in threads {
            let _ = t.join();
        }
        // Closing the channel ends each worker's loop once the queue drains
        // (queued jobs are skipped quickly if their handles are gone).
        self.jobs.take();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: &Shared, rx: &Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        if job.cancelled.load(Ordering::SeqCst) {
            continue;
        }
        let key = (job.file.open_id(), job.req.clone());
        // Another job for the same slice may have finished while this waited.
        let hit = shared.lock().get(&key);
        let outcome = match hit {
            Some(s) => Ok(s),
            None => {
                shared.reads.fetch_add(1, Ordering::SeqCst);
                // No lock is held during the read.
                let read = catch_unwind(AssertUnwindSafe(|| job.file.read_slice(&job.req)));
                match read {
                    Ok(Ok(slice)) => {
                        let slice = Arc::new(slice);
                        shared.lock().insert(key, slice.clone());
                        Ok(slice)
                    }
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(CoreError::Read {
                        var: job.req.var.clone(),
                        msg: "the read failed unexpectedly".into(),
                    }),
                }
            }
        };
        let _ = job.reply.send(outcome);
        shared.ready();
    }
}

/// Pending result of a [`SliceService::request`]. Dropping it cancels the
/// request if a worker has not started it.
pub struct RequestHandle {
    req: SliceRequest,
    cancelled: Arc<AtomicBool>,
    rx: Receiver<Outcome>,
    finished: bool,
}

impl RequestHandle {
    /// The result, once ready. Returns `Some` exactly once; `None` while the
    /// read is pending (and after the result was taken).
    pub fn try_take(&mut self) -> Option<Result<Arc<Slice>, CoreError>> {
        match self.rx.try_recv() {
            Ok(r) => {
                self.finished = true;
                Some(r)
            }
            Err(TryRecvError::Empty) => None,
            // The job was dropped without a reply (service shut down).
            Err(TryRecvError::Disconnected) if !self.finished => {
                self.finished = true;
                Some(Err(CoreError::Read {
                    var: self.req.var.clone(),
                    msg: "the read was abandoned".into(),
                }))
            }
            Err(TryRecvError::Disconnected) => None,
        }
    }

    pub fn req(&self) -> &SliceRequest {
        &self.req
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}
