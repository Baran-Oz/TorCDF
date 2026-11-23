//! Fast chunk reader: locates chunks through HDF5, then reads and decodes
//! them outside the netCDF lock, in parallel.
//!
//! Used by [`NcFile::read_slice`] for chunked netCDF-4 variables whose
//! filters are only shuffle, deflate and Fletcher32. Its results equal the
//! netCDF-C path bit for bit; on any error the caller falls back to it.

pub(crate) mod decode;
pub(crate) mod h5;
#[cfg(test)]
mod tests;

use self::decode::{bytes_to_f64, unfilter};
use self::h5::{ChunkLoc, H5File, H5Var};
use crate::cf::Unpack;
use crate::{DType, FileFormat, NcFile, Slice, SliceRequest, Storage, VarInfo};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Bytes of the checksum Fletcher32 appends to a chunk.
const CHECKSUM_LEN: usize = 4;

/// The fast reader's state for one open file; everything is created on
/// first use.
#[derive(Default)]
pub(crate) struct FastState {
    /// None: the file cannot be opened a second time (library path only).
    file: OnceLock<Option<FastFile>>,
    /// Variable path to its decoder; None when not eligible.
    vars: Mutex<HashMap<String, Option<Arc<FastVar>>>>,
    /// Reads served by the fast path.
    reads: AtomicUsize,
}

struct FastFile {
    h5: H5File,
    /// Raw bytes are read from here with positional reads.
    data: File,
}

/// What the fast reader needs to decode one variable's chunks.
struct FastVar {
    h5: H5Var,
    dtype: DType,
    unpack: Unpack,
    /// Unpacked raw fill value: never-written chunks and records past the
    /// dataset's extent.
    fill: f32,
    /// Elements per chunk.
    chunk_len: usize,
}

/// The decode thread pool: one thread per core, named `torcdf-decode-{i}`.
/// None if the threads could not be started (reads then use netCDF-C).
pub(crate) fn pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("torcdf-decode-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

impl NcFile {
    /// Whether `read_slice` reads `var_path` with the fast reader: false for
    /// netCDF-3 files, unknown (e.g. virtual) paths, and variables that are
    /// not chunked numbers or use filters it cannot undo. Plain HDF5 files
    /// on the HDF5 backend are served too.
    pub fn fast_eligible(&self, var_path: &str) -> bool {
        self.fast_var(var_path).is_some()
    }

    /// The HDF5 chunk shape of `var_path` when it is [`fast_eligible`](Self::fast_eligible).
    pub(crate) fn fast_chunk(&self, var_path: &str) -> Option<Vec<usize>> {
        self.fast_var(var_path).map(|v| v.h5.chunk.clone())
    }

    /// Reads served by the fast reader so far (diagnostics).
    pub fn fast_reads(&self) -> usize {
        self.fast.reads.load(Ordering::Relaxed)
    }

    /// The cached decoder for `path`, opened on first use.
    fn fast_var(&self, path: &str) -> Option<Arc<FastVar>> {
        let lock = || self.fast.vars.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = lock().get(path) {
            return v.clone();
        }
        // Opened outside the map's lock; a racing open of the same variable
        // is harmless (the first stored result wins).
        let v = self.open_fast_var(path).map(Arc::new);
        lock().entry(path.to_string()).or_insert(v).clone()
    }

    fn open_fast_var(&self, path: &str) -> Option<FastVar> {
        if !matches!(
            self.meta.format,
            FileFormat::Netcdf4 | FileFormat::Netcdf4Classic | FileFormat::Hdf5
        ) {
            return None;
        }
        let hdf5 = self.backend_is_hdf5();
        let info = self.meta.var(path)?;
        let size = info.dtype.size()?;
        if !matches!(info.storage, Storage::Chunked { .. }) || info.shape.is_empty() {
            return None;
        }
        let file = self.fast_file()?;
        // Any HDF5 error (e.g. a dataset netCDF-4 stores under another
        // name) just means "not eligible".
        let h5 = H5Var::open(&file.h5, path).ok()??;
        let fits = h5.elem_size == size
            && h5.shape.len() == info.shape.len()
            && h5.shape.iter().zip(&info.shape).all(|(h, n)| h <= n);
        if !fits {
            return None;
        }
        let chunk_len = h5
            .chunk
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .filter(|n| n.checked_mul(size + CHECKSUM_LEN).is_some())?;
        // netCDF-C backend: never-written chunks read as HDF5's fill value,
        // records past the extent as netCDF's: a file from another tool may
        // set an HDF5 fill value that differs from `_FillValue`, and then
        // they would disagree. HDF5 backend: the reference is `H5Dread`, so
        // HDF5's fill (0 unless set) everywhere; the extent is the shape.
        let raw = if hdf5 { h5.fill.unwrap_or(0.0) } else { raw_fill(info) };
        if !hdf5
            && h5
                .fill
                .is_some_and(|h| h != raw && !(h.is_nan() && raw.is_nan()))
        {
            return None;
        }
        let unpack = Unpack::for_var(info);
        let fill = unpack.apply(raw);
        Some(FastVar {
            h5,
            dtype: info.dtype.clone(),
            unpack,
            fill,
            chunk_len,
        })
    }

    fn fast_file(&self) -> Option<&FastFile> {
        self.fast
            .file
            .get_or_init(|| {
                Some(FastFile {
                    h5: H5File::open(&self.c_path).ok()?,
                    data: File::open(&self.meta.path).ok()?,
                })
            })
            .as_ref()
    }

    /// `read_slice` through the fast reader. Any `Err` means "use the
    /// library path instead"; it is never shown to the user.
    pub(crate) fn read_fast(&self, r: &SliceRequest) -> Result<Slice, String> {
        let v = self.fast_var(&r.var).ok_or("not eligible")?;
        let file = self.fast_file().ok_or("no second file handle")?;
        let pool = pool().ok_or("no decode threads")?;
        let (_, shape, total) = self.check_request(r).map_err(|e| e.to_string())?;
        let axes: Vec<Vec<Run>> = (0..shape.len())
            .map(|d| {
                axis_runs(
                    r.start[d],
                    r.stride[d],
                    shape[d],
                    v.h5.chunk[d],
                    v.h5.shape[d],
                )
            })
            .collect();
        let tasks: usize = axes.iter().map(Vec::len).product();
        let out: Vec<AtomicU32> = (0..total)
            .map(|_| AtomicU32::new(f32::NAN.to_bits()))
            .collect();
        let job = Job {
            v: &v,
            file: &file.data,
            axes: &axes,
            out_strides: &row_major(&shape),
            out: &out,
        };
        pool.install(|| (0..tasks).into_par_iter().try_for_each(|t| job.chunk(t)))?;

        let data: Vec<f32> = out
            .into_iter()
            .map(|x| f32::from_bits(x.into_inner()))
            .collect();
        let valid_count = data.iter().filter(|x| !x.is_nan()).count();
        self.fast.reads.fetch_add(1, Ordering::Relaxed);
        Ok(Slice {
            shape,
            data,
            valid_count,
            vector: None,
        })
    }
}

/// The raw value netCDF-C returns where nothing was written: `_FillValue`,
/// else netCDF's default fill for the type.
fn raw_fill(info: &VarInfo) -> f64 {
    let attr = info
        .attrs
        .iter()
        .find(|a| a.name == "_FillValue")
        .and_then(|a| a.value.as_f64s())
        .and_then(|v| v.first().copied());
    attr.unwrap_or(match info.dtype {
        DType::I8 => -127.0,
        DType::U8 => 255.0,
        DType::I16 => -32767.0,
        DType::U16 => 65535.0,
        DType::I32 => -2147483647.0,
        DType::U32 => 4294967295.0,
        DType::I64 => -9223372036854775806i64 as f64,
        DType::U64 => 18446744073709551614u64 as f64,
        DType::F32 => f64::from(9.969_209_968_386_869e36_f64 as f32),
        DType::F64 => 9.969_209_968_386_869e36,
        DType::Char | DType::Str | DType::Other(_) => f64::NAN,
    })
}

/// Row-major element strides of an array of `shape`.
fn row_major(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// The selected indices of one request axis that fall in one chunk.
struct Run {
    /// The chunk's origin on this axis, in elements.
    origin: usize,
    /// (output index, offset within the chunk) of each selected index.
    picks: Vec<(usize, usize)>,
    /// How many leading picks lie inside the dataset's extent; the rest
    /// are past the records written so far and read as fill.
    stored: usize,
}

/// Groups the `count` selected indices `start + i * stride` of one axis by
/// chunk, skipping chunks that hold none of them.
fn axis_runs(start: usize, stride: usize, count: usize, chunk: usize, extent: usize) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for i in 0..count {
        let g = start + i * stride;
        let origin = g / chunk * chunk;
        if runs.last().is_none_or(|r| r.origin != origin) {
            runs.push(Run {
                origin,
                picks: Vec::new(),
                stored: 0,
            });
        }
        if let Some(run) = runs.last_mut() {
            run.picks.push((i, g - origin));
            run.stored += usize::from(g < extent);
        }
    }
    runs
}

/// One `read_fast` call, shared by its chunk tasks.
struct Job<'a> {
    v: &'a FastVar,
    file: &'a File,
    /// Per axis, the chunks the request touches.
    axes: &'a [Vec<Run>],
    out_strides: &'a [usize],
    /// The result, as f32 bits. Chunk tasks write disjoint elements.
    out: &'a [AtomicU32],
}

impl Job<'_> {
    /// Decodes task `t` (row-major over `axes`) and writes its elements.
    fn chunk(&self, t: usize) -> Result<(), String> {
        let mut runs: Vec<&Run> = Vec::with_capacity(self.axes.len());
        let mut rem = t;
        for axis in self.axes.iter().rev() {
            runs.push(axis.get(rem % axis.len()).ok_or("task out of range")?);
            rem /= axis.len();
        }
        runs.reverse();
        let Some((&last, outer)) = runs.split_last() else {
            return Err("scalar variables have no chunks".into());
        };
        let origin: Vec<usize> = runs.iter().map(|r| r.origin).collect();
        let values = if runs.iter().all(|r| r.stored > 0) {
            match self.v.h5.chunk_loc(&origin)? {
                Some(loc) => Some(self.decode(loc)?),
                None => None,
            }
        } else {
            None // wholly past the dataset's extent
        };

        let chunk_strides = row_major(&self.v.h5.chunk);
        let mut idx = vec![0usize; outer.len()];
        loop {
            let (mut obase, mut ibase, mut inside) = (0, 0, true);
            for (d, run) in outer.iter().enumerate() {
                let (o, i) = run.picks[idx[d]];
                obase += o * self.out_strides[d];
                ibase += i * chunk_strides[d];
                inside &= idx[d] < run.stored;
            }
            for (k, &(o, i)) in last.picks.iter().enumerate() {
                let x = match &values {
                    Some(vals) if inside && k < last.stored => {
                        let raw = vals.get(ibase + i).ok_or("chunk offset out of range")?;
                        self.v.unpack.apply(*raw)
                    }
                    _ => self.v.fill,
                };
                let dst = self.out.get(obase + o).ok_or("output index out of range")?;
                dst.store(x.to_bits(), Ordering::Relaxed);
            }
            // Advance the odometer over the outer axes.
            let mut d = outer.len();
            loop {
                if d == 0 {
                    return Ok(());
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < outer[d].picks.len() {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    /// Reads, unfilters and converts one stored chunk to raw `f64`s.
    fn decode(&self, loc: ChunkLoc) -> Result<Vec<f64>, String> {
        let v = self.v;
        let want = v.chunk_len * v.h5.elem_size;
        let max_len = want + CHECKSUM_LEN;
        // Deflate can grow incompressible data slightly; anything far
        // beyond that is not a chunk of this variable.
        let size = usize::try_from(loc.size)
            .ok()
            .filter(|&s| s <= max_len.saturating_add(max_len / 8).saturating_add(4096))
            .ok_or("stored chunk size is implausible")?;
        let mut raw = vec![0u8; size];
        read_exact_at(self.file, loc.offset, &mut raw).map_err(|e| e.to_string())?;
        let bytes = unfilter(raw, &v.h5.filters, loc.filter_mask, v.h5.elem_size, max_len)?;
        if bytes.len() != want {
            return Err(format!(
                "chunk decoded to {} bytes, not {want}",
                bytes.len()
            ));
        }
        let mut vals = vec![0.0; v.chunk_len];
        bytes_to_f64(&bytes, &v.dtype, v.h5.order, &mut vals);
        Ok(vals)
    }
}

/// Fills `buf` from `file` starting at byte `offset`, without a shared file
/// cursor, so several threads can read one handle at once.
pub(crate) fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let (mut buf, mut offset) = (buf, offset);
        while !buf.is_empty() {
            match file.seek_read(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}
