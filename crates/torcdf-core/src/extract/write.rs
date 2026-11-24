//! Writes a [`Plan`] as a new netCDF-4 file on the raw netCDF-C API.
//!
//! Every netCDF-C call is made under `libnetcdf_lock`, one call at a time,
//! so reads elsewhere in the app keep running during a long save.

use super::plan::parent_of;
use super::{DimKey, Plan, PlanVar};
use crate::open::{RawHandle, nc_message};
use crate::{NcFile, SliceRequest};
use netcdf_sys as sys;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Largest piece, in bytes, read and written by one netCDF call.
const BATCH_BYTES: usize = 64 * 1024 * 1024;
/// Missing values of calculated variables (netCDF's default float fill).
const CALC_FILL: f32 = 9.96921e36;

fn locked<R>(f: impl FnOnce() -> R) -> R {
    let _g = sys::libnetcdf_lock.lock();
    f()
}

/// Runs one netCDF call under the lock; its error as text.
fn call(what: &str, f: impl FnOnce() -> c_int) -> Result<(), String> {
    let rc = locked(f);
    if rc == sys::NC_NOERR { Ok(()) } else { Err(format!("{what}: {}", nc_message(rc))) }
}

fn cstr(s: &str) -> Result<CString, String> {
    CString::new(s).map_err(|_| format!("name {s:?} holds a NUL character"))
}

/// The output file while it is written: closed, and deleted unless kept,
/// whatever way the save ends.
struct OutFile {
    ncid: c_int,
    open: bool,
    tmp: PathBuf,
    keep: bool,
}

impl OutFile {
    fn close(&mut self) -> Result<(), String> {
        self.open = false;
        call("closing the new file", || unsafe { sys::nc_close(self.ncid) })
    }
}

impl Drop for OutFile {
    fn drop(&mut self) {
        if self.open {
            locked(|| unsafe { sys::nc_close(self.ncid) });
        }
        if !self.keep {
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// A save running on its own thread. It holds its own `Arc<NcFile>` and the
/// plan's calculated variables, so closing the file or editing a formula
/// does not change or stop it.
pub struct ExtractJob {
    state: Arc<JobState>,
}

#[derive(Default)]
struct JobState {
    progress: AtomicU32,
    cancel: AtomicBool,
    outcome: Mutex<Option<Result<PathBuf, String>>>,
    skipped: Mutex<Vec<String>>,
}

impl ExtractJob {
    pub fn start(file: Arc<NcFile>, plan: Plan, out: PathBuf, history_line: String) -> ExtractJob {
        let state = Arc::new(JobState::default());
        let s = state.clone();
        std::thread::spawn(move || {
            let progress = |f: f32| {
                s.progress.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
                !s.cancel.load(Ordering::Relaxed)
            };
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                write_extract(&file, &plan, &out, &history_line, &progress)
            }));
            let outcome = match run {
                Ok(Ok(skipped)) => {
                    s.progress.store(1.0f32.to_bits(), Ordering::Relaxed);
                    *s.skipped.lock().unwrap_or_else(|e| e.into_inner()) = skipped;
                    Ok(out)
                }
                Ok(Err(e)) => Err(e),
                Err(_) => Err("internal error while saving".into()),
            };
            *s.outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);
        });
        ExtractJob { state }
    }

    /// 0..1.
    pub fn progress(&self) -> f32 {
        f32::from_bits(self.state.progress.load(Ordering::Relaxed))
    }

    pub fn cancel(&self) {
        self.state.cancel.store(true, Ordering::Relaxed);
    }

    /// The outcome, once done.
    pub fn finished(&self) -> Option<Result<PathBuf, String>> {
        self.state.outcome.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// What the finished save could not copy ("name (why)").
    pub fn skipped(&self) -> Vec<String> {
        self.state.skipped.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Writes `plan` to `out`. `progress` gets 0..1 after each piece; `false`
/// cancels. Returns what could not be copied ("name (why)").
pub fn write_extract(
    file: &NcFile,
    plan: &Plan,
    out: &Path,
    history_line: &str,
    progress: &dyn Fn(f32) -> bool,
) -> Result<Vec<String>, String> {
    write_with_budget(file, plan, out, history_line, progress, BATCH_BYTES)
}

/// `write_extract` with pieces of at most `budget` bytes.
fn write_with_budget(
    file: &NcFile,
    plan: &Plan,
    out: &Path,
    history_line: &str,
    progress: &dyn Fn(f32) -> bool,
    budget: usize,
) -> Result<Vec<String>, String> {
    if !file.is_netcdf() {
        return Err("Extract works on netCDF files".into());
    }
    if let (Ok(a), Ok(b)) = (std::fs::canonicalize(out), std::fs::canonicalize(&file.meta.path))
        && a == b
    {
        return Err("the file being read cannot be replaced".into());
    }
    let input = RawHandle::open(&file.c_path)?;
    let (c_tmp, tmp) = temp_paths(out)?;
    let c = cstr(c_tmp.to_str().ok_or("the folder name cannot be passed to netCDF")?)?;
    let mut ncid = 0;
    call("creating the new file", || unsafe {
        sys::nc_create(c.as_ptr(), sys::NC_NETCDF4 | sys::NC_CLOBBER, &mut ncid)
    })?;
    let mut outf = OutFile { ncid, open: true, tmp: tmp.clone(), keep: false };
    let mut skipped = plan.skipped.clone();
    let mut w = Writer { input: input.0, out: ncid, gin: HashMap::new(), gout: HashMap::new(), skipped: &mut skipped };

    w.groups(plan)?;
    let history = file
        .meta
        .root
        .attrs
        .iter()
        .find(|a| a.name == "history")
        .and_then(|a| a.value.as_str())
        .map_or_else(|| history_line.to_string(), |h| format!("{h}\n{history_line}"));
    w.put_text(ncid, sys::NC_GLOBAL, "history", &history)?;
    let dimids = w.dims(plan)?;
    let mut ids = Vec::new();
    for v in &plan.vars {
        ids.push(w.define(v, &dimids)?);
    }
    call("ending definitions", || unsafe { sys::nc_enddef(ncid) })?;

    let total = plan.bytes.max(1) as f64;
    let mut done = 0u64;
    for (v, id) in plan.vars.iter().zip(&ids) {
        for (off, cnt) in slabs(&v.count, id.elem, budget) {
            let n: usize = cnt.iter().product();
            if n == 0 {
                continue;
            }
            let start: Vec<usize> = v.start.iter().zip(&off).map(|(s, o)| s + o).collect();
            w.copy(file, v, id, &start, &off, &cnt, n)?;
            done += (n * id.elem) as u64;
            if !progress((done as f64 / total).min(1.0) as f32) {
                return Err("cancelled".into());
            }
        }
    }
    outf.close()?;
    drop(input);
    std::fs::rename(&tmp, out).map_err(|e| format!("could not save {}: {e}", out.display()))?;
    outf.keep = true;
    Ok(skipped)
}

/// (path given to netCDF, real path) of the temporary file next to `out`.
/// The name is ASCII; a non-ASCII folder is given in its short (8.3) form
/// when it has one, as reading does.
fn temp_paths(out: &Path) -> Result<(PathBuf, PathBuf), String> {
    let dir = match out.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let name = out.file_name().ok_or("no file name given")?.to_string_lossy();
    let ascii: String = name.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c } else { '_' }).collect();
    let tmp_name = format!("{ascii}.tmp-torcdf");
    let c_dir = if dir.to_str().is_some_and(str::is_ascii) {
        dir.clone()
    } else {
        #[cfg(windows)]
        let short = crate::open::short_path(&dir);
        #[cfg(not(windows))]
        let short: Option<PathBuf> = None;
        short.unwrap_or_else(|| dir.clone())
    };
    Ok((c_dir.join(&tmp_name), dir.join(&tmp_name)))
}

/// Pieces of a `count`-shaped block of at most `budget` bytes (at least one
/// value each): (offset, count) per piece, in row-major order.
fn slabs(count: &[usize], elem: usize, budget: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = count.len();
    if n == 0 {
        return vec![(vec![], vec![])];
    }
    // The outermost dimension k whose inner block fits the budget.
    let inner = |k: usize| count[k + 1..].iter().product::<usize>().max(1) * elem.max(1);
    let k = (0..n).find(|&k| inner(k) <= budget).unwrap_or(n - 1);
    let step = (budget / inner(k)).max(1);
    let mut out = Vec::new();
    let mut idx = vec![0usize; k];
    loop {
        let mut s = 0;
        while s < count[k] {
            let len = step.min(count[k] - s);
            let mut off = idx.clone();
            off.push(s);
            off.extend(std::iter::repeat_n(0, n - k - 1));
            let mut cnt = vec![1; k];
            cnt.push(len);
            cnt.extend_from_slice(&count[k + 1..]);
            out.push((off, cnt));
            s += len;
        }
        // Next outer index.
        let mut d = k;
        loop {
            if d == 0 {
                return out;
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < count[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// Ids of one variable in both files.
struct VarIds {
    gin: c_int,
    vin: c_int,
    gout: c_int,
    vout: c_int,
    xtype: sys::nc_type,
    /// Bytes per value in memory (a pointer for strings).
    elem: usize,
}

struct Writer<'a> {
    input: c_int,
    out: c_int,
    gin: HashMap<String, c_int>,
    gout: HashMap<String, c_int>,
    skipped: &'a mut Vec<String>,
}

impl Writer<'_> {
    /// Creates every group a written variable or dimension lives in (and
    /// their parents) and copies their attributes.
    fn groups(&mut self, plan: &Plan) -> Result<(), String> {
        let mut paths: Vec<String> = vec!["/".into()];
        let mut add = |mut p: &str| {
            while p != "/" {
                if !paths.iter().any(|x| x == p) {
                    paths.push(p.to_string());
                }
                p = parent_of(p);
            }
        };
        for v in &plan.vars {
            add(parent_of(&v.path));
        }
        for (k, _, _) in &plan.dims {
            add(&k.group);
        }
        paths.sort_by_key(|p| if p == "/" { 0 } else { p.matches('/').count() });
        for p in paths {
            let gin = if p == "/" {
                self.input
            } else {
                let c = cstr(&p)?;
                let mut g = 0;
                call(&format!("group {p}"), || unsafe { sys::nc_inq_grp_full_ncid(self.input, c.as_ptr(), &mut g) })?;
                g
            };
            let gout = if p == "/" {
                self.out
            } else {
                let parent = self.gout[parent_of(&p)];
                let c = cstr(p.rsplit('/').next().unwrap_or(&p))?;
                let mut g = 0;
                call(&format!("group {p}"), || unsafe { sys::nc_def_grp(parent, c.as_ptr(), &mut g) })?;
                g
            };
            let label = if p == "/" { String::new() } else { format!("{p}/") };
            self.copy_atts(gin, sys::NC_GLOBAL, gout, sys::NC_GLOBAL, &label)?;
            self.gin.insert(p.clone(), gin);
            self.gout.insert(p, gout);
        }
        Ok(())
    }

    /// Copies every attribute; one the library cannot copy is listed in `skipped`.
    fn copy_atts(&mut self, gin: c_int, vin: c_int, gout: c_int, vout: c_int, label: &str) -> Result<(), String> {
        let mut n = 0;
        call("attributes", || unsafe {
            if vin == sys::NC_GLOBAL { sys::nc_inq_natts(gin, &mut n) } else { sys::nc_inq_varnatts(gin, vin, &mut n) }
        })?;
        for i in 0..n {
            let mut buf = vec![0 as c_char; sys::NC_MAX_NAME as usize + 1];
            call("attribute name", || unsafe { sys::nc_inq_attname(gin, vin, i, buf.as_mut_ptr()) })?;
            let name = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
            let rc = locked(|| unsafe { sys::nc_copy_att(gin, vin, name.as_ptr(), gout, vout) });
            if rc != sys::NC_NOERR {
                self.skipped.push(format!("{label}{} ({})", name.to_string_lossy(), nc_message(rc)));
            }
        }
        Ok(())
    }

    fn put_text(&self, g: c_int, v: c_int, name: &str, text: &str) -> Result<(), String> {
        let c = cstr(name)?;
        call(&format!("attribute {name}"), || unsafe {
            sys::nc_put_att_text(g, v, c.as_ptr(), text.len(), text.as_ptr().cast())
        })
    }

    fn dims(&mut self, plan: &Plan) -> Result<HashMap<DimKey, c_int>, String> {
        let mut ids = HashMap::new();
        for (k, len, unlimited) in &plan.dims {
            let g = self.gout[&k.group];
            let c = cstr(&k.name)?;
            let mut id = 0;
            let len = if *unlimited { sys::NC_UNLIMITED } else { *len };
            call(&format!("dimension {}", k.name), || unsafe { sys::nc_def_dim(g, c.as_ptr(), len, &mut id) })?;
            ids.insert(k.clone(), id);
        }
        Ok(ids)
    }

    /// Defines `v` in the new file with its type, chunking, compression and attributes.
    fn define(&mut self, v: &PlanVar, dimids: &HashMap<DimKey, c_int>) -> Result<VarIds, String> {
        let group = parent_of(&v.path);
        let name = v.path.rsplit('/').next().unwrap_or(&v.path);
        let cname = cstr(name)?;
        let (gin, gout) = (self.gin[group], self.gout[group]);
        let dims: Vec<c_int> = v.dims.iter().map(|k| dimids[k]).collect();
        let what = format!("variable {}", v.path);
        let (mut vin, mut xtype) = (-1, sys::NC_FLOAT);
        if v.derived.is_none() {
            call(&what, || unsafe { sys::nc_inq_varid(gin, cname.as_ptr(), &mut vin) })?;
            call(&what, || unsafe {
                sys::nc_inq_var(gin, vin, std::ptr::null_mut(), &mut xtype, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
            })?;
        }
        let elem = if xtype == sys::NC_STRING {
            std::mem::size_of::<*mut c_char>()
        } else {
            let mut size = 0;
            call(&what, || unsafe { sys::nc_inq_type(gin, xtype, std::ptr::null_mut(), &mut size) })?;
            size
        };
        let mut vout = 0;
        call(&what, || unsafe {
            sys::nc_def_var(gout, cname.as_ptr(), xtype, dims.len() as c_int, dims.as_ptr(), &mut vout)
        })?;

        // Chunking and compression: failures only cost size, never values.
        let clip = |chunk: &[usize]| -> Vec<usize> { chunk.iter().zip(&v.count).map(|(c, n)| (*c).min(*n).max(1)).collect() };
        let (chunk, deflate) = match &v.derived {
            Some(d) => match &d.info.storage {
                crate::Storage::Chunked { chunk, compressed } => (Some(clip(chunk)), compressed.then_some((1, 4))),
                _ => (None, None),
            },
            None if !dims.is_empty() => {
                let mut kind = 0;
                let mut chunk = vec![0usize; dims.len()];
                let rc = locked(|| unsafe { sys::nc_inq_var_chunking(gin, vin, &mut kind, chunk.as_mut_ptr()) });
                if rc == sys::NC_NOERR && kind == sys::NC_CHUNKED {
                    let (mut shuffle, mut on, mut level) = (0, 0, 0);
                    let rc = locked(|| unsafe { sys::nc_inq_var_deflate(gin, vin, &mut shuffle, &mut on, &mut level) });
                    let mut nfilters = 0usize;
                    let rc2 = locked(|| unsafe { sys::nc_inq_var_filter_ids(gin, vin, &mut nfilters, std::ptr::null_mut()) });
                    let deflate = if rc == sys::NC_NOERR && on != 0 {
                        Some((shuffle, level))
                    } else if rc2 == sys::NC_NOERR && nfilters > 0 {
                        Some((1, 4))
                    } else {
                        None
                    };
                    (Some(clip(&chunk)), deflate)
                } else {
                    (None, None)
                }
            }
            None => (None, None),
        };
        if let Some(chunk) = chunk {
            locked(|| unsafe { sys::nc_def_var_chunking(gout, vout, sys::NC_CHUNKED, chunk.as_ptr()) });
        }
        if let Some((shuffle, level)) = deflate {
            locked(|| unsafe { sys::nc_def_var_deflate(gout, vout, shuffle, 1, level) });
        }

        match &v.derived {
            Some(d) => {
                for a in &d.info.attrs {
                    if let Some(t) = a.value.as_str() {
                        self.put_text(gout, vout, &a.name, t)?;
                    }
                }
                let fill = cstr("_FillValue")?;
                call(&what, || unsafe { sys::nc_put_att_float(gout, vout, fill.as_ptr(), sys::NC_FLOAT, 1, &CALC_FILL) })?;
            }
            None => self.copy_atts(gin, vin, gout, vout, &format!("{name}:"))?,
        }
        Ok(VarIds { gin, vin, gout, vout, xtype, elem })
    }

    /// Copies one piece: `start` in the input, `off` in the output.
    #[allow(clippy::too_many_arguments)]
    fn copy(
        &self,
        file: &NcFile,
        v: &PlanVar,
        id: &VarIds,
        start: &[usize],
        off: &[usize],
        cnt: &[usize],
        n: usize,
    ) -> Result<(), String> {
        let what = format!("writing {}", v.path);
        if let Some(d) = &v.derived {
            let req = SliceRequest { var: v.path.clone(), start: start.to_vec(), count: cnt.to_vec(), stride: vec![1; cnt.len()] };
            let s = file.read_derived(d, &req).map_err(|e| e.to_string())?;
            let data: Vec<f32> = s.data.iter().map(|&x| if x.is_nan() { CALC_FILL } else { x }).collect();
            return call(&what, || unsafe {
                sys::nc_put_vara_float(id.gout, id.vout, off.as_ptr(), cnt.as_ptr(), data.as_ptr())
            });
        }
        let read = format!("reading {}", v.path);
        if id.xtype == sys::NC_STRING {
            let mut ptrs: Vec<*mut c_char> = vec![std::ptr::null_mut(); n];
            call(&read, || unsafe { sys::nc_get_vara_string(id.gin, id.vin, start.as_ptr(), cnt.as_ptr(), ptrs.as_mut_ptr()) })?;
            let put = call(&what, || unsafe {
                sys::nc_put_vara_string(id.gout, id.vout, off.as_ptr(), cnt.as_ptr(), ptrs.as_mut_ptr().cast())
            });
            locked(|| unsafe { sys::nc_free_string(n, ptrs.as_mut_ptr()) });
            return put;
        }
        let mut buf = vec![0u8; n * id.elem];
        call(&read, || unsafe { sys::nc_get_vara(id.gin, id.vin, start.as_ptr(), cnt.as_ptr(), buf.as_mut_ptr().cast()) })?;
        call(&what, || unsafe { sys::nc_put_vara(id.gout, id.vout, off.as_ptr(), cnt.as_ptr(), buf.as_ptr().cast()) })
    }
}

/// "YYYY-MM-DD HH:MM:SS TorCDF: extracted from <name>: <summary>" with the current UTC time.
pub fn history_line(file_name: &str, summary: &str) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let t = crate::cf::TimeUnits::parse("seconds since 1970-01-01", None)
        .expect("valid units")
        .decode(secs as f64);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} TorCDF: extracted from {file_name}: {summary}",
        t.year, t.month, t.day, t.hour, t.minute, t.second as u32
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{DimKey, Selection};
    use crate::{SliceRequest, fixtures};
    use netcdf::AttributeValue;
    use std::time::{Duration, Instant};

    fn key(group: &str, name: &str) -> DimKey {
        DimKey { group: group.into(), name: name.into() }
    }

    fn sel(vars: &[&str], ranges: &[(DimKey, (usize, usize))]) -> Selection {
        Selection { vars: vars.iter().map(|s| s.to_string()).collect(), ranges: ranges.iter().cloned().collect() }
    }

    fn save(nc: &NcFile, s: &Selection, out: &Path) -> Vec<String> {
        let plan = nc.plan_extract(s).unwrap();
        write_extract(nc, &plan, out, "LINE", &|_| true).unwrap()
    }

    fn attr(f: &netcdf::File, var: &str, name: &str) -> AttributeValue {
        f.variable(var).unwrap().attribute(name).unwrap().value().unwrap()
    }

    /// (shuffle, deflate level) of `var` in group `group` of `path`.
    fn deflate_of(path: &Path, group: &str, var: &str) -> (i32, i32) {
        use std::ffi::CString;
        let _g = netcdf_sys::libnetcdf_lock.lock();
        let c = CString::new(path.to_str().unwrap()).unwrap();
        let (mut ncid, mut gid, mut varid) = (0, 0, 0);
        let (mut shuffle, mut deflate, mut level) = (0, 0, 0);
        unsafe {
            assert_eq!(netcdf_sys::nc_open(c.as_ptr(), netcdf_sys::NC_NOWRITE, &mut ncid), 0);
            let g = CString::new(group).unwrap();
            assert_eq!(netcdf_sys::nc_inq_grp_full_ncid(ncid, g.as_ptr(), &mut gid), 0);
            let v = CString::new(var).unwrap();
            assert_eq!(netcdf_sys::nc_inq_varid(gid, v.as_ptr(), &mut varid), 0);
            assert_eq!(netcdf_sys::nc_inq_var_deflate(gid, varid, &mut shuffle, &mut deflate, &mut level), 0);
            netcdf_sys::nc_close(ncid);
        }
        (shuffle, if deflate != 0 { level } else { 0 })
    }

    #[test]
    fn raw_copy_is_exact() {
        let dir = tempfile::tempdir().unwrap();
        let src = fixtures::write_classic(dir.path());
        let nc = NcFile::open(&src).unwrap();
        let out = dir.path().join("out.nc");
        save(&nc, &sel(&["/t"], &[(key("/", "time"), (1, 2))]), &out);
        let (a, b) = (netcdf::open(&src).unwrap(), netcdf::open(&out).unwrap());
        let want: Vec<i16> = a.variable("t").unwrap().get_values((1..3, .., ..)).unwrap();
        let got: Vec<i16> = b.variable("t").unwrap().get_values(..).unwrap();
        assert_eq!(got, want);
        for name in ["scale_factor", "add_offset", "_FillValue"] {
            assert_eq!(attr(&b, "t", name), attr(&a, "t", name), "{name}");
        }
        let req = |start: Vec<usize>| SliceRequest { var: "/t".into(), start, count: vec![2, 4, 8], stride: vec![1, 1, 1] };
        let x = nc.read_slice(&req(vec![1, 0, 0])).unwrap().data;
        let y = NcFile::open(&out).unwrap().read_slice(&req(vec![0, 0, 0])).unwrap().data;
        assert!(x.iter().zip(&y).all(|(p, q)| p == q || (p.is_nan() && q.is_nan())));

        let src = fixtures::write_i64(dir.path());
        let nc = NcFile::open(&src).unwrap();
        let out = dir.path().join("big.nc");
        save(&nc, &sel(&["/big"], &[]), &out);
        let b = netcdf::open(&out).unwrap();
        let got: Vec<i64> = b.variable("big").unwrap().get_values(..).unwrap();
        assert_eq!(got, fixtures::I64_VALUES);
        assert_eq!(attr(&b, "big", "valid_max"), AttributeValue::Longlong(1 << 62));
    }

    #[test]
    fn keeps_compression_and_history() {
        let dir = tempfile::tempdir().unwrap();
        let src = fixtures::write_desc_lat(dir.path());
        let nc = NcFile::open(&src).unwrap();
        let lats = nc.read_coord("/lat").unwrap();
        let lons = nc.read_coord("/lon").unwrap();
        let lat = crate::extract::range::value_range(&lats, 20.0, -10.0).unwrap();
        assert_eq!(lat, (1, 4));
        let lon = crate::extract::range::lon_range(&lons, 30.0, 60.0).unwrap();
        let out = dir.path().join("crop.nc");
        save(&nc, &sel(&["/t"], &[(key("/", "lat"), lat), (key("/", "lon"), lon)]), &out);
        assert_eq!(deflate_of(&out, "/", "t"), (1, 5));
        let o = NcFile::open(&out).unwrap();
        assert_eq!(o.meta.format, crate::FileFormat::Netcdf4);
        let t = o.meta.var("/t").unwrap();
        assert_eq!(t.storage, crate::Storage::Chunked { chunk: vec![1, 4, 4], compressed: true });
        let g = |n: &str| o.meta.root.attrs.iter().find(|a| a.name == n).and_then(|a| a.value.as_str()).map(str::to_string);
        assert_eq!(g("history").as_deref(), Some("made by fixture\nLINE"));
        assert_eq!(g("title").as_deref(), Some("desc"));
        assert_eq!(o.read_coord("/lat").unwrap(), [20.0, 10.0, 0.0, -10.0]);
        assert_eq!(o.read_coord("/lon").unwrap(), [30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn small_batches_copy_everything() {
        let dir = tempfile::tempdir().unwrap();
        let src = fixtures::write_desc_lat(dir.path());
        let nc = NcFile::open(&src).unwrap();
        let s = sel(&["/t"], &[(key("/", "time"), (1, 3)), (key("/", "lat"), (2, 5)), (key("/", "lon"), (3, 30))]);
        let plan = nc.plan_extract(&s).unwrap();
        let out = dir.path().join("small.nc");
        let calls = std::cell::Cell::new(0);
        // 40 bytes: fewer than one 30-value row of f32, so rows are split.
        write_with_budget(&nc, &plan, &out, "LINE", &|_| { calls.set(calls.get() + 1); true }, 40).unwrap();
        assert!(calls.get() > 3 * 5);
        let (a, b) = (netcdf::open(&src).unwrap(), netcdf::open(&out).unwrap());
        let want: Vec<f32> = a.variable("t").unwrap().get_values((1..4, 2..7, 3..33)).unwrap();
        let got: Vec<f32> = b.variable("t").unwrap().get_values(..).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn strings_copied() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let out = dir.path().join("str.nc");
        save(&nc, &sel(&["/ocean/label"], &[(key("/ocean", "time"), (2, 3))]), &out);
        let o = NcFile::open(&out).unwrap();
        assert_eq!(o.read_strings("/ocean/label", &[0], &[3]).unwrap(), ["step-2", "step-3", "step-4"]);
    }

    #[test]
    fn groups_kept() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let out = dir.path().join("g.nc");
        save(&nc, &sel(&["/ocean/uo"], &[(key("/ocean", "time"), (0, 2))]), &out);
        let o = NcFile::open(&out).unwrap();
        assert_eq!(o.meta.var("/ocean/uo").unwrap().shape, [2, 10, 20]);
        let title = o.meta.root.attrs.iter().find(|a| a.name == "title").and_then(|a| a.value.as_str());
        assert_eq!(title, Some("fixture"));
        // No history before: just the new line.
        let h = o.meta.root.attrs.iter().find(|a| a.name == "history").and_then(|a| a.value.as_str());
        assert_eq!(h, Some("LINE"));
    }

    #[test]
    fn derived_written() {
        let dir = tempfile::tempdir().unwrap();
        let src = fixtures::write_nc4(dir.path());
        let nc = NcFile::open(&src).unwrap();
        let d = nc.check_formula("s", "uo*2", None, None).unwrap();
        assert_eq!(d.path, "/ocean/s");
        nc.add_derived(d);
        let out = dir.path().join("s.nc");
        save(&nc, &sel(&["/ocean/s"], &[(key("/ocean", "time"), (1, 2))]), &out);
        let (a, b) = (netcdf::open(&src).unwrap(), netcdf::open(&out).unwrap());
        let ga = a.group("ocean").unwrap().unwrap();
        let gb = b.group("ocean").unwrap().unwrap();
        let uo: Vec<f32> = ga.variable("uo").unwrap().get_values((1..3, .., ..)).unwrap();
        let s = gb.variable("s").unwrap();
        assert_eq!(s.vartype(), netcdf::types::NcVariableType::Float(netcdf::types::FloatType::F32));
        let got: Vec<f32> = s.get_values(..).unwrap();
        let want: Vec<f32> = uo.iter().map(|&u| if u == 1e20 { 9.96921e36 } else { u * 2.0 }).collect();
        assert_eq!(got, want);
        assert_eq!(s.attribute("formula").unwrap().value().unwrap(), AttributeValue::Str("uo*2".into()));
        assert!(gb.variable("time").is_some() && gb.variable("lat").is_some() && gb.variable("lon").is_some());
    }

    #[test]
    fn cancel_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let target = dir.path().join("target");
        std::fs::create_dir(&target).unwrap();
        let plan = nc.plan_extract(&sel(&["/ocean/uo"], &[])).unwrap();
        let r = write_extract(&nc, &plan, &target.join("out.nc"), "LINE", &|_| false);
        assert!(r.is_err());
        assert_eq!(std::fs::read_dir(&target).unwrap().count(), 0);
    }

    #[test]
    fn writes_into_non_ascii_folder() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let target = dir.path().join("Ölçüm");
        std::fs::create_dir(&target).unwrap();
        let out = target.join("out.nc");
        save(&nc, &sel(&["/ocean/uo"], &[(key("/ocean", "time"), (0, 1))]), &out);
        assert_eq!(NcFile::open(&out).unwrap().meta.var("/ocean/uo").unwrap().shape, [1, 10, 20]);
        let names: Vec<_> = std::fs::read_dir(&target).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names, [std::ffi::OsString::from("out.nc")]);
    }

    #[test]
    fn refuses_the_source_file() {
        let dir = tempfile::tempdir().unwrap();
        let src = fixtures::write_nc4(dir.path());
        let nc = NcFile::open(&src).unwrap();
        let plan = nc.plan_extract(&sel(&["/ocean/uo"], &[])).unwrap();
        assert!(write_extract(&nc, &plan, &src, "LINE", &|_| true).is_err());
        assert!(NcFile::open(&src).is_ok());
    }

    #[test]
    fn job_runs_in_background() {
        let dir = tempfile::tempdir().unwrap();
        let nc = Arc::new(NcFile::open(&fixtures::write_nc4(dir.path())).unwrap());
        let plan = nc.plan_extract(&sel(&["/ocean/uo", "/ocean/vo"], &[])).unwrap();
        let out = dir.path().join("job.nc");
        let job = ExtractJob::start(nc, plan, out.clone(), "LINE".into());
        let t0 = Instant::now();
        let done = loop {
            if let Some(r) = job.finished() {
                break r;
            }
            assert!(t0.elapsed() < Duration::from_secs(20), "job did not finish");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(done, Ok(out));
        assert_eq!(job.progress(), 1.0);
    }

    #[test]
    fn history_line_format() {
        let l = history_line("a.nc", "time 1..2");
        let (stamp, rest) = l.split_at(19);
        assert_eq!(rest, " TorCDF: extracted from a.nc: time 1..2");
        let b = stamp.as_bytes();
        assert!(b[4] == b'-' && b[7] == b'-' && b[10] == b' ' && b[13] == b':' && b[16] == b':', "{stamp}");
        assert!(stamp.starts_with("20"));
    }
}
