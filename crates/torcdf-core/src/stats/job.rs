//! Statistics of variables over a selection, read piece by piece in the
//! background. The lat/lon box is applied to each variable on its own grid.

use super::acc::*;
use crate::extract::DimKey;
use crate::NcFile;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Percentiles are exact up to this many valid values per variable.
pub const EXACT_LIMIT: usize = 20_000_000;
/// Bins of the second pass above [`EXACT_LIMIT`].
pub const HIST_BINS: usize = 65_536;

#[derive(Clone, Debug, PartialEq)]
pub struct StatsRequest {
    pub vars: Vec<String>,
    /// Time and other non-region dimensions (start, count); missing = whole.
    pub ranges: BTreeMap<DimKey, (usize, usize)>,
    /// (south, north, west, east) degrees; applied per variable on its own lat/lon.
    pub region: Option<[f64; 4]>,
    pub weighted: bool,
}

/// A minimum or maximum and where it is.
#[derive(Clone, Debug, PartialEq)]
pub struct Extreme {
    pub value: f64,
    /// Full-file index per dimension of the variable.
    pub index: Vec<usize>,
    /// The cell's (lon, lat), when the variable has them.
    pub lonlat: Option<(f64, f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VarStats {
    pub path: String,
    pub valid: u64,
    pub missing: u64,
    pub min: Extreme,
    pub max: Extreme,
    pub mean: f64,
    pub std: f64,
    /// Mean and std are area-weighted.
    pub weighted: bool,
    /// In [`PERCENTILES`] order.
    pub percentiles: Vec<f64>,
    pub approximate: bool,
    /// [`SHOW_BINS`] counts from min to max.
    pub histogram: Vec<u64>,
    /// false: the variable has no lat/lon, so the region was not applied.
    pub region_applied: bool,
}

/// One entry per requested variable: its statistics, or why there are none.
pub type StatsOutcome = Vec<Result<VarStats, String>>;

pub fn compute_stats(file: &NcFile, req: &StatsRequest, progress: &dyn Fn(f32) -> bool) -> Result<StatsOutcome, String> {
    compute_stats_with(file, req, progress, EXACT_LIMIT)
}

const NONE: &str = "no valid values in the selection";
/// Largest piece read at once (bytes of f32 values).
const BATCH_BYTES: usize = 64 * 1024 * 1024;

/// The region dimensions of one variable over its window: per cell, its
/// weight (`None`: outside the box) and its lon/lat.
struct Geom {
    y: usize,
    x: usize,
    nx: usize,
    cell: Vec<Option<f64>>,
    lonlat: Vec<(f64, f64)>,
}

/// (y dim, x dim, latitudes, longitudes, regular grid?).
type GeoAxes = (usize, usize, Vec<f64>, Vec<f64>, bool);

/// What one variable reads.
struct Prepared {
    path: String,
    start: Vec<usize>,
    count: Vec<usize>,
    geom: Option<Geom>,
    weighted: bool,
}

/// Whether (lat, lon) is inside the box (s, n, w, e), across the seam too.
fn in_box(lat: f64, lon: f64, b: [f64; 4]) -> bool {
    let (s, n) = (b[0].min(b[1]), b[0].max(b[1]));
    let e = 1e-6 * s.abs().max(n.abs()).max(1.0);
    if !(lat >= s - e && lat <= n + e) || !lon.is_finite() {
        return false;
    }
    let width = if b[3] - b[2] >= 360.0 { 360.0 } else { (b[3] - b[2]).rem_euclid(360.0) };
    let off = (lon - b[2]).rem_euclid(360.0);
    off <= width + 1e-6 || off >= 360.0 - 1e-6
}

/// The window, mask and weights of `path` for `req`.
fn prepare(file: &NcFile, path: &str, req: &StatsRequest) -> Result<Prepared, String> {
    use crate::cf::{Geo, detect_geo};
    use crate::extract::range::{SEAM, curv_box, lon_range, value_range};
    let info = file.var_info_any(path).ok_or_else(|| format!("{path} is not in the file"))?;
    if !info.dtype.is_numeric() {
        return Err("not numeric".into());
    }
    let mut start = Vec::new();
    let mut count = Vec::new();
    for (name, &len) in info.dims.iter().zip(&info.shape) {
        let (s, n) = file.dim_key(path, name).and_then(|k| req.ranges.get(&k).copied()).unwrap_or((0, len));
        if s + n > len {
            return Err(format!("the range of {name} is outside the variable"));
        }
        start.push(s);
        count.push(n);
    }
    let ds = file.meta_with_derived();
    let read = |p: &str| file.read_coord(p).map_err(|e| e.to_string());
    // (y dim, x dim, latitudes, longitudes, regular?): regular grids hold 1-D
    // axes, curvilinear ones the full (y, x) planes.
    let geo: Option<GeoAxes> = match detect_geo(&ds, &info) {
        Geo::Regular { lat_dim, lon_dim, lat_var, lon_var } => Some((lat_dim, lon_dim, read(&lat_var)?, read(&lon_var)?, true)),
        Geo::Curvilinear { y_dim, x_dim, lat_var, lon_var } => {
            let (yn, xn) = (&info.dims[y_dim], &info.dims[x_dim]);
            let h = |p: &str| file.read_horizontal(p, yn, xn).map_err(|e| e.to_string());
            Some((y_dim, x_dim, h(&lat_var)?, h(&lon_var)?, false))
        }
        Geo::None => None,
    };
    let Some((y, x, lat, lon, regular)) = geo else {
        if count.contains(&0) {
            return Err(NONE.into());
        }
        return Ok(Prepared { path: path.into(), start, count, geom: None, weighted: false });
    };
    let full_nx = info.shape[x];
    if let Some(b) = req.region {
        let window = if regular {
            let ys = value_range(&lat, b[0], b[1]).ok_or(NONE)?;
            let xs = match lon_range(&lon, b[2], b[3]) {
                Ok(r) => r,
                Err(e) if e == SEAM => (0, lon.len()),
                Err(_) => return Err(NONE.into()),
            };
            [ys, xs]
        } else {
            curv_box(&lat, &lon, info.shape[y], full_nx, b[0], b[1], b[2], b[3]).ok_or(NONE)?
        };
        (start[y], count[y]) = window[0];
        (start[x], count[x]) = window[1];
    }
    if count.contains(&0) {
        return Err(NONE.into());
    }
    let (ny, nx) = (count[y], count[x]);
    let mut cell = Vec::with_capacity(ny * nx);
    let mut lonlat = Vec::with_capacity(ny * nx);
    for j in start[y]..start[y] + ny {
        for i in start[x]..start[x] + nx {
            let (la, lo) = if regular { (lat[j], lon[i]) } else { (lat[j * full_nx + i], lon[j * full_nx + i]) };
            let inside = req.region.is_none_or(|b| in_box(la, lo, b));
            let w = if req.weighted { la.to_radians().cos().max(0.0) } else { 1.0 };
            cell.push(inside.then_some(w));
            lonlat.push((lo, la));
        }
    }
    let geom = Geom { y, x, nx, cell, lonlat };
    Ok(Prepared { path: path.into(), start, count, geom: Some(geom), weighted: req.weighted })
}

/// Cells of a variable's window.
fn cells(p: &Prepared) -> u64 {
    p.count.iter().map(|&n| n as u64).product()
}

/// Reads `p` piece by piece and calls `f(value, weight, flat position)` for
/// every cell inside the box (`value` NaN = missing). `step(n)` after each
/// piece of `n` cells; false cancels.
fn pass(
    file: &NcFile,
    p: &Prepared,
    f: &mut dyn FnMut(f32, f64, u64),
    step: &mut dyn FnMut(u64) -> bool,
) -> Result<(), String> {
    let nd = p.count.len();
    let mut wstride = vec![1u64; nd];
    for d in (0..nd.saturating_sub(1)).rev() {
        wstride[d] = wstride[d + 1] * p.count[d + 1] as u64;
    }
    for (off, cnt) in crate::extract::write::slabs(&p.count, 4, BATCH_BYTES) {
        let n: usize = cnt.iter().product();
        if n == 0 {
            continue;
        }
        let req = crate::SliceRequest {
            var: p.path.clone(),
            start: p.start.iter().zip(&off).map(|(s, o)| s + o).collect(),
            count: cnt.clone(),
            stride: vec![1; nd],
        };
        let data = file.read_slice(&req).map_err(|e| e.to_string())?.data;
        // Window-relative index of each value, as an odometer.
        let mut idx = off.clone();
        for &v in &data {
            let weight = match &p.geom {
                Some(g) => g.cell[(idx[g.y] * g.nx) + idx[g.x]],
                None => Some(1.0),
            };
            if let Some(w) = weight {
                let pos = idx.iter().zip(&wstride).map(|(&i, &s)| i as u64 * s).sum();
                f(v, w, pos);
            }
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < off[d] + cnt[d] {
                    break;
                }
                idx[d] = off[d];
            }
        }
        if !step(n as u64) {
            return Err("cancelled".into());
        }
    }
    Ok(())
}

/// The extreme at window position `pos`.
fn extreme(p: &Prepared, value: f64, pos: u64) -> Extreme {
    let mut rest = pos;
    let mut idx = vec![0usize; p.count.len()];
    for d in (0..p.count.len()).rev() {
        let n = p.count[d].max(1) as u64;
        idx[d] = (rest % n) as usize;
        rest /= n;
    }
    let lonlat = p.geom.as_ref().map(|g| g.lonlat[idx[g.y] * g.nx + idx[g.x]]);
    let index = idx.iter().zip(&p.start).map(|(i, s)| i + s).collect();
    Extreme { value, index, lonlat }
}

pub fn compute_stats_with(
    file: &NcFile,
    req: &StatsRequest,
    progress: &dyn Fn(f32) -> bool,
    exact_limit: usize,
) -> Result<StatsOutcome, String> {
    let prepared: Vec<Result<Prepared, String>> = req.vars.iter().map(|v| prepare(file, v, req)).collect();
    let mut total: u64 = prepared.iter().flatten().map(cells).sum::<u64>().max(1);
    let mut done = 0u64;
    let mut shown = 0f32;
    let mut out = Vec::new();
    for p in prepared {
        let p = match p {
            Ok(p) => p,
            Err(e) => {
                out.push(Err(e));
                continue;
            }
        };
        let mut acc = Acc::default();
        let mut kept: Option<Vec<f32>> = Some(Vec::new());
        let mut step = |n: u64, total: u64| {
            done += n;
            shown = shown.max((done as f64 / total as f64).min(1.0) as f32);
            progress(shown)
        };
        let read = pass(
            file,
            &p,
            &mut |v, w, pos| {
                if v.is_nan() {
                    acc.push_missing(1);
                    return;
                }
                acc.push(v, w, pos);
                if acc.valid() as usize > exact_limit {
                    kept = None;
                } else if let Some(k) = kept.as_mut() {
                    k.push(v);
                }
            },
            &mut |n| step(n, total),
        );
        match read {
            Err(e) if e == "cancelled" => return Err(e),
            Err(e) => {
                out.push(Err(e));
                continue;
            }
            Ok(()) => {}
        }
        let (Some((lo, lo_pos)), Some((hi, hi_pos)), Some(mean), Some(std)) = (acc.min(), acc.max(), acc.mean(), acc.std())
        else {
            out.push(Err(NONE.into()));
            continue;
        };
        let (percentiles, histogram, approximate) = match kept.as_mut() {
            Some(k) => {
                let hist = bin_values(k, lo, hi, SHOW_BINS);
                (exact_percentiles(k, &PERCENTILES), hist, false)
            }
            None => {
                // Second pass: a fine histogram between min and max.
                total += cells(&p);
                let mut counts = vec![0u64; HIST_BINS];
                let read = pass(
                    file,
                    &p,
                    &mut |v, _, _| {
                        if !v.is_nan() {
                            counts[bin_of(f64::from(v), lo, hi, HIST_BINS)] += 1;
                        }
                    },
                    &mut |n| step(n, total),
                );
                match read {
                    Err(e) if e == "cancelled" => return Err(e),
                    Err(e) => {
                        out.push(Err(e));
                        continue;
                    }
                    Ok(()) => {}
                }
                (hist_percentiles(&counts, lo, hi, &PERCENTILES), regroup(&counts, SHOW_BINS), true)
            }
        };
        out.push(Ok(VarStats {
            path: p.path.clone(),
            valid: acc.valid(),
            missing: acc.missing(),
            min: extreme(&p, lo, lo_pos),
            max: extreme(&p, hi, hi_pos),
            mean,
            std,
            weighted: p.weighted && p.geom.is_some(),
            percentiles,
            approximate,
            histogram,
            region_applied: req.region.is_some() && p.geom.is_some(),
        }));
    }
    Ok(out)
}

/// A statistics run on its own thread, holding its own `Arc<NcFile>`.
pub struct StatsJob {
    state: Arc<JobState>,
}

#[derive(Default)]
struct JobState {
    progress: AtomicU32,
    cancel: AtomicBool,
    outcome: Mutex<Option<Result<StatsOutcome, String>>>,
}

impl StatsJob {
    pub fn start(file: Arc<NcFile>, req: StatsRequest) -> StatsJob {
        let state = Arc::new(JobState::default());
        let s = state.clone();
        std::thread::spawn(move || {
            let progress = |f: f32| {
                s.progress.store(f.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
                !s.cancel.load(Ordering::Relaxed)
            };
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute_stats(&file, &req, &progress)));
            let outcome = match run {
                Ok(r) => r,
                Err(_) => Err("internal error while computing statistics".into()),
            };
            if outcome.is_ok() {
                s.progress.store(1.0f32.to_bits(), Ordering::Relaxed);
            }
            *s.outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);
        });
        StatsJob { state }
    }

    pub fn progress(&self) -> f32 {
        f32::from_bits(self.state.progress.load(Ordering::Relaxed))
    }

    pub fn cancel(&self) {
        self.state.cancel.store(true, Ordering::Relaxed);
    }

    pub fn finished(&self) -> Option<Result<StatsOutcome, String>> {
        self.state.outcome.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SliceRequest, fixtures};
    use std::time::{Duration, Instant};

    const NONE: &str = "no valid values in the selection";

    fn key(group: &str, name: &str) -> DimKey {
        DimKey { group: group.into(), name: name.into() }
    }

    fn req(vars: &[&str], ranges: &[(DimKey, (usize, usize))], region: Option<[f64; 4]>, weighted: bool) -> StatsRequest {
        StatsRequest {
            vars: vars.iter().map(|s| s.to_string()).collect(),
            ranges: ranges.iter().cloned().collect(),
            region,
            weighted,
        }
    }

    fn run(nc: &NcFile, r: &StatsRequest) -> StatsOutcome {
        compute_stats(nc, r, &|_| true).unwrap()
    }

    fn one(nc: &NcFile, r: &StatsRequest) -> VarStats {
        run(nc, r).remove(0).unwrap()
    }

    fn all(nc: &NcFile, var: &str) -> Vec<f32> {
        let shape = nc.var_info_any(var).unwrap().shape;
        let r = SliceRequest { var: var.into(), start: vec![0; shape.len()], count: shape.clone(), stride: vec![1; shape.len()] };
        nc.read_slice(&r).unwrap().data
    }

    /// Cells of a 2-D lat/lon grid inside the box (lat, then lon).
    fn inside(lat: &[f64], lon: &[f64], b: [f64; 4]) -> usize {
        lat.iter().zip(lon).filter(|(la, lo)| **la >= b[0] && **la <= b[1] && **lo >= b[2] && **lo <= b[3]).count()
    }

    #[test]
    fn crop_matches_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let s = one(&nc, &req(&["/ocean/uo"], &[(key("/ocean", "time"), (0, 2))], None, false));
        let data: Vec<f64> = all(&nc, "/ocean/uo")[..2 * 10 * 20].iter().filter(|v| !v.is_nan()).map(|&v| v as f64).collect();
        assert_eq!(s.valid, data.len() as u64);
        assert_eq!(s.missing, 2 * 10 * 3);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
        assert!((s.mean - mean).abs() < 1e-9 && (s.std - std).abs() < 1e-9);
        let [t, j, k] = s.min.index[..] else { panic!("3 dims") };
        assert_eq!(all(&nc, "/ocean/uo")[t * 200 + j * 20 + k] as f64, s.min.value);
        assert_eq!(s.percentiles.len(), 5);
        assert_eq!(s.histogram.iter().sum::<u64>(), s.valid);
        assert!(!s.approximate && !s.weighted);
    }

    #[test]
    fn packed_is_unpacked() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_classic(dir.path())).unwrap();
        let s = one(&nc, &req(&["/t"], &[], None, false));
        let v: Vec<f64> = all(&nc, "/t").into_iter().filter(|v| !v.is_nan()).map(f64::from).collect();
        assert_eq!(s.min.value, v.iter().copied().fold(f64::INFINITY, f64::min));
        assert_eq!(s.max.value, v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        assert_eq!(s.missing, 1);
    }

    #[test]
    fn weighting_changes_mean_only() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let b = Some([-9.0, 5.0, 0.0, 19.0]);
        let w = one(&nc, &req(&["/ocean/uo"], &[], b, true));
        let u = one(&nc, &req(&["/ocean/uo"], &[], b, false));
        assert!(w.weighted && !u.weighted && w.region_applied);
        assert!((w.mean - u.mean).abs() > 1e-6);
        assert_eq!((w.min.clone(), w.max.clone(), w.percentiles.clone()), (u.min, u.max, u.percentiles));
    }

    #[test]
    fn all_missing_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        let r = run(&nc, &req(&["/ocean/uo"], &[], Some([-9.0, 9.0, 0.0, 2.0]), true));
        assert_eq!(r, vec![Err(NONE.to_string())]);
    }

    #[test]
    fn seam_box_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_desc_lat(dir.path())).unwrap();
        let s = one(&nc, &req(&["/t"], &[], Some([-30.0, 30.0, -20.0, 20.0]), false));
        assert_eq!(s.valid, 4 * 7 * 5);
    }

    #[test]
    fn curvilinear_counts_only_inside() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_curvilinear(dir.path())).unwrap();
        let lat = nc.read_horizontal("/nav_lat", "y", "x").unwrap();
        let lon = nc.read_horizontal("/nav_lon", "y", "x").unwrap();
        let b = [2.0, 4.15, 0.0, 6.0];
        let rect = crate::extract::range::curv_box(&lat, &lon, 5, 6, b[0], b[1], b[2], b[3]).unwrap();
        let want = inside(&lat, &lon, b);
        assert!(rect[0].1 * rect[1].1 > want, "the rectangle must stick out of the box for this test");
        let s = one(&nc, &req(&["/sst"], &[], Some(b), false));
        assert_eq!((s.valid, s.missing), (want as u64, 0));
    }

    #[test]
    fn staggered_pair_uses_own_grids() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::vectors::write_roms_like(dir.path())).unwrap();
        let grid = |la: &str, lo: &str, y: &str, x: &str| {
            (nc.read_horizontal(la, y, x).unwrap(), nc.read_horizontal(lo, y, x).unwrap())
        };
        let (lat_u, lon_u) = grid("/lat_u", "/lon_u", "eta_u", "xi_u");
        let (lat_v, lon_v) = grid("/lat_v", "/lon_v", "eta_v", "xi_v");
        let span = |v: &[f64]| (v.iter().copied().fold(f64::INFINITY, f64::min), v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        let ((la0, la1), (lo0, lo1)) = (span(&lat_u), span(&lon_u));
        let b = [la0 + 0.25 * (la1 - la0), la0 + 0.75 * (la1 - la0), lo0 + 0.25 * (lo1 - lo0), lo0 + 0.75 * (lo1 - lo0)];
        let r = run(&nc, &req(&["/u", "/v"], &[], Some(b), false));
        let (u, v) = (r[0].clone().unwrap(), r[1].clone().unwrap());
        assert_eq!(u.valid, 2 * inside(&lat_u, &lon_u, b) as u64);
        assert_eq!(v.valid, 2 * inside(&lat_v, &lon_v, b) as u64);
        assert!(u.valid > 0 && u.region_applied && v.region_applied);
    }

    #[test]
    fn no_lat_lon_region_not_applied() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_bare(dir.path())).unwrap();
        let s = one(&nc, &req(&["/data"], &[], Some([0.0, 1.0, 0.0, 1.0]), true));
        assert!(!s.region_applied && !s.weighted);
        assert_eq!(s.valid, 12);
    }

    #[test]
    fn large_selection_is_approximate() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_desc_lat(dir.path())).unwrap();
        let s = compute_stats_with(&nc, &req(&["/t"], &[], None, false), &|_| true, 10).unwrap().remove(0).unwrap();
        assert!(s.approximate);
        let exact = exact_percentiles(&mut all(&nc, "/t"), &PERCENTILES);
        let tol = (s.max.value - s.min.value) / HIST_BINS as f64;
        for (a, e) in s.percentiles.iter().zip(&exact) {
            assert!((a - e).abs() <= tol + 1e-9, "{a} vs {e}");
        }
        assert_eq!(s.histogram.iter().sum::<u64>(), s.valid);
    }

    #[test]
    fn calculated_and_hdf5() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        nc.add_derived(nc.check_formula("s", "uo*2", None, None).unwrap());
        let r = run(&nc, &req(&["/ocean/uo", "/ocean/s"], &[], None, false));
        let (u, s) = (r[0].clone().unwrap(), r[1].clone().unwrap());
        assert!((s.mean - 2.0 * u.mean).abs() < 1e-6);
        let h5 = NcFile::open(&fixtures::hdf5::write_hidden_h5(dir.path())).unwrap();
        let a = one(&h5, &req(&["/a"], &[], None, false));
        assert_eq!(a.valid, 6);
        assert!((a.mean - 1.25).abs() < 1e-12);
        let text = run(&nc, &req(&["/ocean/label"], &[], None, false));
        assert_eq!(text, vec![Err("not numeric".to_string())]);
    }

    #[test]
    fn job_cancels() {
        let dir = tempfile::tempdir().unwrap();
        let nc = Arc::new(NcFile::open(&fixtures::write_nc4(dir.path())).unwrap());
        assert_eq!(compute_stats(&nc, &req(&["/ocean/uo"], &[], None, false), &|_| false), Err("cancelled".into()));
        let job = StatsJob::start(nc.clone(), req(&["/ocean/uo"], &[], None, false));
        job.cancel();
        let t0 = Instant::now();
        while job.finished().is_none() {
            assert!(t0.elapsed() < Duration::from_secs(20));
            std::thread::sleep(Duration::from_millis(10));
        }
        let job = StatsJob::start(nc, req(&["/ocean/uo"], &[], None, false));
        let t0 = Instant::now();
        let done = loop {
            if let Some(r) = job.finished() {
                break r;
            }
            assert!(t0.elapsed() < Duration::from_secs(20));
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(done.is_ok() && job.progress() == 1.0);
    }
}
