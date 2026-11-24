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

/// The region dimensions `y`, `x` of one variable over its window: which
/// cells count, their weights and their lon/lat. Kept small: regular grids
/// hold one weight per row and one flag per column; curvilinear ones 12
/// bytes per cell of the window.
enum Geom {
    Regular {
        y: usize,
        x: usize,
        /// Per window row: its weight.
        row_w: Vec<f32>,
        /// Per window column: inside the box (only across the seam can a
        /// column of the window lie outside it).
        col_in: Vec<bool>,
        lats: Vec<f64>,
        lons: Vec<f64>,
    },
    Curvilinear {
        y: usize,
        x: usize,
        nx: usize,
        /// Per window cell: its weight, NaN when outside the box.
        cell: Vec<f32>,
        lonlat: Vec<(f32, f32)>,
    },
}

impl Geom {
    fn dims(&self) -> (usize, usize) {
        match self {
            Geom::Regular { y, x, .. } | Geom::Curvilinear { y, x, .. } => (*y, *x),
        }
    }

    /// The weight of window cell (iy, ix); `None` outside the box.
    fn weight(&self, iy: usize, ix: usize) -> Option<f64> {
        match self {
            Geom::Regular { row_w, col_in, .. } => col_in[ix].then(|| f64::from(row_w[iy])),
            Geom::Curvilinear { nx, cell, .. } => {
                let w = cell[iy * nx + ix];
                (!w.is_nan()).then_some(f64::from(w))
            }
        }
    }

    fn lonlat(&self, iy: usize, ix: usize) -> (f64, f64) {
        match self {
            Geom::Regular { lats, lons, .. } => (lons[ix], lats[iy]),
            Geom::Curvilinear { nx, lonlat, .. } => {
                let (lo, la) = lonlat[iy * nx + ix];
                (f64::from(lo), f64::from(la))
            }
        }
    }
}

/// What one variable reads.
struct Prepared {
    path: String,
    start: Vec<usize>,
    count: Vec<usize>,
    geom: Option<Geom>,
    weighted: bool,
}

/// Comparison slack for coordinates against typed degrees, as the range
/// helpers use for float32 coordinates.
fn slack(a: f64, b: f64) -> f64 {
    1e-6 * a.abs().max(b.abs()).max(1.0)
}

/// Whether `lon` is inside [w, e], across the seam too.
fn lon_in(lon: f64, w: f64, e: f64) -> bool {
    if !lon.is_finite() {
        return false;
    }
    let width = if e - w >= 360.0 { 360.0 } else { (e - w).rem_euclid(360.0) };
    let off = (lon - w).rem_euclid(360.0);
    let tol = slack(w, e);
    off <= width + tol || off >= 360.0 - tol
}

/// Whether (lat, lon) is inside the box (s, n, w, e).
fn in_box(lat: f64, lon: f64, b: [f64; 4]) -> bool {
    let (s, n) = (b[0].min(b[1]), b[0].max(b[1]));
    let tol = slack(s, n);
    lat >= s - tol && lat <= n + tol && lon_in(lon, b[2], b[3])
}

fn cos_weight(lat: f64, weighted: bool) -> f32 {
    if weighted { lat.to_radians().cos().max(0.0) as f32 } else { 1.0 }
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
    let geom = match detect_geo(&ds, &info) {
        Geo::Regular { lat_dim: y, lon_dim: x, lat_var, lon_var } => {
            let (lat, lon) = (read(&lat_var)?, read(&lon_var)?);
            // The index window matches the box exactly, except across the
            // seam, where the whole longitude axis is read and masked.
            let mut seam = None;
            if let Some(b) = req.region {
                (start[y], count[y]) = value_range(&lat, b[0], b[1]).ok_or(NONE)?;
                (start[x], count[x]) = match lon_range(&lon, b[2], b[3]) {
                    Ok(r) => r,
                    Err(e) if e == SEAM => {
                        seam = Some((b[2], b[3]));
                        (0, lon.len())
                    }
                    Err(_) => return Err(NONE.into()),
                };
            }
            let lats = lat[start[y]..start[y] + count[y]].to_vec();
            let lons = lon[start[x]..start[x] + count[x]].to_vec();
            let row_w = lats.iter().map(|&la| cos_weight(la, req.weighted)).collect();
            let col_in = lons.iter().map(|&lo| seam.is_none_or(|(w, e)| lon_in(lo, w, e))).collect();
            Some(Geom::Regular { y, x, row_w, col_in, lats, lons })
        }
        Geo::Curvilinear { y_dim: y, x_dim: x, lat_var, lon_var } => {
            let (yn, xn) = (&info.dims[y], &info.dims[x]);
            let h = |p: &str| file.read_horizontal(p, yn, xn).map_err(|e| e.to_string());
            let (lat, lon) = (h(&lat_var)?, h(&lon_var)?);
            let full_nx = info.shape[x];
            if let Some(b) = req.region {
                let [ry, rx] = curv_box(&lat, &lon, info.shape[y], full_nx, b[0], b[1], b[2], b[3]).ok_or(NONE)?;
                ((start[y], count[y]), (start[x], count[x])) = (ry, rx);
            }
            let mut cell = Vec::with_capacity(count[y] * count[x]);
            let mut lonlat = Vec::with_capacity(count[y] * count[x]);
            for j in start[y]..start[y] + count[y] {
                for i in start[x]..start[x] + count[x] {
                    let (la, lo) = (lat[j * full_nx + i], lon[j * full_nx + i]);
                    let inside = req.region.is_none_or(|b| in_box(la, lo, b));
                    cell.push(if inside { cos_weight(la, req.weighted) } else { f32::NAN });
                    lonlat.push((lo as f32, la as f32));
                }
            }
            Some(Geom::Curvilinear { y, x, nx: count[x], cell, lonlat })
        }
        Geo::None => None,
    };
    if count.contains(&0) {
        return Err(NONE.into());
    }
    let weighted = req.weighted && geom.is_some();
    Ok(Prepared { path: path.into(), start, count, geom, weighted })
}

/// Reads `p` piece by piece and calls `f(value, weight, window index)` for
/// every cell inside the box (`value` not finite = missing). `step(done)`
/// after each piece with the share of `p` read; false cancels.
fn pass(
    file: &NcFile,
    p: &Prepared,
    f: &mut dyn FnMut(f32, f64, &[usize]),
    step: &mut dyn FnMut(f64) -> bool,
) -> Result<(), String> {
    let nd = p.count.len();
    let total: f64 = p.count.iter().map(|&n| n as f64).product::<f64>().max(1.0);
    let mut done = 0.0;
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
                Some(g) => {
                    let (y, x) = g.dims();
                    g.weight(idx[y], idx[x])
                }
                None => Some(1.0),
            };
            if let Some(w) = weight {
                f(v, w, &idx);
            }
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < off[d] + cnt[d] {
                    break;
                }
                idx[d] = off[d];
            }
        }
        done += n as f64;
        if !step(done / total) {
            return Err("cancelled".into());
        }
    }
    Ok(())
}

/// Flat position of a window index (row-major), as the accumulator keeps it.
fn flat(idx: &[usize], count: &[usize]) -> u64 {
    idx.iter().zip(count).fold(0u64, |acc, (&i, &n)| acc * n as u64 + i as u64)
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
    let lonlat = p.geom.as_ref().map(|g| {
        let (y, x) = g.dims();
        g.lonlat(idx[y], idx[x])
    });
    let index = idx.iter().zip(&p.start).map(|(i, s)| i + s).collect();
    Extreme { value, index, lonlat }
}

pub fn compute_stats_with(
    file: &NcFile,
    req: &StatsRequest,
    progress: &dyn Fn(f32) -> bool,
    exact_limit: usize,
) -> Result<StatsOutcome, String> {
    let nvars = req.vars.len().max(1) as f64;
    let mut out = Vec::new();
    for (k, var) in req.vars.iter().enumerate() {
        // Each variable takes an equal share of the bar; a second pass
        // shares it with the first.
        let share = |frac: f64| progress(((k as f64 + frac.clamp(0.0, 1.0)) / nvars) as f32);
        // Prepared one at a time: masks are never held for all variables.
        let p = match prepare(file, var, req) {
            Ok(p) => p,
            Err(e) => {
                out.push(Err(e));
                continue;
            }
        };
        let mut acc = Acc::default();
        let mut kept: Option<Vec<f32>> = Some(Vec::new());
        let read = pass(
            file,
            &p,
            &mut |v, w, idx| {
                if !v.is_finite() {
                    acc.push_missing(1);
                    return;
                }
                acc.push(v, w, flat(idx, &p.count));
                if acc.valid() as usize > exact_limit {
                    kept = None;
                } else if let Some(k) = kept.as_mut() {
                    k.push(v);
                }
            },
            &mut |f| share(f / 2.0),
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
                let mut counts = vec![0u64; HIST_BINS];
                let read = pass(
                    file,
                    &p,
                    &mut |v, _, _| {
                        if v.is_finite() {
                            counts[bin_of(f64::from(v), lo, hi, HIST_BINS)] += 1;
                        }
                    },
                    &mut |f| share(0.5 + f / 2.0),
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
        share(1.0);
        out.push(Ok(VarStats {
            path: p.path.clone(),
            valid: acc.valid(),
            missing: acc.missing(),
            min: extreme(&p, lo, lo_pos),
            max: extreme(&p, hi, hi_pos),
            mean,
            std,
            weighted: p.weighted,
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

    /// `f32grid.nc`: lat 10, 11 and lon 120.0..120.5 step 0.1 stored as
    /// float32; `t(lat, lon)` = 1, except one +inf.
    fn write_f32_grid(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("f32grid.nc");
        let mut f = netcdf::create(&path).unwrap();
        f.add_dimension("lat", 2).unwrap();
        f.add_dimension("lon", 6).unwrap();
        let mut lat = f.add_variable::<f32>("lat", &["lat"]).unwrap();
        lat.put_attribute("units", "degrees_north").unwrap();
        lat.put_values(&[10.0f32, 11.0], ..).unwrap();
        let mut lon = f.add_variable::<f32>("lon", &["lon"]).unwrap();
        lon.put_attribute("units", "degrees_east").unwrap();
        let lons: Vec<f32> = (0..6).map(|i| 120.0 + 0.1 * i as f32).collect();
        lon.put_values(&lons, ..).unwrap();
        let mut t = f.add_variable::<f32>("t", &["lat", "lon"]).unwrap();
        let mut v = vec![1.0f32; 12];
        v[11] = f32::INFINITY;
        t.put_values(&v, ..).unwrap();
        path
    }

    #[test]
    fn float32_edges_are_kept() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&write_f32_grid(dir.path())).unwrap();
        // Columns 120.1 .. 120.3 (three), both rows.
        let s = one(&nc, &req(&["/t"], &[], Some([10.0, 11.0, 120.1, 120.3]), false));
        assert_eq!(s.valid, 6);
    }

    #[test]
    fn infinity_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&write_f32_grid(dir.path())).unwrap();
        let s = one(&nc, &req(&["/t"], &[], None, false));
        assert_eq!((s.valid, s.missing), (11, 1));
        assert_eq!((s.mean, s.max.value), (1.0, 1.0));
        assert_eq!(s.histogram.iter().sum::<u64>(), 11);
    }
}
