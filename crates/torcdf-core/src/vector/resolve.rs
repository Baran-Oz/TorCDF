//! Reading a vector pair (spec 7.2): staggering, rotation, magnitude.
//!
//! [`NcFile::resolve_vector`] works out once per pair path which side of
//! each centre point the component points sit on, whether and by what angle
//! to rotate, and the centre coordinates. A pair read then reads each
//! component through `read_slice`, one neighbour further along its staggered
//! axes, averages the bracketing points at full resolution, subsamples,
//! rotates to east/north and takes the magnitude.

use super::{AngleSource, Frame, RotateChoice, VectorPair, VectorParts};
use crate::cf::{Geo, detect_geo, wrap_lon};
use crate::read::check_shape;
use crate::{CoreError, NcFile, Slice, SliceRequest, VarInfo};
use std::collections::HashMap;
use std::sync::Arc;

const NOTE_LIKELY: &str = "Components assumed to follow the grid and rotated to east/north. \
                           Turn off if they already point east/north.";
const NOTE_UNKNOWN: &str = "Components assumed to point east/north (the grid is not visibly \
                            turned). Turn on rotation if they follow the grid.";
const NOTE_STAGGER: &str =
    "Staggering assumed: U points east of T points, V points north of T points.";

/// A grid counts as visibly turned when its angle exceeds 1° somewhere.
const TURNED: f64 = std::f64::consts::PI / 180.0;

/// [`Resolved`] per pair path, with the pair it was resolved for.
pub(crate) type ResolvedCache = HashMap<String, (VectorPair, Arc<Resolved>)>;

/// Which component points average onto centre point `i` along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Point `i` itself (not staggered on this axis).
    Same,
    /// Points `i` and `i + 1`.
    Plus,
    /// Points `i - 1` and `i`.
    Minus,
}

/// Why the pair is or is not rotated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Already east/north: eastward/northward names or a regular lat/lon grid.
    Geographic,
    /// Grid-relative by the names, a rotated-pole mapping or an angle variable.
    Known,
    /// Undeclared on a visibly turned 2-D grid: assumed grid-relative.
    Likely,
    /// Undeclared on a grid that is not visibly turned: assumed east/north.
    Unknown,
}

/// How a pair is read: staggering, rotation and centre coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    /// `[east, north][y, x]`.
    pub sides: [[Side; 2]; 2],
    /// Radians per centre (y, x) cell, row-major; `None` = not rotated.
    pub angle: Option<Arc<Vec<f32>>>,
    pub basis: Basis,
    pub rotated: bool,
    /// Centre coordinates: 1-D each when `regular`, else (y, x) row-major;
    /// empty when unknown.
    pub lat: Arc<Vec<f64>>,
    pub lon: Arc<Vec<f64>>,
    pub regular: bool,
    /// Inspector copy that applies.
    pub notes: Vec<String>,
}

/// Which side the component sits on along one axis.
///
/// With positions (component and centre coordinates along the axis, in any
/// monotonic unit), component point `i` after centre point `i` gives
/// `Minus`, before gives `Plus`, and within a quarter cell gives `Same` when
/// the lengths are equal. Otherwise the lengths decide: one longer gives
/// `Plus`, one shorter or equal (the NEMO convention) gives `Minus`.
pub fn stagger_side(
    comp_len: usize,
    centre_len: usize,
    comp_pos: Option<&[f64]>,
    centre_pos: Option<&[f64]>,
) -> Side {
    let by_len = if comp_len == centre_len + 1 { Side::Plus } else { Side::Minus };
    let offset = match (comp_pos, centre_pos) {
        (Some(c), Some(t)) => offset_cells(c, t),
        _ => None,
    };
    match offset {
        Some(d) if d.abs() < 0.25 => {
            if comp_len == centre_len {
                Side::Same
            } else {
                by_len
            }
        }
        Some(d) if d > 0.0 => Side::Minus,
        Some(_) => Side::Plus,
        None => by_len,
    }
}

/// Median offset of component point `i` from centre point `i`, in local
/// centre cells (positive = further along the axis); `None` when no point
/// gives a finite offset.
fn offset_cells(comp: &[f64], centre: &[f64]) -> Option<f64> {
    let n = centre.len();
    if n < 2 {
        return None;
    }
    let mut d: Vec<f64> = comp
        .iter()
        .zip(centre)
        .enumerate()
        .map(|(i, (c, t))| {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
            let cell = (centre[b] - centre[a]) / (b - a) as f64;
            (c - t) / cell
        })
        .filter(|x| x.is_finite())
        .collect();
    if d.is_empty() {
        return None;
    }
    d.sort_by(f64::total_cmp);
    Some(d[d.len() / 2])
}

/// Grid angle (radians, counter-clockwise from east to the +x grid
/// direction) per cell of a (y, x) row-major lat/lon grid: centred
/// differences along x, one-sided at the edges, longitude differences
/// wrapped to [-180, 180).
pub fn angle_from_latlon(lat: &[f64], lon: &[f64], ny: usize, nx: usize) -> Vec<f32> {
    let get = |v: &[f64], k: usize| v.get(k).copied().unwrap_or(f64::NAN);
    angle_with(ny, nx, |j, i| (get(lat, j * nx + i), get(lon, j * nx + i)))
}

fn angle_with(ny: usize, nx: usize, at: impl Fn(usize, usize) -> (f64, f64)) -> Vec<f32> {
    let mut out = Vec::with_capacity(ny * nx);
    for j in 0..ny {
        for i in 0..nx {
            if nx < 2 {
                out.push(0.0);
                continue;
            }
            let (a, b) = (i.saturating_sub(1), (i + 1).min(nx - 1));
            let ((lat_a, lon_a), (lat_b, lon_b), (lat_c, _)) = (at(j, a), at(j, b), at(j, i));
            let dx = wrap_lon(lon_b - lon_a) * lat_c.to_radians().cos();
            out.push((lat_b - lat_a).atan2(dx) as f32);
        }
    }
    out
}

/// The two component indices bracketing centre index `ci` (equal for
/// `Same`), or `None` when one falls outside the component's `len`.
fn taps(side: Side, ci: usize, len: usize) -> Option<(usize, usize)> {
    let (a, b) = match side {
        Side::Same => (ci, ci),
        Side::Plus => (ci, ci + 1),
        Side::Minus => (ci.checked_sub(1)?, ci),
    };
    (b < len).then_some((a, b))
}

/// Latitude/longitude of a horizontal grid: 1-D each when `regular`, else
/// (y, x) row-major.
#[derive(Clone)]
struct Plane {
    lat: Arc<Vec<f64>>,
    lon: Arc<Vec<f64>>,
    ny: usize,
    nx: usize,
    regular: bool,
}

impl Plane {
    /// (lat, lon) at (j, i); NaN outside the grid.
    fn at(&self, j: usize, i: usize) -> (f64, f64) {
        let get = |v: &[f64], k: usize| v.get(k).copied().unwrap_or(f64::NAN);
        if self.regular {
            (get(&self.lat, j), get(&self.lon, i))
        } else if j < self.ny && i < self.nx {
            let k = j * self.nx + i;
            (get(&self.lat, k), get(&self.lon, k))
        } else {
            (f64::NAN, f64::NAN)
        }
    }

    /// This grid's coordinates averaged onto an `ny` × `nx` centre grid,
    /// each centre point from its bracketing points (longitudes unwrapped
    /// first); NaN where a bracketing point is missing.
    fn onto_centre(&self, sides: [Side; 2], ny: usize, nx: usize) -> Plane {
        let (mut lat, mut lon) = (Vec::new(), Vec::new());
        if self.regular {
            lat.extend((0..ny).map(|j| {
                taps(sides[0], j, self.ny).map_or(f64::NAN, |(a, b)| (self.at(a, 0).0 + self.at(b, 0).0) / 2.0)
            }));
            lon.extend((0..nx).map(|i| {
                taps(sides[1], i, self.nx).map_or(f64::NAN, |(a, b)| mean_lon(&[self.at(0, a).1, self.at(0, b).1]))
            }));
        } else {
            for j in 0..ny {
                for i in 0..nx {
                    let (y, x) = (taps(sides[0], j, self.ny), taps(sides[1], i, self.nx));
                    let (Some((ja, jb)), Some((ia, ib))) = (y, x) else {
                        lat.push(f64::NAN);
                        lon.push(f64::NAN);
                        continue;
                    };
                    let pts = [self.at(ja, ia), self.at(ja, ib), self.at(jb, ia), self.at(jb, ib)];
                    lat.push(pts.iter().map(|p| p.0).sum::<f64>() / 4.0);
                    lon.push(mean_lon(&pts.map(|p| p.1)));
                }
            }
        }
        Plane { lat: Arc::new(lat), lon: Arc::new(lon), ny, nx, regular: self.regular }
    }
}

/// Mean of longitudes, each unwrapped against the first.
fn mean_lon(v: &[f64]) -> f64 {
    let a = v[0];
    a + v.iter().map(|&x| wrap_lon(x - a)).sum::<f64>() / v.len() as f64
}

/// `a` with its missing points filled from `b` when both have the same layout.
fn merge(a: Option<Plane>, b: Option<Plane>) -> Option<Plane> {
    match (a, b) {
        (Some(mut a), Some(b)) if a.regular == b.regular && a.lat.len() == b.lat.len() && a.lon.len() == b.lon.len() => {
            for (x, y) in [(&mut a.lat, &b.lat), (&mut a.lon, &b.lon)] {
                for (p, q) in Arc::make_mut(x).iter_mut().zip(y.iter()) {
                    if p.is_nan() {
                        *p = *q;
                    }
                }
            }
            Some(a)
        }
        (a, b) => a.or(b),
    }
}

/// Positions along axis `axis` (0 = y, 1 = x) for [`stagger_side`]: the
/// centre points along the middle line of the other axis are `0, 1, ..`
/// and each component point `i` is centre point `i` plus its offset,
/// projected on the local axis direction, in local cells.
fn positions(comp: &Plane, centre: &Plane, axis: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let (tn, pn, tm, pm) = if axis == 1 {
        (centre.nx, comp.nx, centre.ny / 2, (centre.ny / 2).min(comp.ny.checked_sub(1)?))
    } else {
        (centre.ny, comp.ny, centre.nx / 2, (centre.nx / 2).min(comp.nx.checked_sub(1)?))
    };
    if tn < 2 {
        return None;
    }
    let t_at = |i: usize| if axis == 1 { centre.at(tm, i) } else { centre.at(i, tm) };
    let p_at = |i: usize| if axis == 1 { comp.at(pm, i) } else { comp.at(i, pm) };
    let comp_pos = (0..tn.min(pn))
        .map(|i| {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(tn - 1));
            // Local plane at centre point i: x east, y north, in degrees.
            let (lat0, lon0) = t_at(i);
            let k = lat0.to_radians().cos();
            let xy = |(lat, lon): (f64, f64)| (wrap_lon(lon - lon0) * k, lat - lat0);
            let (pa, pb, q) = (xy(t_at(a)), xy(t_at(b)), xy(p_at(i)));
            let (wx, wy) = (pb.0 - pa.0, pb.1 - pa.1);
            // (q · w / |w|) / (|w| / (b - a)): the offset in local cells.
            i as f64 + (q.0 * wx + q.1 * wy) * (b - a) as f64 / (wx * wx + wy * wy)
        })
        .collect();
    Some((comp_pos, (0..tn).map(|i| i as f64).collect()))
}

/// Calls `f(flat, index)` for every index of `shape`, row-major.
fn odometer(shape: &[usize], mut f: impl FnMut(usize, &[usize])) {
    let total: usize = shape.iter().product();
    let mut idx = vec![0; shape.len()];
    for flat in 0..total {
        f(flat, &idx);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

impl NcFile {
    /// Staggering, rotation and centre coordinates of `pair`, worked out on
    /// first use (reading coordinates and angles) and cached per pair path.
    pub fn resolve_vector(&self, pair: &VectorPair) -> Result<Arc<Resolved>, CoreError> {
        let cache = || self.resolved.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((p, r)) = cache().get(&pair.path)
            && p == pair
        {
            return Ok(r.clone());
        }
        // Resolved outside the lock; a racing resolve gives the same result.
        let r = Arc::new(self.resolve_uncached(pair)?);
        cache().insert(pair.path.clone(), (pair.clone(), r.clone()));
        Ok(r)
    }

    /// Drops the cached resolution of the pair at `path`.
    pub(crate) fn forget_resolved(&self, path: &str) {
        self.resolved.lock().unwrap_or_else(|e| e.into_inner()).remove(path);
    }

    fn resolve_uncached(&self, pair: &VectorPair) -> Result<Resolved, CoreError> {
        let unsupported = |msg: &str| CoreError::Unsupported(format!("{}: {msg}", pair.path));
        let comps = [self.numeric_var(&pair.east)?, self.numeric_var(&pair.north)?];
        let shape = &pair.centre_shape;
        let rank = shape.len();
        let [hy, hx] = pair.horiz;
        if hy >= rank || hx >= rank || hy == hx || pair.centre_dims.len() != rank {
            return Err(unsupported("the centre grid is malformed"));
        }
        for c in comps {
            if c.shape.len() != rank {
                return Err(unsupported("components and centre grid differ in rank"));
            }
            if (0..rank).any(|k| k != hy && k != hx && c.shape[k] != shape[k]) {
                return Err(unsupported("components differ from the centre grid off the horizontal axes"));
            }
        }
        let (cy, cx) = (shape[hy], shape[hx]);
        let centre = match (&pair.centre_lat, &pair.centre_lon) {
            (Some(lat), Some(lon)) => self.coord_plane(&pair.var_info(), pair.horiz, lat, lon)?,
            _ => None,
        };
        let planes = [self.geo_plane(comps[0], pair.horiz)?, self.geo_plane(comps[1], pair.horiz)?];

        let mut assumed = false;
        // [east, north][y, x]: from coordinates when both sides have them,
        // else from lengths; equal lengths without coordinates are staggered
        // only along the component's own axis (east along x, north along y).
        let mut side = |c: usize, a: usize| {
            let k = pair.horiz[a];
            let (cl, tl) = (comps[c].shape[k], shape[k]);
            let pos = match (&planes[c], &centre) {
                (Some(p), Some(t)) => positions(p, t, a).filter(|(p, t)| offset_cells(p, t).is_some()),
                _ => None,
            };
            match pos {
                Some((p, t)) => stagger_side(cl, tl, Some(&p), Some(&t)),
                None if cl != tl => stagger_side(cl, tl, None, None),
                None if a != c => {
                    assumed = true;
                    stagger_side(cl, tl, None, None)
                }
                None => Side::Same,
            }
        };
        let sides = [[side(0, 0), side(0, 1)], [side(1, 0), side(1, 1)]];

        let centre = centre.or_else(|| {
            let avg = |c: usize| planes[c].as_ref().map(|p| p.onto_centre(sides[c], cy, cx));
            merge(avg(0), avg(1))
        });
        let regular = centre.as_ref().is_some_and(|p| p.regular);
        let latlon_angle = || centre.as_ref().map(|p| angle_with(cy, cx, |j, i| p.at(j, i)));
        let (yd, xd) = (&pair.centre_dims[hy], &pair.centre_dims[hx]);
        let source_angle = || -> Result<Option<Vec<f32>>, CoreError> {
            Ok(match &pair.angle {
                AngleSource::Angle(path) => {
                    Some(self.read_horizontal(path, yd, xd)?.iter().map(|&a| a as f32).collect())
                }
                AngleSource::CosSin(cos, sin) => {
                    let (cos, sin) = (self.read_horizontal(cos, yd, xd)?, self.read_horizontal(sin, yd, xd)?);
                    Some(cos.iter().zip(&sin).map(|(c, s)| s.atan2(*c) as f32).collect())
                }
                AngleSource::FromLatLon => latlon_angle(),
            })
        };
        let (basis, auto, angle) = if pair.frame == Frame::Geographic || regular {
            (Basis::Geographic, false, None)
        } else if pair.frame == Frame::GridRelative || pair.angle != AngleSource::FromLatLon {
            (Basis::Known, true, source_angle()?)
        } else {
            let a = latlon_angle();
            let turned = a.as_ref().is_some_and(|a| a.iter().any(|x| f64::from(*x).abs() > TURNED));
            if turned { (Basis::Likely, true, a) } else { (Basis::Unknown, false, a) }
        };
        let rotate = match pair.rotate {
            RotateChoice::Auto => auto,
            RotateChoice::On => true,
            RotateChoice::Off => false,
        };
        let angle = match (rotate, angle) {
            (false, _) => None,
            (true, Some(a)) => Some(a),
            (true, None) => source_angle()?,
        };
        if angle.as_ref().is_some_and(|a| a.len() != cy * cx) {
            return Err(unsupported("the grid angle does not match the centre grid"));
        }

        let mut notes = Vec::new();
        match basis {
            Basis::Likely => notes.push(NOTE_LIKELY.to_string()),
            Basis::Unknown => notes.push(NOTE_UNKNOWN.to_string()),
            Basis::Geographic | Basis::Known => {}
        }
        if assumed {
            notes.push(NOTE_STAGGER.to_string());
        }
        let (lat, lon) = centre.map_or_else(Default::default, |p| (p.lat, p.lon));
        Ok(Resolved {
            sides,
            rotated: angle.is_some(),
            angle: angle.map(Arc::new),
            basis,
            lat,
            lon,
            regular,
            notes,
        })
    }

    /// The lat/lon plane of a component's own coordinates, if it has any.
    fn geo_plane(&self, v: &VarInfo, horiz: [usize; 2]) -> Result<Option<Plane>, CoreError> {
        match detect_geo(&self.meta, v) {
            Geo::Regular { lat_var, lon_var, .. } | Geo::Curvilinear { lat_var, lon_var, .. } => {
                self.coord_plane(v, horiz, &lat_var, &lon_var)
            }
            Geo::None => Ok(None),
        }
    }

    /// The `lat`/`lon` plane over `v`'s horizontal dims (`horiz`, positions
    /// in `v.dims`); `None` when the coordinates do not lie on those dims.
    fn coord_plane(
        &self,
        v: &VarInfo,
        [hy, hx]: [usize; 2],
        lat: &str,
        lon: &str,
    ) -> Result<Option<Plane>, CoreError> {
        let (Some(yd), Some(xd)) = (v.dims.get(hy), v.dims.get(hx)) else { return Ok(None) };
        let (ny, nx) = (v.shape[hy], v.shape[hx]);
        let (Some(la), Some(lo)) = (self.meta.var(lat), self.meta.var(lon)) else { return Ok(None) };
        let plane = |lat: Vec<f64>, lon: Vec<f64>, regular| {
            Some(Plane { lat: Arc::new(lat), lon: Arc::new(lon), ny, nx, regular })
        };
        if la.dims.len() == 1 && lo.dims.len() == 1 {
            if la.dims[0] != *yd || lo.dims[0] != *xd {
                return Ok(None);
            }
            let (lat, lon) = (self.read_coord(lat)?, self.read_coord(lon)?);
            let fits = lat.len() == ny && lon.len() == nx;
            return Ok(if fits { plane(lat, lon, true) } else { None });
        }
        let on_plane = |c: &VarInfo| c.dims.contains(yd) && c.dims.contains(xd);
        if !on_plane(la) || !on_plane(lo) {
            return Ok(None);
        }
        let (lat, lon) = (self.read_horizontal(lat, yd, xd)?, self.read_horizontal(lon, yd, xd)?);
        let fits = lat.len() == ny * nx && lon.len() == ny * nx;
        Ok(if fits { plane(lat, lon, false) } else { None })
    }

    /// Reads a pair slice (`r` in centre-grid index space): the magnitude in
    /// `data`, east/north in `vector`. A centre point with any bracketing
    /// component point missing is missing as a whole.
    pub(crate) fn read_pair(&self, pair: &VectorPair, r: &SliceRequest) -> Result<Slice, CoreError> {
        let (shape, total) = check_shape(r, &pair.centre_shape)?;
        let res = self.resolve_vector(pair)?;
        let east_len = self.numeric_var(&pair.east)?.shape.clone();
        let north_len = self.numeric_var(&pair.north)?.shape.clone();
        let mut east = self.read_component(&pair.east, &east_len, r, &shape, pair.horiz, res.sides[0])?;
        let mut north = self.read_component(&pair.north, &north_len, r, &shape, pair.horiz, res.sides[1])?;
        let [hy, hx] = pair.horiz;
        let cx = pair.centre_shape[hx];
        let mut data = vec![f32::NAN; total];
        odometer(&shape, |k, idx| {
            let (u, v) = (f64::from(east[k]), f64::from(north[k]));
            let (mut e, mut n) = (u, v);
            if let Some(angle) = &res.angle {
                let j = r.start[hy] + idx[hy] * r.stride[hy];
                let i = r.start[hx] + idx[hx] * r.stride[hx];
                let (s, c) = angle.get(j * cx + i).map_or(f64::NAN, |a| f64::from(*a)).sin_cos();
                (e, n) = (u * c - v * s, u * s + v * c);
            }
            if e.is_nan() || n.is_nan() {
                (e, n) = (f64::NAN, f64::NAN);
            }
            east[k] = e as f32;
            north[k] = n as f32;
            data[k] = e.hypot(n) as f32;
        });
        let valid_count = data.iter().filter(|v| !v.is_nan()).count();
        Ok(Slice { shape, data, valid_count, vector: Some(VectorParts { east, north }) })
    }

    /// One component on the centre-grid request `r` (output `shape`): read
    /// through `read_slice` with each staggered axis at full resolution and
    /// one neighbour further (clamped to the component), then each centre
    /// point averages its bracketing points. Missing if any is missing or
    /// outside the component.
    fn read_component(
        &self,
        path: &str,
        comp_shape: &[usize],
        r: &SliceRequest,
        shape: &[usize],
        [hy, hx]: [usize; 2],
        sides: [Side; 2],
    ) -> Result<Vec<f32>, CoreError> {
        let n = shape.len();
        let total: usize = shape.iter().product();
        let mut w = SliceRequest { var: path.into(), ..r.clone() };
        // Per axis, the window index of each output index's lower bracketing
        // point (`None` = missing); the staggered axes' window strides.
        let mut lower: Vec<Vec<Option<usize>>> = Vec::with_capacity(n);
        let mut staggered = Vec::new();
        for k in 0..n {
            let side = if k == hy { sides[0] } else if k == hx { sides[1] } else { Side::Same };
            let len = comp_shape[k];
            if side == Side::Same {
                if r.start[k] + r.count[k] > len {
                    return Err(CoreError::Read {
                        var: r.var.clone(),
                        msg: format!("{path} is shorter than the centre grid on dimension {k}"),
                    });
                }
                lower.push((0..shape[k]).map(Some).collect());
                continue;
            }
            let centres = || (0..shape[k]).map(|o| r.start[k] + o * r.stride[k]);
            let lows: Vec<Option<usize>> = centres().map(|ci| taps(side, ci, len).map(|t| t.0)).collect();
            let (Some(w0), Some(w1)) = (lows.iter().flatten().min(), lows.iter().flatten().max()) else {
                return Ok(vec![f32::NAN; total]);
            };
            let (w0, w1) = (*w0, *w1 + 1);
            w.start[k] = w0;
            w.count[k] = w1 - w0 + 1;
            w.stride[k] = 1;
            lower.push(lows.iter().map(|l| l.map(|l| l - w0)).collect());
            staggered.push(k);
        }
        let win = self.read_slice(&w)?;
        let mut wstride = vec![1; n];
        for k in (0..n.saturating_sub(1)).rev() {
            wstride[k] = wstride[k + 1] * win.shape[k + 1];
        }
        // Offsets of the 1, 2 or 4 bracketing points from the lower one.
        let mut offsets = vec![0usize];
        for &k in &staggered {
            let more: Vec<usize> = offsets.iter().map(|o| o + wstride[k]).collect();
            offsets.extend(more);
        }
        let mut out = vec![f32::NAN; total];
        odometer(shape, |flat, idx| {
            let base = (0..n).try_fold(0, |b, k| lower[k][idx[k]].map(|l| b + l * wstride[k]));
            if let Some(base) = base {
                let sum: f64 = offsets
                    .iter()
                    .map(|o| win.data.get(base + o).map_or(f64::NAN, |v| f64::from(*v)))
                    .sum();
                out[flat] = (sum / offsets.len() as f64) as f32;
            }
        });
        Ok(out)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{self, vectors};
    use crate::fixtures::vectors::{known_east, known_north};
    use crate::{NcFile, RotateChoice, SeriesRequest, Slice, SliceRequest, SliceService};
    use std::sync::Arc;

    const LIKELY: &str = "Components assumed to follow the grid and rotated to east/north. \
                          Turn off if they already point east/north.";
    const UNKNOWN: &str = "Components assumed to point east/north (the grid is not visibly \
                           turned). Turn on rotation if they follow the grid.";

    /// The whole horizontal plane of pair `var` at index `t` of the leading dims.
    fn plane(f: &NcFile, var: &str, t: usize) -> SliceRequest {
        let p = f.pair(var).expect("registered pair");
        let n = p.centre_shape.len();
        let [hy, hx] = p.horiz;
        let mut start = vec![t; n];
        let mut count = vec![1; n];
        for k in [hy, hx] {
            start[k] = 0;
            count[k] = p.centre_shape[k];
        }
        SliceRequest { var: var.into(), start, count, stride: vec![1; n] }
    }

    fn nan_eq(a: f32, b: f32) -> bool {
        (a.is_nan() && b.is_nan()) || a == b
    }

    /// East/north match the known field at centre (j, i) within `tol`
    /// (east, north) wherever `missing` is false; elsewhere the whole vector
    /// and the magnitude are missing.
    fn check_known(s: &Slice, ny: usize, nx: usize, tol: (f64, f64), missing: impl Fn(usize, usize) -> bool) {
        let v = s.vector.as_ref().expect("vector parts");
        assert_eq!(s.shape[s.shape.len() - 2..], [ny, nx]);
        assert_eq!(s.data.len(), ny * nx);
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if missing(j, i) {
                    assert!(s.data[k].is_nan(), "({j},{i}) magnitude should be missing");
                    assert!(v.east[k].is_nan() && v.north[k].is_nan(), "({j},{i}) half a vector");
                    continue;
                }
                let (e, n) = (v.east[k] as f64, v.north[k] as f64);
                let (we, wn) = (known_east(i as f64), known_north(j as f64));
                assert!((e - we).abs() < tol.0, "east ({j},{i}): {e} vs {we}");
                assert!((n - wn).abs() < tol.1, "north ({j},{i}): {n} vs {wn}");
                assert!((s.data[k] as f64 - e.hypot(n)).abs() < 1e-4, "magnitude ({j},{i})");
            }
        }
        assert_eq!(s.valid_count, s.data.iter().filter(|x| !x.is_nan()).count());
    }

    #[test]
    fn stagger_side_rules() {
        assert_eq!(stagger_side(15, 14, None, None), Side::Plus);
        assert_eq!(stagger_side(15, 16, None, None), Side::Minus);
        assert_eq!(stagger_side(20, 20, None, None), Side::Minus);
        let centre: Vec<f64> = (0..20).map(f64::from).collect();
        let after: Vec<f64> = (0..20).map(|i| f64::from(i) + 0.5).collect();
        assert_eq!(stagger_side(20, 20, Some(&after), Some(&centre)), Side::Minus);
        let c14: Vec<f64> = (0..14).map(f64::from).collect();
        let before: Vec<f64> = (0..15).map(|i| f64::from(i) - 0.5).collect();
        assert_eq!(stagger_side(15, 14, Some(&before), Some(&c14)), Side::Plus);
        // Within a quarter cell with equal lengths: the same points.
        let near: Vec<f64> = (0..20).map(|i| f64::from(i) + 0.1).collect();
        assert_eq!(stagger_side(20, 20, Some(&near), Some(&centre)), Side::Same);
        // Positions decreasing along the axis (a south-to-north flipped grid).
        let down: Vec<f64> = (0..20).map(|i| -f64::from(i)).collect();
        let down_after: Vec<f64> = (0..20).map(|i| -f64::from(i) - 0.5).collect();
        assert_eq!(stagger_side(20, 20, Some(&down_after), Some(&down)), Side::Minus);
    }

    #[test]
    fn angle_from_latlon_turned_grid() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_noclue(d.path(), true)).unwrap();
        let lat = f.read_horizontal("/lat", "y", "x").unwrap();
        let lon = f.read_horizontal("/lon", "y", "x").unwrap();
        let a = angle_from_latlon(&lat, &lon, 15, 20);
        assert_eq!(a.len(), 15 * 20);
        let want = 20f64.to_radians();
        for j in 1..14 {
            for i in 1..19 {
                let got = a[j * 20 + i] as f64;
                assert!((got - want).abs() < 0.005, "({j},{i}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn angle_from_latlon_across_dateline() {
        let (ny, nx, theta) = (3, 4, 10f64.to_radians());
        let (s, c) = theta.sin_cos();
        let (mut lat, mut lon) = (Vec::new(), Vec::new());
        for j in 0..ny {
            for i in 0..nx {
                let (p, q) = (i as f64 - 1.5, j as f64 - 1.0);
                let la = 60.0 + 0.5 * (p * s + q * c);
                lat.push(la);
                lon.push(crate::cf::wrap_lon(180.0 + 0.5 * (p * c - q * s) / la.to_radians().cos()));
            }
        }
        assert!(lon.iter().any(|&l| l > 179.0) && lon.iter().any(|&l| l < -179.0));
        for (k, a) in angle_from_latlon(&lat, &lon, ny, nx).iter().enumerate() {
            assert!((*a as f64 - theta).abs() < 0.01, "cell {k}: {a}");
        }
    }

    #[test]
    fn nemo_known_field() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_nemo_like(d.path())).unwrap();
        assert!(!f.fast_eligible("/uo+vo"));
        let s = f.read_slice(&plane(&f, "/uo+vo", 0)).unwrap();
        check_known(&s, 20, 30, (0.01, 0.01), |j, i| i == 0 || j == 0 || (j == 5 && (i == 10 || i == 11)));
        let r = f.resolve_vector(&f.pair("/uo+vo").unwrap()).unwrap();
        assert_eq!(r.basis, Basis::Likely);
        assert!(r.rotated);
        assert!(r.angle.is_some());
        assert!(r.notes.iter().any(|n| n == LIKELY), "{:?}", r.notes);
        assert_eq!(r.sides, [[Side::Same, Side::Minus], [Side::Minus, Side::Same]]);
        assert!(!r.regular);
        assert_eq!((r.lat.len(), r.lon.len()), (600, 600));
    }

    #[test]
    fn nemo_land_neighbour_missing() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_nemo_like(d.path())).unwrap();
        let s = f.read_slice(&plane(&f, "/uo+vo", 1)).unwrap();
        let v = s.vector.as_ref().unwrap();
        for i in [10, 11] {
            let k = 5 * 30 + i;
            assert!(s.data[k].is_nan(), "(5,{i})");
            assert!(v.east[k].is_nan() && v.north[k].is_nan(), "(5,{i}) half a vector");
        }
        assert!(!s.data[5 * 30 + 12].is_nan());
    }

    fn roms_edge(j: usize, i: usize) -> bool {
        i == 0 || i == 15 || j == 0 || j == 11
    }

    #[test]
    fn roms_edges_missing() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_roms_like(d.path())).unwrap();
        let s = f.read_slice(&plane(&f, "/u+v", 0)).unwrap();
        let v = s.vector.as_ref().unwrap();
        for j in 0..12 {
            for i in 0..16 {
                let k = j * 16 + i;
                assert_eq!(s.data[k].is_nan(), roms_edge(j, i), "({j},{i})");
                assert_eq!(v.east[k].is_nan(), roms_edge(j, i), "east ({j},{i})");
                assert_eq!(v.north[k].is_nan(), roms_edge(j, i), "north ({j},{i})");
            }
        }
    }

    #[test]
    fn roms_known_field() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_roms_like(d.path())).unwrap();
        let s = f.read_slice(&plane(&f, "/u+v", 1)).unwrap();
        check_known(&s, 12, 16, (0.01, 0.01), roms_edge);
        let r = f.resolve_vector(&f.pair("/u+v").unwrap()).unwrap();
        assert_eq!(r.basis, Basis::Known);
        assert!(r.rotated);
        assert_eq!(r.sides, [[Side::Same, Side::Minus], [Side::Minus, Side::Same]]);
        assert!(r.notes.is_empty(), "{:?}", r.notes);
    }

    #[test]
    fn wrf_known_field() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_wrf_like(d.path())).unwrap();
        let s = f.read_slice(&plane(&f, "/U+V", 0)).unwrap();
        check_known(&s, 10, 14, (0.01, 0.01), |_, _| false);
        let r = f.resolve_vector(&f.pair("/U+V").unwrap()).unwrap();
        assert_eq!(r.sides, [[Side::Same, Side::Plus], [Side::Plus, Side::Same]]);
        assert_eq!(r.basis, Basis::Known);
        assert!(r.rotated);
    }

    #[test]
    fn noclue_flat_unknown() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_noclue(d.path(), false)).unwrap();
        let s = f.read_slice(&plane(&f, "/u+v", 0)).unwrap();
        let v = s.vector.as_ref().unwrap();
        let raw = |var: &str| {
            let r = SliceRequest { var: var.into(), start: vec![0, 0], count: vec![15, 20], stride: vec![1, 1] };
            f.read_slice(&r).unwrap().data
        };
        // Not rotated: east/north are exactly the raw components.
        assert!(raw("/u").iter().zip(&v.east).all(|(a, b)| nan_eq(*a, *b)));
        assert!(raw("/v").iter().zip(&v.north).all(|(a, b)| nan_eq(*a, *b)));
        // u ≈ E.
        for j in 0..15 {
            for i in 0..20 {
                let (e, we) = (v.east[j * 20 + i] as f64, known_east(i as f64));
                assert!((e - we).abs() < 0.01, "east ({j},{i}): {e} vs {we}");
            }
        }
        let r = f.resolve_vector(&f.pair("/u+v").unwrap()).unwrap();
        assert_eq!(r.basis, Basis::Unknown);
        assert!(!r.rotated);
        assert!(r.angle.is_none());
        assert!(r.notes.iter().any(|n| n == UNKNOWN), "{:?}", r.notes);
        assert_eq!(r.sides, [[Side::Same; 2]; 2]);
    }

    #[test]
    fn noclue_turned_likely() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_noclue(d.path(), true)).unwrap();
        let s = f.read_slice(&plane(&f, "/u+v", 0)).unwrap();
        check_known(&s, 15, 20, (0.01, 0.01), |_, _| false);
        let r = f.resolve_vector(&f.pair("/u+v").unwrap()).unwrap();
        assert_eq!(r.basis, Basis::Likely);
        assert!(r.rotated);
        assert!(r.notes.iter().any(|n| n == LIKELY), "{:?}", r.notes);
    }

    #[test]
    fn rotate_off_overrides() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_nemo_like(d.path())).unwrap();
        let off = f.add_pair(f.pair("/uo+vo").unwrap().with_rotate(RotateChoice::Off));
        assert_eq!(off.path, "/uo+vo#off");
        let s = f.read_slice(&plane(&f, &off.path, 0)).unwrap();
        let r = f.resolve_vector(&off).unwrap();
        assert!(!r.rotated);
        assert!(r.angle.is_none());
        assert_eq!(r.basis, Basis::Likely, "the basis is kept for the notes");
        let uo = f
            .read_slice(&SliceRequest {
                var: "/uo".into(),
                start: vec![0, 0, 0],
                count: vec![1, 20, 30],
                stride: vec![1, 1, 1],
            })
            .unwrap()
            .data;
        let east = &s.vector.as_ref().unwrap().east;
        for j in 1..20 {
            for i in 1..30 {
                let k = j * 30 + i;
                let want = (uo[k - 1] as f64 + uo[k] as f64) / 2.0;
                if want.is_nan() {
                    assert!(east[k].is_nan(), "({j},{i})");
                } else {
                    assert!((east[k] as f64 - want).abs() < 1e-5, "({j},{i}): {} vs {want}", east[k]);
                }
            }
        }
    }

    #[test]
    fn rotate_on_overrides_unknown() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_noclue(d.path(), false)).unwrap();
        let on = f.add_pair(f.pair("/u+v").unwrap().with_rotate(RotateChoice::On));
        assert_eq!(on.path, "/u+v#on");
        let r = f.resolve_vector(&on).unwrap();
        assert!(r.rotated);
        assert!(r.angle.is_some());
        assert_eq!(r.basis, Basis::Unknown, "the basis is kept for the notes");
        let all = |var: &str| SliceRequest { var: var.into(), start: vec![0, 0], count: vec![15, 20], stride: vec![1, 1] };
        let s = f.read_slice(&all(&on.path)).unwrap();
        let u = f.read_slice(&all("/u")).unwrap().data;
        let east = &s.vector.as_ref().unwrap().east;
        // θ = 0.005 rad turns every vector a little: east is no longer raw u.
        assert!(u.iter().zip(east).any(|(a, b)| (a - b).abs() > 1e-4), "east equals raw u");
    }

    /// Strided reads of pair `var` (centre plane `ny` × `nx`) at time 1 equal
    /// the full plane subsampled, for strides (`sty`, `stx`) and `starts`.
    fn check_strided(f: &NcFile, var: &str, (ny, nx): (usize, usize), (sty, stx): (usize, usize), starts: &[(usize, usize)]) {
        let full = f.read_slice(&plane(f, var, 1)).unwrap();
        let fv = full.vector.as_ref().unwrap();
        for &(sy, sx) in starts {
            let r = SliceRequest {
                var: var.into(),
                start: vec![1, sy, sx],
                count: vec![1, ny - sy, nx - sx],
                stride: vec![1, sty, stx],
            };
            let s = f.read_slice(&r).unwrap();
            let (cy, cx) = ((ny - sy).div_ceil(sty), (nx - sx).div_ceil(stx));
            assert_eq!(s.shape, vec![1, cy, cx], "{var} start ({sy},{sx})");
            let v = s.vector.as_ref().unwrap();
            for a in 0..cy {
                for b in 0..cx {
                    let (k, kf) = (a * cx + b, (sy + sty * a) * nx + sx + stx * b);
                    assert!(nan_eq(s.data[k], full.data[kf]), "{var} start ({sy},{sx}) at ({a},{b})");
                    assert!(nan_eq(v.east[k], fv.east[kf]), "{var} east ({sy},{sx}) at ({a},{b})");
                    assert!(nan_eq(v.north[k], fv.north[kf]), "{var} north ({sy},{sx}) at ({a},{b})");
                }
            }
        }
    }

    #[test]
    fn strided_vector_equals_subsampled_full() {
        let d = tempfile::tempdir().unwrap();
        // ROMS: both components on the minus side, windows [i−1, i].
        let f = NcFile::open(&vectors::write_roms_like(d.path())).unwrap();
        // Starts chosen so the samples reach the last row (11) and column (15).
        check_strided(&f, "/u+v", (12, 16), (3, 4), &[(2, 3), (0, 0), (1, 1), (11, 15)]);
        // WRF: both components on the plus side, windows [i, i+1], whose
        // `count + 1` read ends at the staggered component's last row/column.
        let f = NcFile::open(&vectors::write_wrf_like(d.path())).unwrap();
        // 10 × 14: with strides 3 and 4, starts (0, 1), (9, 13) and (3, 5)
        // reach the last row (9) and column (13); stride 1 reads every cell
        // and strides (9, 13) jump straight to the last ones.
        check_strided(&f, "/U+V", (10, 14), (3, 4), &[(0, 1), (1, 3), (9, 13), (0, 0), (3, 5)]);
        check_strided(&f, "/U+V", (10, 14), (1, 1), &[(0, 0), (2, 2), (9, 13)]);
        check_strided(&f, "/U+V", (10, 14), (9, 13), &[(0, 0)]);
    }

    #[test]
    fn cmems_like_pair_unrotated() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
        let req = |var: &str| SliceRequest {
            var: var.into(),
            start: vec![0, 0, 0],
            count: vec![1, 10, 20],
            stride: vec![1, 1, 1],
        };
        let s = f.read_slice(&req("/ocean/uo+vo")).unwrap();
        let uo = f.read_slice(&req("/ocean/uo")).unwrap().data;
        let vo = f.read_slice(&req("/ocean/vo")).unwrap().data;
        let v = s.vector.as_ref().unwrap();
        assert!(uo.iter().zip(&v.east).all(|(a, b)| nan_eq(*a, *b)));
        assert!(vo.iter().zip(&v.north).all(|(a, b)| nan_eq(*a, *b)));
        for k in 0..200 {
            assert!(nan_eq(s.data[k], uo[k].hypot(vo[k])), "cell {k}");
        }
        let r = f.resolve_vector(&f.pair("/ocean/uo+vo").unwrap()).unwrap();
        assert_eq!(r.basis, Basis::Geographic);
        assert!(!r.rotated && r.regular);
        assert!(r.notes.is_empty());
    }

    #[test]
    fn vector_series_has_components() {
        let d = tempfile::tempdir().unwrap();
        let f = Arc::new(NcFile::open(&vectors::write_nemo_like(d.path())).unwrap());
        let svc = SliceService::new(1 << 20, 1, Arc::new(|| {}));
        let req = SeriesRequest { var: "/uo+vo".into(), point: vec![0, 10, 15], along: 0, centre: 1 };
        let wait = |h: &mut crate::SeriesHandle| {
            let mut blocks = Vec::new();
            for _ in 0..1000 {
                blocks.extend(h.take_blocks());
                if let Some(done) = h.finished() {
                    blocks.extend(h.take_blocks());
                    done.unwrap();
                    return blocks;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            panic!("series did not finish");
        };
        let blocks = wait(&mut svc.request_series(&f, req.clone()));
        assert!(!blocks.is_empty());
        let mut seen = 0;
        for b in &blocks {
            let v = b.vector.as_ref().expect("vector block");
            assert_eq!(v.east.len(), b.values.len());
            for (o, e) in v.east.iter().enumerate() {
                let full = f.read_slice(&plane(&f, "/uo+vo", b.start + o)).unwrap();
                let want = full.vector.as_ref().unwrap().east[10 * 30 + 15];
                assert!(nan_eq(*e, want), "t {}", b.start + o);
                assert!(nan_eq(b.values[o], full.data[10 * 30 + 15]));
                seen += 1;
            }
        }
        assert_eq!(seen, 2);
        // A cache hit carries the components too.
        let hit = wait(&mut svc.request_series(&f, req));
        assert_eq!(hit.len(), 1);
        let v = hit[0].vector.as_ref().expect("cached vector");
        assert_eq!(v.east.len(), 2);
        assert_eq!(v.north.len(), 2);
    }
}
