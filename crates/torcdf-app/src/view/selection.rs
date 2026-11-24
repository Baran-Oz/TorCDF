//! The selection shared by the Extract and Statistics windows: variables,
//! time, region and other ranges, typed or drawn on screen, turned into a
//! core [`XSelection`] or a statistics request.

use std::collections::BTreeSet;
use std::sync::Arc;

use torcdf_core::cf::{Geo, TimeUnits, detect_axes, detect_geo};
use torcdf_core::extract::range::{curv_box, lon_range, parse_date, value_range};
use torcdf_core::extract::{DimKey, Selection as XSelection};
use torcdf_core::stats::StatsRequest;
use torcdf_core::NcFile;

use crate::app::Job;

const NONE: &str = "the selection holds no cells";

/// The region's four boxes, in degrees as typed.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RegionText {
    pub south: String,
    pub north: String,
    pub west: String,
    pub east: String,
}

/// One dimension's From/To, as typed.
#[derive(Clone, Debug, PartialEq)]
pub struct DimRow {
    pub key: DimKey,
    pub label: String,
    pub from: String,
    pub to: String,
}

/// A dimension of the ticked variables, with its values read off the UI thread.
pub struct DimCoord {
    pub key: DimKey,
    pub len: usize,
    /// Coordinate values; `None`: indices are typed.
    pub values: Option<Vec<f64>>,
    /// Time units when this is a time axis that has them.
    pub time: Option<TimeUnits>,
    pub is_time: bool,
}

/// The dimensions the region crops.
pub enum RegionAxes {
    Regular { lat: DimKey, lon: DimKey, lats: Vec<f64>, lons: Vec<f64> },
    Curvilinear { y: DimKey, x: DimKey, ny: usize, nx: usize, lat: Arc<Vec<f64>>, lon: Arc<Vec<f64>> },
}

impl RegionAxes {
    fn keys(&self) -> [&DimKey; 2] {
        match self {
            RegionAxes::Regular { lat, lon, .. } => [lat, lon],
            RegionAxes::Curvilinear { y, x, .. } => [y, x],
        }
    }

    /// (south, north, west, east) of the grid's cells.
    fn extent(&self) -> Option<(f64, f64, f64, f64)> {
        let (lat, lon): (&[f64], &[f64]) = match self {
            RegionAxes::Regular { lats, lons, .. } => (lats, lons),
            RegionAxes::Curvilinear { lat, lon, .. } => (lat, lon),
        };
        let span = |v: &[f64]| {
            let f = v.iter().copied().filter(|x| x.is_finite());
            let lo = f.clone().fold(f64::INFINITY, f64::min);
            let hi = f.fold(f64::NEG_INFINITY, f64::max);
            (lo <= hi).then_some((lo, hi))
        };
        let ((s, n), (w, e)) = (span(lat)?, span(lon)?);
        Some((s, n, w, e))
    }
}

/// Coordinates of the ticked variables (read on a helper thread).
pub struct FormCoords {
    pub ticked: BTreeSet<String>,
    /// Time first, then in the variables' order; region dimensions included.
    pub dims: Vec<DimCoord>,
    pub region: Option<RegionAxes>,
}

/// Reads what the form needs about `ticked` (small 1-D coordinates and
/// 2-D lat/lon). Unreadable coordinates fall back to indices.
pub fn load_coords(nc: &NcFile, ticked: &BTreeSet<String>) -> FormCoords {
    let ds = nc.meta_with_derived();
    let mut dims: Vec<DimCoord> = Vec::new();
    let mut region = None;
    for path in ticked {
        let Some(info) = nc.var_info_any(path) else { continue };
        if region.is_none() {
            let key = |d: usize| nc.dim_key(path, &info.dims[d]);
            region = match detect_geo(&ds, &info) {
                Geo::Regular { lat_dim, lon_dim, lat_var, lon_var } => (|| {
                    Some(RegionAxes::Regular {
                        lat: key(lat_dim)?,
                        lon: key(lon_dim)?,
                        lats: nc.read_coord(&lat_var).ok()?,
                        lons: nc.read_coord(&lon_var).ok()?,
                    })
                })(),
                Geo::Curvilinear { y_dim, x_dim, lat_var, lon_var } => (|| {
                    let (yn, xn) = (&info.dims[y_dim], &info.dims[x_dim]);
                    Some(RegionAxes::Curvilinear {
                        y: key(y_dim)?,
                        x: key(x_dim)?,
                        ny: info.shape[y_dim],
                        nx: info.shape[x_dim],
                        lat: Arc::new(nc.read_horizontal(&lat_var, yn, xn).ok()?),
                        lon: Arc::new(nc.read_horizontal(&lon_var, yn, xn).ok()?),
                    })
                })(),
                Geo::None => None,
            };
        }
        let axes = detect_axes(&ds, &info);
        for (i, name) in info.dims.iter().enumerate() {
            let Some(key) = nc.dim_key(path, name) else { continue };
            if dims.iter().any(|d| d.key == key) {
                continue;
            }
            let is_time = axes.time == Some(i);
            let coord = if is_time {
                axes.time_var.clone()
            } else {
                let p = if key.group == "/" { format!("/{name}") } else { format!("{}/{name}", key.group) };
                ds.var(&p).filter(|c| c.dims.len() == 1 && c.dims[0] == *name).map(|c| c.path.clone())
            };
            let coord = coord.and_then(|p| ds.var(&p).cloned()).filter(|c| c.dtype.is_numeric());
            let values = coord.as_ref().and_then(|c| nc.read_coord(&c.path).ok()).filter(|v| v.len() == info.shape[i]);
            let time = coord.as_ref().filter(|_| is_time && values.is_some()).and_then(|c| {
                let attr = |n: &str| c.attrs.iter().find(|a| a.name == n).and_then(|a| a.value.as_str());
                TimeUnits::parse(attr("units")?, attr("calendar"))
            });
            dims.push(DimCoord { key, len: info.shape[i], values, time, is_time });
        }
    }
    dims.sort_by_key(|d| !d.is_time);
    FormCoords { ticked: ticked.clone(), dims, region }
}

/// A number as typed back: up to 6 decimals, no trailing zeros.
pub fn fmt_num(v: f64) -> String {
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.to_string() }
}

/// The text for position `i` of `d`: a date, a value or an index.
fn fmt_at(d: &DimCoord, i: usize) -> String {
    match (&d.values, &d.time) {
        (Some(v), Some(t)) => t.decode(v[i]).to_string(),
        (Some(v), None) => fmt_num(v[i]),
        (None, _) => i.to_string(),
    }
}

/// (start, count) of `d` from its typed From/To.
fn row_range(d: &DimCoord, from: &str, to: &str) -> Result<(usize, usize), String> {
    let name = &d.key.name;
    let bad = |t: &str| format!("{name}: cannot read \"{}\"", t.trim());
    match &d.values {
        Some(vals) => {
            let read = |t: &str| {
                d.time.as_ref().and_then(|u| parse_date(t, u)).or_else(|| t.trim().parse::<f64>().ok()).ok_or_else(|| bad(t))
            };
            value_range(vals, read(from)?, read(to)?).ok_or_else(|| NONE.to_string())
        }
        None => {
            let read = |t: &str| t.trim().parse::<usize>().map_err(|_| bad(t));
            let (a, b) = (read(from)?, read(to)?);
            let (a, b) = (a.min(b), a.max(b).min(d.len.saturating_sub(1)));
            if a >= d.len {
                return Err(NONE.into());
            }
            Ok((a, b - a + 1))
        }
    }
}

/// The selection fields of the Extract or Statistics window.
pub struct SelectionForm {
    pub file: u64,
    pub file_name: String,
    pub ticked: BTreeSet<String>,
    /// Degrees; Some when the ticked variables have lat/lon.
    pub region: Option<RegionText>,
    /// Every other dimension of the ticked variables, time first.
    pub dims: Vec<DimRow>,
    pub coords: Option<FormCoords>,
    pub loading: Option<Job<FormCoords>>,
    /// A region to use once coordinates arrive (map view or drawn box;
    /// `true`: limit it to the grid).
    pub pending_region: Option<(RegionText, bool)>,
    /// A time range (axis values) dragged on a plot, waiting for coordinates.
    pub pending_time: Option<(f64, f64)>,
    /// Index ranges (first, last) drawn on an image, waiting for coordinates.
    pub pending_index: Vec<(DimKey, usize, usize)>,
    /// The y rows of a box drawn on a curvilinear image, until its x arrives.
    curv_rows: Option<(usize, usize)>,
    /// The selection last checked.
    last: Option<XSelection>,
}

impl SelectionForm {
    pub fn new(file: u64, file_name: String, ticked: BTreeSet<String>) -> SelectionForm {
        SelectionForm {
            file,
            file_name,
            ticked,
            region: None,
            dims: Vec::new(),
            coords: None,
            loading: None,
            pending_region: None,
            pending_time: None,
            pending_index: Vec::new(),
            curv_rows: None,
            last: None,
        }
    }

    /// Coordinates are being read.
    pub fn loading_coords(&self) -> bool {
        self.loading.is_some()
    }

    /// The coordinates loaded are not those of the ticked variables.
    pub fn coords_stale(&self) -> bool {
        self.coords.as_ref().is_none_or(|c| c.ticked != self.ticked)
    }

    fn coord(&self, key: &DimKey) -> Option<&DimCoord> {
        self.coords.as_ref()?.dims.iter().find(|d| d.key == *key)
    }

    /// Takes freshly read coordinates: rows for new dimensions start whole,
    /// rows kept keep their text; the region starts as given or whole.
    pub fn set_coords(&mut self, c: FormCoords) {
        let region_keys: Vec<DimKey> = c.region.as_ref().map(|r| r.keys().map(|k| k.clone()).to_vec()).unwrap_or_default();
        let old = std::mem::take(&mut self.dims);
        for d in c.dims.iter().filter(|d| !region_keys.contains(&d.key) && d.len > 0) {
            let row = old.iter().find(|r| r.key == d.key).cloned().unwrap_or_else(|| DimRow {
                key: d.key.clone(),
                label: d.key.name.clone(),
                from: fmt_at(d, 0),
                to: fmt_at(d, d.len - 1),
            });
            self.dims.push(row);
        }
        let has_region = c.region.is_some();
        self.coords = Some(c);
        if !has_region {
            self.region = None;
        } else if self.region.is_none() {
            self.region = self.whole_region();
        }
        self.apply_pending();
    }

    /// Applies what was drawn or asked for before the coordinates were
    /// read (kept while they are being read).
    pub fn apply_pending(&mut self) {
        if self.coords_stale() {
            return;
        }
        if let Some((r, clamp)) = self.pending_region.take()
            && self.region.is_some()
        {
            self.region = Some(if clamp { self.clamped(r) } else { r });
        }
        if let Some((a, b)) = self.pending_time.take() {
            let time = self.coords.as_ref().and_then(|c| c.dims.iter().find(|d| d.is_time));
            if let Some(d) = time
                && let Some(i) = self.dims.iter().position(|r| r.key == d.key)
            {
                let text = |v: f64| match (&d.values, &d.time) {
                    (Some(_), Some(t)) => t.decode(v).to_string(),
                    (Some(_), None) => fmt_num(v),
                    (None, _) => (v.round().max(0.0) as usize).to_string(),
                };
                let (a, b) = if a <= b { (a, b) } else { (b, a) };
                (self.dims[i].from, self.dims[i].to) = (text(a), text(b));
            }
        }
        for (key, a, b) in std::mem::take(&mut self.pending_index) {
            let (a, b) = (a.min(b), a.max(b));
            if let Some(i) = self.dims.iter().position(|r| r.key == key) {
                if let (Some(x), Some(y)) = (self.text_at(&key, a), self.text_at(&key, b)) {
                    (self.dims[i].from, self.dims[i].to) = (x, y);
                }
                continue;
            }
            self.index_to_region(&key, a, b);
        }
    }

    /// A drawn index range along a region dimension becomes the region's
    /// latitudes or longitudes (regular grids), or the lat/lon bounds of
    /// the cells (curvilinear grids, once both y and x arrived).
    fn index_to_region(&mut self, key: &DimKey, a: usize, b: usize) {
        let Some(axes) = self.coords.as_ref().and_then(|c| c.region.as_ref()) else { return };
        let Some(r) = self.region.as_mut() else { return };
        match axes {
            RegionAxes::Regular { lat, lon, lats, lons } => {
                let (Some(&x), Some(&y)) = (if key == lat { lats } else { lons }.get(a), if key == lat { lats } else { lons }.get(b))
                else {
                    return;
                };
                let (lo, hi) = (x.min(y), x.max(y));
                if key == lat {
                    (r.south, r.north) = (fmt_num(lo), fmt_num(hi));
                } else if key == lon {
                    (r.west, r.east) = (fmt_num(lo), fmt_num(hi));
                }
            }
            RegionAxes::Curvilinear { y, x, nx, lat, lon, .. } => {
                // y arrives first; x completes the box.
                if key == y {
                    self.curv_rows = Some((a, b));
                    return;
                }
                let (Some((y0, y1)), true) = (self.curv_rows.take(), key == x) else { return };
                let mut pts = Vec::new();
                for j in y0..=y1 {
                    for i in a..=b {
                        if let (Some(&la), Some(&lo)) = (lat.get(j * nx + i), lon.get(j * nx + i))
                            && la.is_finite()
                            && lo.is_finite()
                        {
                            pts.push((lo, la));
                        }
                    }
                }
                if pts.is_empty() {
                    return;
                }
                let (s, n, w, e) = lonlat_bounds(&pts);
                *r = RegionText { south: fmt_num(s), north: fmt_num(n), west: fmt_num(w), east: fmt_num(e) };
            }
        }
    }

    /// The whole grid's extent as region text.
    pub fn whole_region(&self) -> Option<RegionText> {
        let (s, n, w, e) = self.coords.as_ref()?.region.as_ref()?.extent()?;
        Some(RegionText { south: fmt_num(s), north: fmt_num(n), west: fmt_num(w), east: fmt_num(e) })
    }

    /// `r` limited to the grid's extent: latitudes always; longitudes when
    /// the box holds the whole grid's span (a partial box is kept as is, so
    /// a seam stays visible).
    pub fn clamped(&self, r: RegionText) -> RegionText {
        let Some((s, n, w, e)) = self.coords.as_ref().and_then(|c| c.region.as_ref()?.extent()) else { return r };
        let num = |t: &str| t.trim().parse::<f64>().ok();
        let (Some(rs), Some(rn), Some(rw), Some(re)) = (num(&r.south), num(&r.north), num(&r.west), num(&r.east)) else {
            return r;
        };
        let (rw, re) = if re - rw >= 360.0 || (rw <= w && re >= e) { (w, e) } else { (rw, re) };
        RegionText { south: fmt_num(rs.max(s)), north: fmt_num(rn.min(n)), west: fmt_num(rw), east: fmt_num(re) }
    }

    /// The typed form as index ranges (no file access).
    pub fn selection(&self) -> Result<XSelection, String> {
        if self.ticked.is_empty() {
            return Err("no variable is ticked".into());
        }
        let c = self.coords.as_ref().filter(|_| !self.coords_stale()).ok_or("reading coordinates…")?;
        let mut sel = XSelection { vars: self.ticked.iter().cloned().collect(), ..Default::default() };
        for row in &self.dims {
            let Some(d) = self.coord(&row.key) else { continue };
            sel.ranges.insert(row.key.clone(), row_range(d, &row.from, &row.to)?);
        }
        if let (Some(r), Some(axes)) = (&self.region, &c.region) {
            let num = |t: &str, what: &str| t.trim().parse::<f64>().map_err(|_| format!("{what}: cannot read \"{}\"", t.trim()));
            let (s, n) = (num(&r.south, "latitude")?, num(&r.north, "latitude")?);
            let (w, e) = (num(&r.west, "longitude")?, num(&r.east, "longitude")?);
            match axes {
                RegionAxes::Regular { lat, lon, lats, lons } => {
                    sel.ranges.insert(lat.clone(), value_range(lats, s, n).ok_or(NONE)?);
                    sel.ranges.insert(lon.clone(), lon_range(lons, w, e)?);
                }
                RegionAxes::Curvilinear { y, x, ny, nx, lat, lon } => {
                    let [ry, rx] = curv_box(lat, lon, *ny, *nx, s, n, w, e).ok_or(NONE)?;
                    sel.ranges.insert(y.clone(), ry);
                    sel.ranges.insert(x.clone(), rx);
                }
            }
        }
        Ok(sel)
    }

    /// The rows (not the region) as index ranges.
    fn row_ranges(&self) -> Result<std::collections::BTreeMap<DimKey, (usize, usize)>, String> {
        if self.ticked.is_empty() {
            return Err("no variable is ticked".into());
        }
        if self.coords_stale() {
            return Err("reading coordinates…".into());
        }
        let mut out = std::collections::BTreeMap::new();
        for row in &self.dims {
            if let Some(d) = self.coord(&row.key) {
                out.insert(row.key.clone(), row_range(d, &row.from, &row.to)?);
            }
        }
        Ok(out)
    }

    /// Checks the form again: rows are rewritten to their first and last
    /// kept steps, and the whole selection (region included) is returned
    /// and remembered for the overlays and the summary.
    pub fn refresh(&mut self) -> Result<XSelection, String> {
        let rows = self.row_ranges();
        if let Ok(rows) = &rows {
            for i in 0..self.dims.len() {
                let key = self.dims[i].key.clone();
                if let (Some(&(s, n)), Some(d)) = (rows.get(&key), self.coord(&key)) {
                    let (from, to) = (fmt_at(d, s), fmt_at(d, s + n - 1));
                    (self.dims[i].from, self.dims[i].to) = (from, to);
                }
            }
        }
        let sel = rows.clone().and_then(|_| self.selection());
        // A region the index form cannot hold (Statistics accepts boxes Save
        // refuses) still leaves the rows for the overlays and the summary.
        self.last = match (&sel, rows) {
            (Ok(s), _) => Some(s.clone()),
            (Err(_), Ok(ranges)) => Some(XSelection { vars: self.ticked.iter().cloned().collect(), ranges }),
            (Err(_), Err(_)) => None,
        };
        sel
    }

    /// The form as a statistics request: the rows as index ranges, and the
    /// region as a box in degrees (each variable applies it on its own grid,
    /// so a box Save refuses, such as one across the seam, is fine here).
    pub fn stats_request(&self, weighted: bool) -> Result<StatsRequest, String> {
        let ranges = self.row_ranges()?;
        let has_region = self.coords.as_ref().is_some_and(|c| c.region.is_some());
        let region = match self.region.as_ref().filter(|_| has_region) {
            Some(r) => {
                let num = |t: &str, what: &str| t.trim().parse::<f64>().map_err(|_| format!("{what}: cannot read \"{}\"", t.trim()));
                Some([num(&r.south, "latitude")?, num(&r.north, "latitude")?, num(&r.west, "longitude")?, num(&r.east, "longitude")?])
            }
            None => None,
        };
        Ok(StatsRequest { vars: self.ticked.iter().cloned().collect(), ranges, region, weighted })
    }

    /// The text for position `i` of the row with `key`.
    /// How many positions the row with `key` keeps, as last checked.
    pub fn kept(&self, key: &DimKey) -> Option<usize> {
        self.last.as_ref()?.ranges.get(key).map(|r| r.1)
    }

    pub fn text_at(&self, key: &DimKey, i: usize) -> Option<String> {
        self.coord(key).filter(|d| i < d.len).map(|d| fmt_at(d, i))
    }

    /// What was cropped, for the `history` line: "time A..B, lat S..N, lon W..E[, depth …]".
    pub fn summary(&self) -> String {
        let Some(sel) = &self.last else { return String::new() };
        let cropped = |k: &DimKey| {
            let len = self.coord(k).map_or(0, |d| d.len);
            sel.ranges.get(k).is_some_and(|&r| r != (0, len))
        };
        let row = |r: &DimRow| format!("{} {}..{}", r.label, r.from.trim(), r.to.trim());
        let is_time = |r: &DimRow| self.coord(&r.key).is_some_and(|d| d.is_time);
        let mut parts: Vec<String> = self.dims.iter().filter(|r| is_time(r) && cropped(&r.key)).map(row).collect();
        if let (Some(r), Some(axes)) = (&self.region, self.coords.as_ref().and_then(|c| c.region.as_ref()))
            && axes.keys().iter().any(|k| cropped(k) || !sel.ranges.contains_key(*k))
        {
            parts.push(format!("lat {}..{}, lon {}..{}", r.south.trim(), r.north.trim(), r.west.trim(), r.east.trim()));
        }
        parts.extend(self.dims.iter().filter(|r| !is_time(r) && cropped(&r.key)).map(row));
        if parts.is_empty() { "whole variables".into() } else { parts.join(", ") }
    }
}

/// (south, north, west, east) of lon/lat points. Longitudes are taken the
/// short way round, so a box across the date line stays narrow (east may
/// then pass 180).
pub fn lonlat_bounds(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let f = |g: fn(f64, f64) -> f64, init: f64, sel: fn(&(f64, f64)) -> f64| points.iter().map(sel).fold(init, g);
    let (s, n) = (f(f64::min, f64::INFINITY, |p| p.1), f(f64::max, f64::NEG_INFINITY, |p| p.1));
    let (w, e) = (f(f64::min, f64::INFINITY, |p| p.0), f(f64::max, f64::NEG_INFINITY, |p| p.0));
    if e - w <= 180.0 {
        return (s, n, w, e);
    }
    // Shifting the western half by 360 gives the narrower box.
    let shifted = |p: &(f64, f64)| if p.0 < 0.0 { p.0 + 360.0 } else { p.0 };
    let w2 = points.iter().map(shifted).fold(f64::INFINITY, f64::min);
    let e2 = points.iter().map(shifted).fold(f64::NEG_INFINITY, f64::max);
    if e2 - w2 < e - w { (s, n, w2, e2) } else { (s, n, w, e) }
}

/// What the Extract window shows on the plots of its file.
pub struct SelectionOverlay {
    pub file: u64,
    /// (south, north, west, east) as typed.
    pub region: Option<[f64; 4]>,
    /// Index ranges last checked, per dimension.
    pub ranges: std::collections::BTreeMap<DimKey, (usize, usize)>,
    /// The time dimension's name and its kept first and last axis values.
    pub time: Option<(String, f64, f64)>,
}

impl SelectionForm {
    pub fn overlay(&self) -> Option<SelectionOverlay> {
        let sel = self.last.as_ref()?;
        let region = self.region.as_ref().and_then(|r| {
            let n = |t: &str| t.trim().parse::<f64>().ok();
            Some([n(&r.south)?, n(&r.north)?, n(&r.west)?, n(&r.east)?])
        });
        let time = self.coords.as_ref()?.dims.iter().find(|d| d.is_time).and_then(|d| {
            let &(s, c) = sel.ranges.get(&d.key)?;
            let at = |i: usize| d.values.as_ref().map_or(i as f64, |v| v[i]);
            Some((d.key.name.clone(), at(s), at(s + c - 1)))
        });
        Some(SelectionOverlay { file: self.file, region, ranges: sel.ranges.clone(), time })
    }
}
