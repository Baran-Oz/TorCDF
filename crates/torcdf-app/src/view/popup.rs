//! Pop-ups of a point's time series or vertical profile: the model (what is
//! plotted, how it fills in) and CSV export. The windows are in `ui`.

use std::sync::Arc;

use torcdf_core::cf::{TimeUnits, detect_axes};
use torcdf_core::{NcFile, SeriesHandle, SeriesRequest, SliceService, VarInfo};

use crate::view::var_view::VarView;

const NOTHING: &str = "No time or vertical dimension to plot here.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopupKind {
    Series,
    Profile,
}

/// What differs between the kinds of one pop-up.
struct KindInfo {
    kind: PopupKind,
    along: usize,
    centre: usize,
    axis: Vec<f64>,
    time: Option<TimeUnits>,
    down: bool,
    title: String,
    along_name: String,
}

pub struct Popup {
    pub id: u64,
    pub pinned: bool,
    pub file_id: u64,
    pub var_path: String,
    pub var_name: String,
    pub units: Option<String>,
    /// Full index at the click.
    pub point: Vec<usize>,
    /// Marker position on the map; `None` in the image view.
    pub lonlat: Option<(f64, f64)>,
    /// Available kinds: `Series` if there is a time dim, `Profile` if a vertical dim.
    pub kinds: Vec<PopupKind>,
    pub kind: PopupKind,
    pub along: usize,
    /// E.g. "uo at 12.31°E, 40.12°N — depth 3 (1.5 m), member 1".
    pub title: String,
    /// Coordinate values along `along` (indices when unknown).
    pub axis: Vec<f64>,
    /// When `along` is time.
    pub time: Option<TimeUnits>,
    /// Profile y runs downward.
    pub down: bool,
    /// `None` = not loaded yet; `Some(NaN)` = missing.
    pub values: Vec<Option<f32>>,
    /// Vector pairs only.
    pub east: Vec<Option<f32>>,
    pub north: Vec<Option<f32>>,
    pub show_east: bool,
    pub show_north: bool,
    /// The window's rectangle as drawn last frame (for the PNG save).
    pub rect: Option<eframe::egui::Rect>,
    pub error: Option<String>,
    /// Name of the dimension `along` (the CSV's first column).
    along_name: String,
    /// Component names of a vector pair, for the CSV.
    east_name: String,
    north_name: String,
    infos: Vec<KindInfo>,
    handle: Option<SeriesHandle>,
    nc: Arc<NcFile>,
}

fn attr<'a>(v: &'a VarInfo, name: &str) -> Option<&'a str> {
    v.attrs.iter().find(|a| a.name == name).and_then(|a| a.value.as_str()).map(str::trim)
}

/// The 1-D coordinate variable of dimension `dim`, found as `detect_axes` does.
fn coord_var<'a>(nc: &'a NcFile, var: &VarInfo, dim: &str) -> Option<&'a VarInfo> {
    let ds = &nc.meta;
    ds.group_of(&var.path)
        .vars
        .iter()
        .chain(ds.root.vars.iter())
        .find(|c| c.name == dim && c.dims.len() == 1)
}

/// The vertical coordinate increases downward: `positive = "down"` (an
/// explicit `positive` decides), else its standard name or units say depth
/// or pressure.
fn runs_down(c: Option<&VarInfo>) -> bool {
    let Some(c) = c else { return false };
    if let Some(p) = attr(c, "positive") {
        return p.eq_ignore_ascii_case("down");
    }
    let std_name = attr(c, "standard_name").map(str::to_ascii_lowercase).unwrap_or_default();
    let units = attr(c, "units").map(str::to_ascii_lowercase).unwrap_or_default();
    std_name.contains("depth")
        || std_name == "air_pressure"
        || matches!(units.as_str(), "pa" | "hpa" | "dbar")
}

fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

fn lonlat_text((lon, lat): (f64, f64)) -> String {
    let (ew, ns) = (if lon < 0.0 { 'W' } else { 'E' }, if lat < 0.0 { 'S' } else { 'N' });
    format!("{:.2}°{ew}, {:.2}°{ns}", lon.abs(), lat.abs())
}

fn leaf(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Quotes a CSV field when it holds a comma, quote or line break.
fn field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

impl Popup {
    /// Err(text) when there is nothing to plot (message for the plot area).
    pub fn open(
        view: &VarView,
        iy: usize,
        ix: usize,
        lonlat: Option<(f64, f64)>,
        id: u64,
        service: &SliceService,
    ) -> Result<Popup, String> {
        // A layered view pops up its coloured variable's own series; only a
        // pair's own entry adds its components.
        let var = match &view.base {
            Some(p) => view.nc.meta.var(p).ok_or_else(|| NOTHING.to_string())?,
            None => &view.var,
        };
        let pair = view.vector.as_ref().filter(|_| view.base.is_none());
        let shape = &var.shape;
        if view.slider_pos.len() != shape.len() {
            return Err(NOTHING.into());
        }
        let mut point = view.slider_pos.clone();
        // The click is on the plotted y and x; a 1-D view has no such point.
        if let [py, px] = view.plotted_dims()[..] {
            if iy >= shape[py] || ix >= shape[px] {
                return Err("That point is outside the grid.".into());
            }
            point[py] = iy;
            point[px] = ix;
        }
        let vertical = detect_axes(&view.nc.meta, var).vertical;
        let mut infos = Vec::new();
        for (kind, dim) in [(PopupKind::Series, view.time_dim), (PopupKind::Profile, vertical)] {
            if let Some(d) = dim.filter(|&d| d < shape.len() && shape[d] > 0) {
                infos.push(kind_info(view, var, kind, d, &point, lonlat));
            }
        }
        let Some(first) = infos.first() else { return Err(NOTHING.into()) };
        let (east_name, north_name) =
            pair.map(|p| (leaf(&p.east).to_string(), leaf(&p.north).to_string())).unwrap_or_default();
        let mut popup = Popup {
            id,
            pinned: false,
            file_id: view.file_id,
            var_path: var.path.clone(),
            var_name: var.name.clone(),
            units: pair
                .and_then(|p| p.units.clone())
                .or_else(|| attr(var, "units").map(str::to_string)),
            point,
            lonlat,
            kinds: infos.iter().map(|i| i.kind).collect(),
            kind: first.kind,
            along: first.along,
            title: String::new(),
            axis: Vec::new(),
            time: None,
            down: false,
            values: Vec::new(),
            east: Vec::new(),
            north: Vec::new(),
            show_east: false,
            show_north: false,
            rect: None,
            error: None,
            along_name: String::new(),
            east_name,
            north_name,
            infos,
            handle: None,
            nc: view.nc.clone(),
        };
        popup.select(popup.kind, service);
        Ok(popup)
    }

    /// Adopts kind `k` and requests its series (the previous one is dropped,
    /// which cancels its unstarted reads).
    fn select(&mut self, k: PopupKind, service: &SliceService) {
        self.handle = None;
        let Some(i) = self.infos.iter().position(|i| i.kind == k) else { return };
        let info = &self.infos[i];
        self.kind = k;
        self.along = info.along;
        self.title = info.title.clone();
        self.axis = info.axis.clone();
        self.time = info.time.clone();
        self.down = info.down;
        self.along_name = info.along_name.clone();
        let len = self.axis.len();
        self.values = vec![None; len];
        let vector = !self.east_name.is_empty();
        self.east = if vector { vec![None; len] } else { Vec::new() };
        self.north = self.east.clone();
        self.error = None;
        let req = SeriesRequest {
            var: self.var_path.clone(),
            point: self.point.clone(),
            along: info.along,
            centre: info.centre,
        };
        self.handle = Some(service.request_series(&self.nc, req));
    }

    /// Takes the values that arrived; true when there were new ones (or the
    /// read just failed).
    pub fn poll(&mut self) -> bool {
        let Some(h) = self.handle.as_mut() else { return false };
        let blocks = h.take_blocks();
        let mut changed = !blocks.is_empty();
        for b in blocks {
            for (i, v) in b.values.iter().enumerate() {
                if let Some(slot) = self.values.get_mut(b.start + i) {
                    *slot = Some(*v);
                }
            }
            if let Some(parts) = &b.vector {
                for (src, dst) in [(&parts.east, &mut self.east), (&parts.north, &mut self.north)] {
                    for (i, v) in src.iter().enumerate() {
                        if let Some(slot) = dst.get_mut(b.start + i) {
                            *slot = Some(*v);
                        }
                    }
                }
            }
        }
        if self.error.is_none()
            && let Some(Err(e)) = h.finished()
        {
            self.error = Some(e.to_string());
            changed = true;
        }
        changed
    }

    /// (steps delivered, total steps).
    pub fn progress(&self) -> (usize, usize) {
        self.handle.as_ref().map_or((0, 0), SeriesHandle::progress)
    }

    /// Name of the dimension plotted along.
    pub fn along_name(&self) -> &str {
        &self.along_name
    }

    /// A vector pair: east and north can be added to the plot.
    pub fn has_components(&self) -> bool {
        !self.east_name.is_empty()
    }

    /// Switches between series and profile; re-requests along the other dim.
    pub fn set_kind(&mut self, k: PopupKind, service: &SliceService) {
        if k != self.kind && self.kinds.contains(&k) {
            self.select(k, service);
        }
    }

    /// The CSV as written to a file: UTF-8 with a byte order mark, so Excel
    /// reads units such as "degC" right.
    pub fn csv_file_bytes(&self) -> Vec<u8> {
        let mut out = vec![0xEF, 0xBB, 0xBF];
        out.extend_from_slice(self.csv().as_bytes());
        out
    }

    /// The plotted data as CSV: the coordinate (ISO dates when time
    /// decodes), the value, and the vector components when shown. Missing or
    /// unloaded values are empty fields.
    pub fn csv(&self) -> String {
        let head = |name: &str| match &self.units {
            Some(u) => field(&format!("{name} ({u})")),
            None => field(name),
        };
        let mut cols: Vec<(String, &[Option<f32>])> = vec![(head(&self.var_name), &self.values)];
        if self.show_east {
            cols.push((head(&self.east_name), &self.east));
        }
        if self.show_north {
            cols.push((head(&self.north_name), &self.north));
        }
        let mut out = field(&self.along_name);
        for (name, _) in &cols {
            out.push(',');
            out.push_str(name);
        }
        out.push('\n');
        for i in 0..self.values.len() {
            let x = self.axis.get(i).copied().unwrap_or(i as f64);
            match &self.time {
                Some(t) => out.push_str(&t.decode(x).iso()),
                None => out.push_str(&x.to_string()),
            }
            for (_, col) in &cols {
                out.push(',');
                if let Some(v) = col.get(i).copied().flatten()
                    && !v.is_nan()
                {
                    out.push_str(&v.to_string());
                }
            }
            out.push('\n');
        }
        out
    }
}

/// Axis, title and orientation of the pop-up of `var` (the view's plain
/// variable) along dimension `d`.
fn kind_info(
    view: &VarView,
    var: &VarInfo,
    kind: PopupKind,
    d: usize,
    point: &[usize],
    lonlat: Option<(f64, f64)>,
) -> KindInfo {
    let len = var.shape[d];
    let coord = |dim: usize| {
        view.coords.dims.get(dim).and_then(|c| c.as_ref()).filter(|c| c.len() == var.shape[dim])
    };
    // Indices when the coordinate is not loaded (yet) or failed; they are
    // not times, so dates come only with the coordinate.
    let (axis, time) = match coord(d) {
        Some(c) => (c.clone(), view.time_units.clone().filter(|_| view.time_dim == Some(d))),
        None => ((0..len).map(|i| i as f64).collect(), None),
    };
    let down = kind == PopupKind::Profile && runs_down(coord_var(&view.nc, var, &var.dims[d]));

    let plotted = view.plotted_dims();
    let place = match (lonlat, &plotted[..]) {
        (Some(ll), _) => format!(" at {}", lonlat_text(ll)),
        (None, &[y, x]) => format!(" at {} {}, {} {}", var.dims[y], point[y], var.dims[x], point[x]),
        _ => String::new(),
    };
    let others: Vec<String> = view
        .slider_dims()
        .into_iter()
        .filter(|&s| s != d)
        .map(|s| {
            let idx = point[s];
            let name = &var.dims[s];
            let value = coord(s).and_then(|c| c.get(idx)).map(|&v| {
                if view.time_dim == Some(s)
                    && let Some(t) = &view.time_units
                {
                    return t.decode(v).to_string();
                }
                let units = coord_var(&view.nc, var, name).and_then(|c| attr(c, "units"));
                match units {
                    Some(u) => format!("{} {u}", round4(v)),
                    None => round4(v).to_string(),
                }
            });
            match value {
                Some(v) => format!("{name} {idx} ({v})"),
                None => format!("{name} {idx}"),
            }
        })
        .collect();
    let rest = if others.is_empty() { String::new() } else { format!(" — {}", others.join(", ")) };
    KindInfo {
        kind,
        along: d,
        centre: view.slider_pos[d],
        axis,
        time,
        down,
        title: format!("{}{place}{rest}", var.name),
        along_name: var.dims[d].clone(),
    }
}

/// The open pop-ups: at most one unpinned, any number pinned.
#[derive(Default)]
pub struct PopupSet {
    pub list: Vec<Popup>,
    next_id: u64,
}

impl PopupSet {
    /// Replaces the unpinned pop-up if any; pinned ones stay.
    pub fn place(&mut self, p: Popup) {
        self.list.retain(|q| q.pinned);
        self.list.push(p);
    }

    /// Drops it, which cancels its reads.
    pub fn close(&mut self, id: u64) {
        self.list.retain(|p| p.id != id);
    }

    pub fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// True when any pop-up got new values.
    pub fn poll(&mut self) -> bool {
        self.list.iter_mut().fold(false, |any, p| p.poll() | any)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui;
    use std::sync::{Mutex, mpsc};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    use torcdf_core::{NcFile, SliceRequest, fixtures};

    const TIMEOUT: Duration = Duration::from_secs(10);
    const NOTHING: &str = "No time or vertical dimension to plot here.";

    fn noop() -> Arc<dyn Fn() + Send + Sync> {
        Arc::new(|| {})
    }

    fn view_of(path: std::path::PathBuf, var: &str) -> VarView {
        let nc = Arc::new(fixtures::paired(NcFile::open(&path).unwrap()));
        let info = nc.meta.var(var).expect("variable").clone();
        let mut v = VarView::new(&egui::Context::default(), 1, nc, info);
        wait_coords(&mut v);
        v
    }

    fn wait_coords(v: &mut VarView) {
        let ctx = egui::Context::default();
        let end = Instant::now() + TIMEOUT;
        loop {
            if v.poll(&ctx).1 {
                return;
            }
            assert!(Instant::now() < end, "coords job timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn fill(p: &mut Popup) {
        let end = Instant::now() + TIMEOUT;
        loop {
            p.poll();
            if p.handle.as_ref().is_some_and(|h| h.finished().is_some()) {
                p.poll();
                return;
            }
            assert!(Instant::now() < end, "series timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn popup_series_fills() {
        let d = tempfile::tempdir().unwrap();
        let v = view_of(fixtures::write_nc4(d.path()), "/ocean/uo");
        let svc = SliceService::new(1 << 20, 2, noop());
        let mut p = Popup::open(&v, 4, 7, Some((1.0, 2.0)), 3, &svc).unwrap();
        assert_eq!(p.kind, PopupKind::Series);
        assert_eq!(p.kinds, vec![PopupKind::Series]);
        assert_eq!((p.id, p.point.clone()), (3, vec![0, 4, 7]));
        fill(&mut p);
        assert!(p.error.is_none(), "{:?}", p.error);
        assert_eq!(p.values.len(), 24);
        assert!(p.values.iter().all(Option::is_some));
        let col = v
            .nc
            .read_slice(&SliceRequest {
                var: "/ocean/uo".into(),
                start: vec![0, 4, 7],
                count: vec![24, 1, 1],
                stride: vec![1; 3],
            })
            .unwrap();
        let got: Vec<f32> = p.values.iter().map(|v| v.unwrap()).collect();
        assert_eq!(got, col.data);
        assert_eq!(p.progress(), (24, 24));
        assert_eq!(p.axis.len(), 24);
        assert!(p.time.is_some());
    }

    /// WRF: `Time` has no same-named variable; `XTIME(Time)` makes it time
    /// and gives the dates, for scalars and for the vector pair alike.
    #[test]
    fn wrf_popup_series() {
        let d = tempfile::tempdir().unwrap();
        let path = fixtures::vectors::write_wrf_like(d.path());
        let svc = SliceService::new(1 << 20, 2, noop());
        let check = |v: &VarView, p: &mut Popup| {
            assert_eq!(v.time_dim, Some(0));
            assert_eq!(p.kinds, vec![PopupKind::Series]);
            assert_eq!(p.axis, vec![0.0, 60.0]);
            assert!(p.time.is_some());
            fill(p);
            assert!(p.error.is_none(), "{:?}", p.error);
            assert!(p.values.iter().all(Option::is_some));
            let csv = p.csv();
            let rows: Vec<&str> = csv.lines().skip(1).collect();
            assert!(rows[0].starts_with("2000-01-01T00:00:00,"), "{csv}");
            assert!(rows[1].starts_with("2000-01-01T01:00:00,"), "{csv}");
        };
        for var in ["/T2", "/U"] {
            let v = view_of(path.clone(), var);
            let mut p = Popup::open(&v, 3, 4, None, 1, &svc).unwrap();
            check(&v, &mut p);
        }
        let nc = Arc::new(fixtures::paired(NcFile::open(&path).unwrap()));
        let pair = nc.pair("/U+V").expect("pair");
        let mut v = VarView::new_vector(&egui::Context::default(), 1, nc, pair);
        wait_coords(&mut v);
        let mut p = Popup::open(&v, 3, 4, None, 1, &svc).unwrap();
        check(&v, &mut p);
        assert!(p.east.iter().chain(&p.north).all(Option::is_some));
    }

    /// Before (or without) the coordinates the axis is indices, and indices
    /// are not dates.
    #[test]
    fn popup_without_time_coordinates_has_no_dates() {
        let d = tempfile::tempdir().unwrap();
        let mut v = view_of(fixtures::write_nc4(d.path()), "/ocean/uo");
        assert!(v.time_units.is_some());
        v.coords.dims[0] = None;
        let svc = SliceService::new(1 << 20, 1, noop());
        let p = Popup::open(&v, 4, 7, None, 1, &svc).unwrap();
        assert_eq!(p.axis, (0..24).map(f64::from).collect::<Vec<_>>());
        assert!(p.time.is_none());
        assert_eq!(p.csv().lines().nth(2).map(|l| l.split(',').next()), Some(Some("1")));
    }

    #[test]
    fn popup_needs_series_dimension() {
        let d = tempfile::tempdir().unwrap();
        let v = view_of(fixtures::write_curvilinear(d.path()), "/sst");
        let svc = SliceService::new(1 << 20, 1, noop());
        assert_eq!(Popup::open(&v, 1, 1, None, 1, &svc).err().as_deref(), Some(NOTHING));
    }

    #[test]
    fn click_off_grid_opens_nothing() {
        let d = tempfile::tempdir().unwrap();
        let v = view_of(fixtures::write_nc4(d.path()), "/ocean/uo");
        let svc = SliceService::new(1 << 20, 1, noop());
        assert!(Popup::open(&v, 10, 0, None, 1, &svc).is_err());
        assert!(Popup::open(&v, 0, 20, None, 1, &svc).is_err());
        assert!(Popup::open(&v, usize::MAX, usize::MAX, None, 1, &svc).is_err());
    }

    #[test]
    fn profile_down_for_depth() {
        let d = tempfile::tempdir().unwrap();
        let v = view_of(fixtures::write_profile(d.path()), "/t");
        let svc = SliceService::new(1 << 20, 1, noop());
        let mut p = Popup::open(&v, 1, 2, None, 1, &svc).unwrap();
        assert_eq!((p.kind, p.down), (PopupKind::Profile, true));
        assert_eq!(p.axis, vec![0.5, 1.5, 3.0, 5.0, 10.0]);
        assert!(p.time.is_none());
        fill(&mut p);
        let got: Vec<f32> = p.values.iter().map(|v| v.unwrap()).collect();
        assert_eq!(got, vec![12.0, 112.0, 212.0, 312.0, 412.0]);
    }

    fn bare(id: u64, svc: &SliceService, v: &VarView) -> Popup {
        Popup::open(v, 4, 7, None, id, svc).unwrap()
    }

    #[test]
    fn place_replaces_unpinned() {
        let d = tempfile::tempdir().unwrap();
        let v = view_of(fixtures::write_nc4(d.path()), "/ocean/uo");
        let svc = SliceService::new(1 << 20, 1, noop());
        let mut set = PopupSet::default();
        let ids = |s: &PopupSet| s.list.iter().map(|p| p.id).collect::<Vec<_>>();
        let (a, b, c) = (set.next_id(), set.next_id(), set.next_id());
        assert!(a != b && b != c);
        set.place(bare(a, &svc, &v));
        set.place(bare(b, &svc, &v));
        assert_eq!(ids(&set), vec![b]);
        set.list[0].pinned = true;
        set.place(bare(c, &svc, &v));
        assert_eq!(ids(&set), vec![b, c]);
        set.close(b);
        assert_eq!(ids(&set), vec![c]);
    }

    #[test]
    fn closing_popup_cancels() {
        let d = tempfile::tempdir().unwrap();
        let nc = Arc::new(fixtures::paired(NcFile::open(&fixtures::write_long_series(d.path())).unwrap()));
        assert!(!nc.fast_eligible("/v"));
        // An `on_ready` whose first call blocks until released.
        let (entered_tx, entered) = mpsc::sync_channel::<()>(1);
        let (release, release_rx) = mpsc::channel::<()>();
        let release_rx = Mutex::new(release_rx);
        let first = Arc::new(AtomicBool::new(true));
        let on_ready: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            if first.swap(false, Ordering::SeqCst) {
                entered_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv_timeout(TIMEOUT).expect("gate never released");
            }
        });
        let svc = SliceService::new(1 << 20, 1, on_ready);
        let info = nc.meta.var("/v").unwrap().clone();
        let mut v = VarView::new(&egui::Context::default(), 1, nc.clone(), info);
        wait_coords(&mut v);
        // `time` is unlimited with no variable: still the time dimension.
        assert_eq!(v.time_dim, Some(0));
        v.slider_pos = vec![2048];
        let mut set = PopupSet::default();
        let id = set.next_id();
        set.place(Popup::open(&v, 0, 0, None, id, &svc).unwrap());
        entered.recv_timeout(TIMEOUT).unwrap();
        // Nothing taken; close while the coordinator holds on its first block.
        set.close(id);
        assert!(set.list.is_empty());
        drop(v);
        release.send(()).unwrap();
        // Only the coordinator and this test hold the file now: once the
        // coordinator is gone it has exited, stopped by the closed pop-up alone.
        let end = Instant::now() + TIMEOUT;
        while Arc::strong_count(&nc) > 1 {
            assert!(Instant::now() < end, "the series coordinator did not stop");
            std::thread::yield_now();
        }
        assert!(svc.reads_performed() <= 2, "reads: {}", svc.reads_performed());
        let (tx, rx) = mpsc::sync_channel::<()>(1);
        std::thread::spawn(move || {
            drop(svc);
            tx.send(()).unwrap();
        });
        rx.recv_timeout(TIMEOUT).expect("service drop joins the series coordinator");
    }

    #[test]
    fn csv_format() {
        let d = tempfile::tempdir().unwrap();
        let v = view_of(fixtures::write_nc4(d.path()), "/ocean/uo");
        let svc = SliceService::new(1 << 20, 1, noop());
        let mut p = bare(1, &svc, &v);
        p.var_name = "uo".into();
        p.units = Some("m s-1".into());
        p.along_name = "time".into();
        p.axis = vec![0.0, 1.0, 2.0];
        p.time = TimeUnits::parse("hours since 2000-01-01", None);
        p.values = vec![Some(1.5), Some(f32::NAN), None];
        assert_eq!(
            p.csv(),
            "time,uo (m s-1)\n2000-01-01T00:00:00,1.5\n2000-01-01T01:00:00,\n2000-01-01T02:00:00,\n"
        );
        p.time = None;
        p.var_name = "a,b".into();
        assert_eq!(p.csv().lines().next(), Some("time,\"a,b (m s-1)\""));
    }

    #[test]
    fn csv_starts_with_bom() {
        let d = tempfile::tempdir().unwrap();
        let v = view_of(fixtures::write_nc4(d.path()), "/ocean/uo");
        let svc = SliceService::new(1 << 20, 1, noop());
        let mut p = bare(1, &svc, &v);
        p.units = Some("\u{b0}C".into());
        let bytes = p.csv_file_bytes();
        assert_eq!(&bytes[..3], [0xEF, 0xBB, 0xBF]);
        assert_eq!(&bytes[3..], p.csv().as_bytes());
    }

    #[test]
    fn vector_popup_has_components() {
        let d = tempfile::tempdir().unwrap();
        let nc = Arc::new(fixtures::paired(NcFile::open(&fixtures::vectors::write_nemo_like(d.path())).unwrap()));
        let pair = nc.pairs().into_iter().next().expect("pair");
        let mut v = VarView::new_vector(&egui::Context::default(), 1, nc, pair);
        wait_coords(&mut v);
        assert_eq!(v.time_dim, Some(0));
        let svc = SliceService::new(1 << 20, 2, noop());
        let mut p = Popup::open(&v, 3, 4, None, 1, &svc).unwrap();
        fill(&mut p);
        assert!(p.error.is_none(), "{:?}", p.error);
        assert_eq!(p.east.len(), 2);
        assert!(p.east.iter().chain(&p.north).all(Option::is_some));
        p.show_east = true;
        p.show_north = true;
        let csv = p.csv();
        let head = csv.lines().next().unwrap();
        // The pair's name holds a comma, so it is quoted.
        let mut quoted = false;
        let cols = 1 + head.chars().filter(|&c| {
            quoted ^= c == '"';
            c == ',' && !quoted
        }).count();
        assert_eq!(cols, 4, "{head}");
        assert!(head.contains("uo (m/s)") && head.contains("vo (m/s)"), "{head}");
        assert_eq!(csv.lines().count(), 3);
    }
}
