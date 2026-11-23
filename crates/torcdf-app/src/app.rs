//! Application state and wiring: open files, the selection, and every
//! background job. The UI thread never touches a file: opens, slice reads,
//! coordinate reads, text reads and large-slice statistics all run on other
//! threads and report back through channels, then wake the UI.

use std::io::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::time::Instant;

use eframe::egui;
use torcdf_core::{
    Attr, CoreError, DEFAULT_WORKERS, NcFile, RotateChoice, SliceService,
    default_cache_budget_bytes, pair_by_hand,
};

use crate::render::colormap::Colormap;
use crate::ui::{self, inspector, plot_area, tree};
use crate::view::defaults::ViewKind;
use crate::view::playback::{FPS_RANGE, Playback, Tick, frame_bytes};
use crate::view::popup::{Popup, PopupKind, PopupSet};
use crate::view::var_view::{Display, ImageState, VarView};

/// A CSV save: `None` when the dialog was cancelled, else the write result.
type SaveOutcome = Option<Result<(), String>>;

/// How long a pop-up notice stays in the plot area.
const NOTICE_TIME: std::time::Duration = std::time::Duration::from_secs(4);

/// Frames slower than this are logged (UI-thread stalls).
const SLOW_FRAME_MS: f64 = 20.0;

/// While playback waits for a frame, the UI looks again this often (reads
/// finishing wake it sooner).
const WAIT_RECHECK: std::time::Duration = std::time::Duration::from_millis(50);

/// The dimension to animate by default: the time axis when it has a slider,
/// else the first slider (0 when there is none).
fn default_play_dim(v: &VarView) -> usize {
    let dims = v.slider_dims();
    match v.time_dim {
        Some(t) if dims.contains(&t) => t,
        _ => dims.first().copied().unwrap_or(0),
    }
}

/// A result produced on a helper thread.
pub struct Job<T> {
    rx: Receiver<T>,
}

impl<T: Send + 'static> Job<T> {
    /// Runs `f` on a new thread; the UI is woken when it finishes.
    pub fn spawn(ctx: &egui::Context, f: impl FnOnce() -> T + Send + 'static) -> Job<T> {
        let (tx, rx) = channel();
        let ctx = ctx.clone();
        // If the thread cannot start, `tx` is dropped and `poll` reports it.
        let _ = std::thread::Builder::new().name("torcdf-job".into()).spawn(move || {
            if let Ok(v) = catch_unwind(AssertUnwindSafe(f)) {
                let _ = tx.send(v);
            }
            drop(tx);
            ctx.request_repaint();
        });
        Job { rx }
    }

    /// `Some(Ok)` once finished, `Some(Err)` if the job died, `None` while running.
    pub fn poll(&self) -> Option<Result<T, ()>> {
        match self.rx.try_recv() {
            Ok(v) => Some(Ok(v)),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(())),
        }
    }
}

/// Optional timing log for headless checks: `TORCDF_LOG=<file>` appends
/// timestamped lines (written by a logger thread, not the UI thread); with
/// it, `TORCDF_SELECT=<var path>` selects that variable when a file opens,
/// so first-plot times can be measured without clicks.
struct DebugLog {
    tx: Option<Sender<String>>,
    t0: Instant,
    select: Option<String>,
}

impl DebugLog {
    fn from_env() -> DebugLog {
        let tx = std::env::var_os("TORCDF_LOG").map(|path| {
            let (tx, rx) = channel::<String>();
            std::thread::spawn(move || {
                let file = std::fs::OpenOptions::new().create(true).append(true).open(path);
                if let Ok(mut f) = file {
                    for line in rx {
                        let _ = writeln!(f, "{line}");
                    }
                }
            });
            tx
        });
        let select = std::env::var("TORCDF_SELECT").ok().filter(|_| tx.is_some());
        DebugLog { tx, t0: Instant::now(), select }
    }

    fn enabled(&self) -> bool {
        self.tx.is_some()
    }

    fn log(&mut self, msg: impl std::fmt::Display) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(format!("[{:9.1} ms] {msg}", ms(self.t0)));
        }
    }
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

pub struct OpenFile {
    pub id: u64,
    pub name: String,
    pub nc: Arc<NcFile>,
}

pub struct Opening {
    pub name: String,
    path: PathBuf,
    started: Instant,
    job: Job<Result<NcFile, CoreError>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    File(u64),
    Group(u64, String),
    Var(u64, String),
}

/// What the panels asked for this frame; applied after drawing.
pub enum Action {
    OpenDialog,
    Select(Selection),
    CloseFile(u64),
    DismissError(usize),
    SliderMoved,
    SetKind(ViewKind),
    /// Pair `east` and `north` of a file by hand.
    PairWith { file: u64, east: String, north: String },
    /// Remove a pair (and its rotation variants) for this session.
    Unpair { file: u64, path: String },
    SetRotate(RotateChoice),
    FitView,
    /// A click on the map or image: pop up the series or profile there.
    OpenPopup { iy: usize, ix: usize, lonlat: Option<(f64, f64)> },
    ClosePopup(u64),
    SetPopupKind(u64, PopupKind),
    /// A click in a pop-up: move the slider of that dimension (plan P7).
    JumpTo { file: u64, dim_name: String, index: usize },
    SavePopupCsv(u64),
    /// Animate the playback dimension (locks the colour range as shown).
    Play,
    Pause,
    SetFps(f32),
    SetLoop(bool),
    SetPlayDim(usize),
    Exit,
}

fn file_name(p: &Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

pub struct TorApp {
    pub files: Vec<OpenFile>,
    next_id: u64,
    pub opening: Vec<Opening>,
    dialog: Option<Job<Vec<PathBuf>>>,
    pub errors: Vec<String>,
    pub selection: Option<Selection>,
    pub view: Option<VarView>,
    pub display: Display,
    pub popups: PopupSet,
    /// Pop-up id and its CSV save: `None` when the dialog was cancelled.
    saves: Vec<(u64, Job<SaveOutcome>)>,
    service: SliceService,
    /// The service's cache budget (read-ahead holds at most an eighth).
    cache_budget: usize,
    /// Animation of the current view; dropped with the view.
    pub playback: Option<Playback>,
    /// When a slider last moved, until the next upload (for the step log).
    slider_changed_at: Option<Instant>,
    log: DebugLog,
    first_image_logged: bool,
    ctx: egui::Context,
}

impl TorApp {
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> TorApp {
        let ctx = cc.egui_ctx.clone();
        let wake = ctx.clone();
        let cache_budget = default_cache_budget_bytes();
        let service = SliceService::new(
            cache_budget,
            DEFAULT_WORKERS,
            Arc::new(move || wake.request_repaint()),
        );
        let mut app = TorApp {
            files: Vec::new(),
            next_id: 1,
            opening: Vec::new(),
            dialog: None,
            errors: Vec::new(),
            selection: None,
            view: None,
            display: Display {
                colormap: Colormap::Viridis,
                log: false,
                lock: false,
                range: (0.0, 1.0),
            },
            popups: PopupSet::default(),
            saves: Vec::new(),
            service,
            cache_budget,
            playback: None,
            slider_changed_at: None,
            log: DebugLog::from_env(),
            first_image_logged: false,
            ctx,
        };
        app.log.log("window created");
        if cc.wgpu_render_state.is_none() {
            app.errors.push("GPU rendering is unavailable, so images cannot be drawn.".into());
        }
        if let Some(p) = path {
            app.open(p);
        }
        app
    }

    /// Opens `path` on a helper thread.
    fn open(&mut self, path: PathBuf) {
        self.log.log(format_args!("open {}", path.display()));
        let p = path.clone();
        let job = Job::spawn(&self.ctx, move || NcFile::open(&p));
        self.opening.push(Opening { name: file_name(&path), path, started: Instant::now(), job });
    }

    fn poll(&mut self) {
        let mut i = 0;
        while i < self.opening.len() {
            let Some(r) = self.opening[i].job.poll() else {
                i += 1;
                continue;
            };
            let o = self.opening.remove(i);
            match r {
                Ok(Ok(nc)) => {
                    self.log.log(format_args!("tree ready {:.1} ms after open", ms(o.started)));
                    let id = self.next_id;
                    self.next_id += 1;
                    self.files.push(OpenFile { id, name: o.name, nc: Arc::new(nc) });
                    if let Some(var) = self.log.select.take() {
                        self.select(Selection::Var(id, var));
                    }
                }
                Ok(Err(e)) => {
                    self.log.log(format_args!("open failed: {e}"));
                    self.errors.push(e.to_string());
                }
                Err(()) => self.errors.push(format!(
                    "cannot open {}: the open failed unexpectedly",
                    o.path.display()
                )),
            }
        }
        if let Some(r) = self.dialog.as_ref().and_then(Job::poll) {
            self.dialog = None;
            for p in r.unwrap_or_default() {
                self.open(p);
            }
        }
        self.popups.poll();
        let mut i = 0;
        while i < self.saves.len() {
            let Some(r) = self.saves[i].1.poll() else {
                i += 1;
                continue;
            };
            let (id, _) = self.saves.remove(i);
            let failed = match r {
                Ok(Some(Err(e))) => Some(e),
                Err(()) => Some("The CSV could not be saved.".to_string()),
                _ => None,
            };
            if let (Some(e), Some(p)) = (failed, self.popups.list.iter_mut().find(|p| p.id == id)) {
                p.error = Some(e);
            }
        }
        if let Some(v) = &mut self.view {
            let error_before = v.error.is_some();
            let (arrived, coords) = v.poll(&self.ctx);
            if arrived {
                let shape = v.shown.as_ref().map(|s| s.slice.shape.clone()).unwrap_or_default();
                self.log.log(format_args!(
                    "data ready {:.1} ms after request, shape {shape:?}",
                    ms(v.requested_at)
                ));
            }
            if let (true, Some(d)) = (coords, v.time_dim) {
                let n = v.var.shape[d];
                let label = |i: usize| {
                    let c = v.coords.dims.get(d).and_then(|c| c.as_ref()?.get(i).copied());
                    ui::slider_label(i, c, v.time_units.as_ref())
                };
                let (first, last) = (label(0), label(n.saturating_sub(1)));
                self.log.log(format_args!("time axis {}: {first} .. {last} ({n} steps)", v.var.dims[d]));
            }
            if !error_before
                && let Some(e) = &v.error
            {
                self.log.log(format_args!("view error: {e}"));
            }
        }
    }

    fn select(&mut self, sel: Selection) {
        if self.selection.as_ref() == Some(&sel) {
            return;
        }
        if let Selection::Var(id, path) = &sel {
            let found = self.files.iter().find(|f| f.id == *id);
            let pair = found.and_then(|f| Some((f.nc.clone(), f.nc.pair(path)?)));
            let var = found.and_then(|f| f.nc.meta.var(path).map(|v| (f.nc.clone(), v.clone())));
            // Dropping the old view drops its pending request (cancelling it).
            self.view = None;
            // Its playback (and so its read-ahead) goes with it.
            self.playback = None;
            if let Some((nc, pair)) = pair {
                self.log.log(format_args!("select {}", pair.path));
                let v = VarView::new_vector(&self.ctx, *id, nc, pair);
                self.show_view(v);
            } else if let Some((nc, var)) = var {
                self.log.log(format_args!("select {}", var.path));
                let v = VarView::new(&self.ctx, *id, nc, var);
                self.show_view(v);
            }
        }
        self.selection = Some(sel);
    }

    /// Reads the first slice of a new view and puts it on show.
    fn show_view(&mut self, mut v: VarView) {
        if let (true, Some(m)) = (self.log.enabled(), &mut v.map) {
            m.events = Some(Vec::new());
        }
        v.refresh(&self.ctx, &self.service);
        // A new view starts paused; the old view's read-ahead is dropped.
        self.playback = Some(Playback::new(default_play_dim(&v)));
        self.view = Some(v);
        self.display.lock = false;
        self.first_image_logged = false;
        self.slider_changed_at = None;
    }

    /// Advances the animation: keeps its read-ahead going and, when a frame
    /// is due and read, moves the slider to it. While the view is still
    /// taking in the frame on show, the next one waits.
    fn tick_playback(&mut self) {
        let (Some(pb), Some(v)) = (&mut self.playback, &mut self.view) else { return };
        let dims = v.slider_dims();
        if !dims.contains(&pb.dim) {
            // The view kind changed which dimensions have sliders.
            pb.stop();
            pb.dim = default_play_dim(v);
        }
        if !pb.playing {
            return;
        }
        let now = Instant::now();
        if v.loading() {
            pb.waiting = true;
            return;
        }
        let dim = pb.dim;
        let (len, current) = (v.var.shape[dim], v.slider_pos[dim]);
        let components = if v.vector.is_some() { 3 } else { 1 };
        let bytes = frame_bytes(&v.requests_at(&v.slider_pos), components);
        let base = v.slider_pos.clone();
        let requests = |i: usize| {
            let mut pos = base.clone();
            pos[dim] = i;
            v.requests_at(&pos)
        };
        let tick = pb.tick(now, current, len, bytes, self.cache_budget, &self.service, &v.nc, &requests);
        match tick {
            Tick::Show(i) => {
                v.slider_pos[dim] = i;
                let ready = pb.take_ready();
                v.refresh_with(&self.ctx, &self.service, &ready);
                self.slider_changed_at = Some(now);
            }
            Tick::Stop => pb.stop(),
            Tick::Idle | Tick::Wait => {}
        }
        if pb.playing {
            // While waiting, finished reads wake the UI; this is a fallback.
            let after = if pb.waiting { WAIT_RECHECK } else { pb.due_in(Instant::now()) };
            self.ctx.request_repaint_after(after);
        }
    }

    /// Shows `pair` again in place of the current view of the same pair
    /// (another rotation choice): slider positions and the map camera stay.
    fn reselect_pair(&mut self, pair: Arc<torcdf_core::VectorPair>) {
        let Some(old) = self.view.take() else { return };
        let mut v = VarView::new_vector(&self.ctx, old.file_id, old.nc.clone(), pair.clone());
        if v.slider_pos.len() == old.slider_pos.len() {
            v.slider_pos = old.slider_pos.clone();
        }
        v.image.view = old.image.view;
        v.image.fitted_to = old.image.fitted_to;
        v.image.user_moved = old.image.user_moved;
        if let (Some(new), Some(prev)) = (&mut v.map, &old.map) {
            new.inherit_view(prev);
        }
        if let (Some(new), Some(prev)) = (&mut v.arrows, &old.arrows) {
            (new.on, new.length) = (prev.on, prev.length);
        }
        // The slider range and colours stay as the user left them.
        let lock = self.display.lock;
        let settings = self.playback.as_ref().map(|p| (p.dim, p.fps, p.looping));
        self.selection = Some(Selection::Var(old.file_id, pair.path.clone()));
        self.show_view(v);
        self.display.lock = lock;
        // The new view starts paused, with the same playback settings.
        if let (Some(p), Some((dim, fps, looping))) = (&mut self.playback, settings) {
            (p.dim, p.fps, p.looping) = (dim, fps, looping);
        }
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::OpenDialog => {
                if self.dialog.is_none() {
                    // The native dialog blocks its thread, so it gets its own.
                    self.dialog = Some(Job::spawn(&self.ctx, || {
                        rfd::FileDialog::new()
                            .set_title("Open netCDF / HDF5 files")
                            .add_filter("netCDF / HDF5", &["nc", "nc4", "cdf", "h5", "hdf5", "he5"])
                            .add_filter("All files", &["*"])
                            .pick_files()
                            .unwrap_or_default()
                    }));
                }
            }
            Action::Select(sel) => self.select(sel),
            Action::CloseFile(id) => {
                let (closed, kept): (Vec<_>, Vec<_>) =
                    std::mem::take(&mut self.files).into_iter().partition(|f| f.id == id);
                self.files = kept;
                let view = self.view.take_if(|v| v.file_id == id);
                if view.is_some() {
                    // Its read-ahead stops with it.
                    self.playback = None;
                }
                let gone: Vec<u64> =
                    self.popups.list.iter().filter(|p| p.file_id == id).map(|p| p.id).collect();
                for p in gone {
                    self.popups.close(p);
                }
                // The last handle closes the file (I/O, and it may wait for
                // the netCDF lock), so it is dropped off the UI thread.
                std::thread::spawn(move || drop((closed, view)));
                let on_file = match &self.selection {
                    Some(Selection::File(f) | Selection::Group(f, _) | Selection::Var(f, _)) => {
                        *f == id
                    }
                    None => false,
                };
                if on_file {
                    self.selection = None;
                }
            }
            Action::DismissError(i) => {
                if i < self.errors.len() {
                    self.errors.remove(i);
                }
            }
            Action::SliderMoved => {
                if let Some(v) = &mut self.view {
                    v.refresh(&self.ctx, &self.service);
                    self.slider_changed_at = Some(Instant::now());
                }
            }
            Action::SetKind(kind) => {
                if let Some(v) = &mut self.view
                    && v.kind != kind
                {
                    v.kind = kind;
                    v.shown = None;
                    v.image = ImageState::default();
                    // The map keeps its camera and cell index, not its tile.
                    if let Some(m) = &mut v.map {
                        m.reset_tiles();
                    }
                    v.refresh(&self.ctx, &self.service);
                }
            }
            Action::PairWith { file, east, north } => {
                let Some(f) = self.files.iter().find(|f| f.id == file) else { return };
                match pair_by_hand(&f.nc.meta, &east, &north) {
                    Ok(p) => {
                        let p = f.nc.add_pair(p);
                        self.select(Selection::Var(file, p.path.clone()));
                    }
                    Err(e) => self.errors.push(e),
                }
            }
            Action::Unpair { file, path } => {
                let Some(f) = self.files.iter().find(|f| f.id == file) else { return };
                for tag in ["", "#on", "#off"] {
                    f.nc.remove_pair(&format!("{path}{tag}"));
                }
                let shown = matches!(&self.selection, Some(Selection::Var(id, p))
                    if *id == file && p.strip_prefix(&path).is_some_and(|t| t.is_empty() || t.starts_with('#')));
                if shown {
                    self.view = None;
                    self.playback = None;
                    self.selection = None;
                }
            }
            Action::SetRotate(choice) => {
                let Some(old) = self.view.as_ref().and_then(|v| Some((v.nc.clone(), v.vector.clone()?)))
                else {
                    return;
                };
                let (nc, pair) = old;
                let new = pair.with_rotate(choice);
                if new.path != pair.path {
                    let new = nc.add_pair(new);
                    self.reselect_pair(new);
                }
            }
            Action::FitView => {
                if let Some(v) = &mut self.view {
                    v.image.view = None;
                    v.image.user_moved = false;
                    if let Some(m) = &mut v.map {
                        m.request_fit();
                    }
                }
            }
            Action::OpenPopup { iy, ix, lonlat } => {
                let Some(v) = &mut self.view else { return };
                let id = self.popups.next_id();
                match Popup::open(v, iy, ix, lonlat, id, &self.service) {
                    Ok(p) => self.popups.place(p),
                    Err(e) => {
                        v.notice = Some(e);
                        v.notice_until = Some(Instant::now() + NOTICE_TIME);
                    }
                }
            }
            Action::ClosePopup(id) => self.popups.close(id),
            Action::SetPopupKind(id, kind) => {
                if let Some(p) = self.popups.list.iter_mut().find(|p| p.id == id) {
                    p.set_kind(kind, &self.service);
                }
            }
            Action::JumpTo { file, dim_name, index } => {
                let Some(v) = self.view.as_mut().filter(|v| v.file_id == file) else { return };
                let Some(d) = v.var.dims.iter().position(|n| *n == dim_name) else { return };
                if index >= v.var.shape[d] || !v.slider_dims().contains(&d) || v.slider_pos[d] == index {
                    return;
                }
                v.slider_pos[d] = index;
                v.refresh(&self.ctx, &self.service);
                self.slider_changed_at = Some(Instant::now());
            }
            Action::SavePopupCsv(id) => {
                let Some(p) = self.popups.list.iter().find(|p| p.id == id) else { return };
                if self.saves.iter().any(|(s, _)| *s == id) {
                    return;
                }
                let (csv, name) = (p.csv(), format!("{}.csv", p.var_name));
                // The dialog and the write block, so they get their own thread.
                let job = Job::spawn(&self.ctx, move || {
                    let path = rfd::FileDialog::new()
                        .set_title("Save the series as CSV")
                        .set_file_name(name)
                        .add_filter("CSV", &["csv"])
                        .save_file()?;
                    Some(std::fs::write(&path, csv).map_err(|e| format!("Cannot save {}: {e}", path.display())))
                });
                self.saves.push((id, job));
            }
            Action::Play => {
                let (Some(pb), Some(v)) = (&mut self.playback, &mut self.view) else { return };
                if !v.slider_dims().contains(&pb.dim) {
                    return;
                }
                // At the end without looping, play starts over from the first step.
                let d = pb.dim;
                if !pb.looping && v.slider_pos[d] + 1 >= v.var.shape[d] {
                    v.slider_pos[d] = 0;
                    v.refresh(&self.ctx, &self.service);
                    self.slider_changed_at = Some(Instant::now());
                }
                // Frames keep the colours they start with (unlocking stays possible).
                if let Some(range) = self.display.effective(v.shown.as_ref().map(|s| &s.stats)) {
                    self.display.range = range;
                    self.display.lock = true;
                }
                pb.start(Instant::now());
                self.ctx.request_repaint();
            }
            Action::Pause => {
                if let Some(pb) = &mut self.playback {
                    pb.stop();
                }
            }
            Action::SetFps(fps) => {
                if let Some(pb) = &mut self.playback
                    && fps.is_finite()
                {
                    pb.fps = fps.clamp(*FPS_RANGE.start(), *FPS_RANGE.end());
                }
            }
            Action::SetLoop(looping) => {
                if let Some(pb) = &mut self.playback {
                    pb.looping = looping;
                }
            }
            Action::SetPlayDim(d) => {
                let (Some(pb), Some(v)) = (&mut self.playback, &self.view) else { return };
                if pb.dim == d || !v.slider_dims().contains(&d) {
                    return;
                }
                // Read-ahead along the old dimension is dropped.
                let playing = pb.playing;
                pb.stop();
                pb.dim = d;
                if playing {
                    pb.start(Instant::now());
                }
            }
            Action::Exit => self.ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

}

/// Title and attributes of the selected item.
fn selected_attrs<'a>(files: &'a [OpenFile], sel: Option<&Selection>) -> Option<(String, &'a [Attr])> {
    let file = |id: &u64| files.iter().find(|f| f.id == *id);
    match sel? {
        Selection::File(id) => {
            let f = file(id)?;
            Some((format!("{} (global)", f.name), &f.nc.meta.root.attrs[..]))
        }
        Selection::Group(id, path) => {
            let g = find_group(&file(id)?.nc.meta.root, path)?;
            Some((format!("group {}", g.path), &g.attrs[..]))
        }
        Selection::Var(id, path) => {
            let v = file(id)?.nc.meta.var(path)?;
            Some((v.path.clone(), &v.attrs[..]))
        }
    }
}

fn find_group<'a>(g: &'a torcdf_core::Group, path: &str) -> Option<&'a torcdf_core::Group> {
    if g.path == path {
        return Some(g);
    }
    g.groups.iter().find_map(|s| find_group(s, path))
}

impl eframe::App for TorApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let frame_start = Instant::now();
        self.poll();
        self.tick_playback();
        let dropped: Vec<PathBuf> = ui.ctx().input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect()
        });
        for p in dropped {
            self.open(p);
        }

        let mut actions = Vec::new();
        egui::Panel::top("menu").show(ui, |ui| ui::menu_bar(ui, self.view.as_ref(), &mut actions));
        if !self.errors.is_empty() {
            egui::Panel::top("errors").show(ui, |ui| ui::error_banners(ui, &self.errors, &mut actions));
        }
        egui::Panel::left("tree").default_size(300.0).show(ui, |ui| {
            tree::show(ui, &self.files, &self.opening, self.selection.as_ref(), &mut actions)
        });
        // A vector pair is no variable of the file: its attributes come from the view.
        let pair_attrs = self.view.as_ref().filter(|v| v.vector.is_some()).map(|v| v.var.attrs.clone());
        let attrs = match (&self.selection, &pair_attrs) {
            (Some(Selection::Var(_, path)), Some(a)) => Some((path.clone(), &a[..])),
            _ => selected_attrs(&self.files, self.selection.as_ref()),
        };
        let var_selected = matches!(self.selection, Some(Selection::Var(..)));
        egui::Panel::right("inspector").default_size(300.0).show(ui, |ui| {
            inspector::show(ui, &mut self.display, self.view.as_mut(), var_selected, attrs, &mut actions)
        });
        if let Some(v) = &mut self.view
            && !v.slider_dims().is_empty()
        {
            let pb = self.playback.as_ref();
            egui::Panel::bottom("sliders").show(ui, |ui| ui::slider_strip(ui, v, pb, &mut actions));
        }
        let uploaded = egui::CentralPanel::default()
            .show(ui, |ui| {
                plot_area::show(
                    ui,
                    self.view.as_mut(),
                    &self.display,
                    frame.wgpu_render_state(),
                    &self.popups,
                    &mut actions,
                )
            })
            .inner;
        let current = self.view.as_ref().map(|v| (v.file_id, &v.var.dims[..], &v.slider_pos[..]));
        ui::popup::show_all(ui.ctx(), &mut self.popups, current, &mut actions);
        ui::drop_overlay(ui.ctx());
        // Detail reads start only after the map has planned them for a resting camera.
        if let Some(v) = &mut self.view {
            v.start_detail(&self.service);
        }

        if uploaded && let Some(t) = self.slider_changed_at.take() {
            self.log.log(format_args!("step shown {:.1} ms after slider change", ms(t)));
        }
        if uploaded && !self.first_image_logged {
            self.first_image_logged = true;
            if let Some(v) = &self.view {
                let range = self.display.effective(v.shown.as_ref().map(|s| &s.stats));
                if let (ViewKind::Map, Some(m)) = (v.kind, &v.map) {
                    self.log.log(format_args!(
                        "first map tile uploaded {:.1} ms after request; tile {:?}, camera {:?}, range {range:?}",
                        ms(v.requested_at),
                        m.overview.as_ref().map(|t| t.spec),
                        m.camera,
                    ));
                } else {
                    self.log.log(format_args!(
                        "first image uploaded {:.1} ms after request; grid {:?}, flip_y (north up) {}, range {range:?}",
                        ms(v.requested_at),
                        v.image.grid,
                        v.flip_y(),
                    ));
                }
            }
        }
        for a in actions {
            self.apply(a);
        }
        let took = ms(frame_start);
        if let Some(m) = self.view.as_mut().and_then(|v| v.map.as_mut())
            && let Some(events) = m.events.as_mut()
        {
            for e in std::mem::take(events) {
                self.log.log(e);
            }
        }
        if took > SLOW_FRAME_MS && self.log.enabled() {
            let doing = self.view.as_ref().map_or(String::new(), |v| {
                let activity = v.map.as_ref().map_or("", |m| m.activity);
                format!("view {:?}, map activity \"{activity}\", uploaded {uploaded}", v.kind)
            });
            self.log.log(format_args!("slow frame: {took:.1} ms ({doing})"));
        }
    }
}

#[cfg(test)]
impl TorApp {
    /// An app without a window, with `path` already open as file 1 (opened
    /// on the calling thread). Panics if the file cannot be opened.
    pub fn headless(path: &Path) -> TorApp {
        let ctx = egui::Context::default();
        let cache_budget = default_cache_budget_bytes();
        let service = SliceService::new(
            cache_budget,
            DEFAULT_WORKERS,
            Arc::new(|| {}),
        );
        let nc = NcFile::open(path).expect("fixture opens");
        TorApp {
            files: vec![OpenFile { id: 1, name: file_name(path), nc: Arc::new(nc) }],
            next_id: 2,
            opening: Vec::new(),
            dialog: None,
            errors: Vec::new(),
            selection: None,
            view: None,
            display: Display {
                colormap: Colormap::Viridis,
                log: false,
                lock: false,
                range: (0.0, 1.0),
            },
            popups: PopupSet::default(),
            saves: Vec::new(),
            service,
            cache_budget,
            playback: None,
            slider_changed_at: None,
            log: DebugLog { tx: None, t0: Instant::now(), select: None },
            first_image_logged: false,
            ctx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torcdf_core::fixtures::vectors::write_nemo_like;

    #[test]
    fn rotate_toggle_reselects() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        let pair = app.files[0].nc.pairs().remove(0);
        app.apply(Action::Select(Selection::Var(1, pair.path.clone())));
        app.view.as_mut().unwrap().slider_pos[0] = 1;
        app.apply(Action::SetRotate(RotateChoice::Off));
        let Some(Selection::Var(_, path)) = &app.selection else { panic!("no selection") };
        assert!(path.ends_with("#off"), "{path}");
        let v = app.view.as_ref().expect("view");
        assert_eq!(v.var.path, *path);
        assert_eq!(v.slider_pos[0], 1);
        // The tagged variant is registered; the tree still shows one row.
        assert!(app.files[0].nc.pair(path).is_some());
    }

    #[test]
    fn pair_by_hand_and_unpair() {
        let dir = tempfile::tempdir().unwrap();
        let path = torcdf_core::fixtures::vectors::write_noclue_renamed(dir.path());
        let mut app = TorApp::headless(&path);
        assert!(app.files[0].nc.pairs().is_empty());
        app.apply(Action::PairWith { file: 1, east: "/aa".into(), north: "/bb".into() });
        let p = app.files[0].nc.pairs().remove(0);
        assert_eq!(app.selection, Some(Selection::Var(1, p.path.clone())));
        app.apply(Action::Unpair { file: 1, path: p.path.clone() });
        assert!(app.files[0].nc.pairs().is_empty());
        assert!(app.view.is_none() && app.selection.is_none());
        app.apply(Action::PairWith { file: 1, east: "/aa".into(), north: "/aa".into() });
        assert_eq!(app.errors.len(), 1);
    }

    fn uo_app(dir: &Path) -> TorApp {
        let mut app = TorApp::headless(&torcdf_core::fixtures::write_nc4(dir));
        app.apply(Action::Select(Selection::Var(1, "/ocean/uo".into())));
        app
    }

    #[test]
    fn jump_sets_matching_slider() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let before = app.view.as_ref().unwrap().requested_at;
        std::thread::sleep(std::time::Duration::from_millis(5));
        app.apply(Action::JumpTo { file: 2, dim_name: "time".into(), index: 9 });
        assert_eq!(app.view.as_ref().unwrap().slider_pos[0], 0);
        app.apply(Action::JumpTo { file: 1, dim_name: "nope".into(), index: 9 });
        assert_eq!(app.view.as_ref().unwrap().slider_pos[0], 0);
        assert_eq!(app.view.as_ref().unwrap().requested_at, before);
        app.apply(Action::JumpTo { file: 1, dim_name: "time".into(), index: 9 });
        let v = app.view.as_ref().unwrap();
        assert_eq!(v.slider_pos[0], 9);
        assert!(v.requested_at > before, "the view was not refreshed");
        // Past the end: ignored.
        app.apply(Action::JumpTo { file: 1, dim_name: "time".into(), index: 999 });
        assert_eq!(app.view.as_ref().unwrap().slider_pos[0], 9);
    }

    #[test]
    fn play_locks_range() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let t0 = Instant::now();
        while app.view.as_ref().unwrap().shown.is_none() {
            assert!(t0.elapsed().as_secs() < 20, "the first slice never arrived");
            std::thread::sleep(std::time::Duration::from_millis(2));
            app.poll();
        }
        let stats = app.view.as_ref().unwrap().shown.as_ref().map(|s| s.stats);
        let range = app.display.effective(stats.as_ref()).expect("a range");
        assert!(!app.display.lock);
        app.apply(Action::Play);
        assert!(app.display.lock);
        assert_eq!(app.display.range, range);
        assert!(app.playback.as_ref().is_some_and(|p| p.playing));
        app.apply(Action::Pause);
        assert!(!app.playback.as_ref().unwrap().playing);
        // Another variable drops the playback.
        app.apply(Action::Play);
        app.apply(Action::Select(Selection::Var(1, "/ocean/vo".into())));
        assert!(app.playback.as_ref().is_none_or(|p| !p.playing));
    }

    #[test]
    fn open_popup_action_places() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        app.apply(Action::OpenPopup { iy: 4, ix: 7, lonlat: None });
        assert_eq!(app.popups.list.len(), 1);
        let first = app.popups.list[0].id;
        app.apply(Action::OpenPopup { iy: 5, ix: 6, lonlat: None });
        assert_eq!(app.popups.list.len(), 1);
        assert_ne!(app.popups.list[0].id, first);
        app.popups.list[0].pinned = true;
        app.apply(Action::OpenPopup { iy: 1, ix: 1, lonlat: None });
        assert_eq!(app.popups.list.len(), 2);
        // Off the grid: nothing opens, a notice appears.
        app.apply(Action::OpenPopup { iy: 99, ix: 99, lonlat: None });
        assert_eq!(app.popups.list.len(), 2);
        assert!(app.view.as_ref().unwrap().notice.is_some());
        let id = app.popups.list[1].id;
        app.apply(Action::ClosePopup(id));
        assert_eq!(app.popups.list.len(), 1);
    }
}
