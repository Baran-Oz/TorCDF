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
    Attr, CoreError, DEFAULT_WORKERS, DirConvention, NcFile, PairForm, RotateChoice, SliceService, VarInfo,
    VectorPair, default_cache_budget_bytes, layer_path, pair_by_hand, pair_magdir_by_hand,
};

use crate::render::colormap::Colormap;
use crate::ui::{self, inspector, plot_area, tree};
use crate::view::arrow_layer::{ArrowLayer, ArrowMode, PICK_TWO};
use crate::view::defaults::ViewKind;
use crate::view::playback::{FPS_RANGE, Playback, Tick, frame_bytes};
use crate::view::popup::{Popup, PopupKind, PopupSet};
use crate::view::var_view::{Display, ImageState, VarView};

/// A CSV save: `None` when the dialog was cancelled, else the write result.
type SaveOutcome = Option<Result<(), String>>;

/// How long a pop-up click message (`VarView::flash`) stays in the plot area.
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
#[cfg_attr(not(test), expect(dead_code, reason = "the Plot tab's Arrows section sends the arrow actions"))]
pub enum Action {
    OpenDialog,
    Select(Selection),
    CloseFile(u64),
    DismissError(usize),
    SliderMoved,
    SetKind(ViewKind),
    /// Remove a pair (and its rotation variants) for this session.
    Unpair { file: u64, path: String },
    SetRotate(RotateChoice),
    /// Arrows section: Off, x & y, or magnitude & direction.
    SetArrowMode(ArrowMode),
    /// Arrows section drop-down: slot 0 = x / magnitude, 1 = y / direction.
    PickArrowVar { slot: usize, path: String },
    /// Direction convention of the arrows' magnitude/direction pair.
    SetConvention(DirConvention),
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
    /// The window's arrow slot (the Plot tab's Arrows section).
    pub arrow_layer: Option<ArrowLayer>,
    /// Pop-up id and its CSV save: `None` when the dialog was cancelled.
    saves: Vec<(u64, Job<SaveOutcome>)>,
    service: SliceService,
    /// The service's cache budget (read-ahead holds at most an eighth).
    cache_budget: usize,
    /// Animation of the current view; dropped with the view.
    pub playback: Option<Playback>,
    /// When a slider last moved and the positions it moved to, until a
    /// slice for those positions is uploaded (for the step log).
    slider_changed_at: Option<(Instant, Vec<usize>)>,
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
            arrow_layer: None,
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
                // A pair's own entry draws its own arrows.
                self.arrow_layer = Some(ArrowLayer::of_pair(*id, pair.clone()));
                let v = VarView::new_vector(&self.ctx, *id, nc, pair);
                self.show_view(v);
            } else if let Some((nc, var)) = var {
                self.log.log(format_args!("select {}", var.path));
                // Arrows never carry over between files; while Off, the
                // pre-fill follows the selected variable.
                let keep = self.arrow_layer.as_ref().is_some_and(|l| l.file == *id && l.mode != ArrowMode::Off);
                if !keep {
                    self.arrow_layer = Some(ArrowLayer::fresh(*id, &nc, &var));
                }
                let v = self.x_view(*id, nc, var);
                self.show_view(v);
            }
        }
        self.selection = Some(sel);
    }

    /// The view of plain variable `x` under the arrow slot: layered when
    /// the slot's pair fits `x`'s grid; plain otherwise, with a note when
    /// arrows are on but hidden.
    fn x_view(&self, file: u64, nc: Arc<NcFile>, x: VarInfo) -> VarView {
        match self.slot_pair(file) {
            Some(p) if nc.layer_fits(&x.path, &p) => VarView::new_layered(&self.ctx, file, nc, x, p),
            Some(_) => {
                let note = hidden_note(&x.name);
                let mut v = VarView::new(&self.ctx, file, nc, x);
                v.notice = Some(match v.notice.take() {
                    Some(n) => format!("{n}\n{note}"),
                    None => note,
                });
                v
            }
            None => VarView::new(&self.ctx, file, nc, x),
        }
    }

    /// The pair the slot draws on `file`, when arrows are on.
    fn slot_pair(&self, file: u64) -> Option<Arc<VectorPair>> {
        self.arrow_layer.as_ref().filter(|l| l.file == file && l.mode != ArrowMode::Off)?.pair.clone()
    }

    /// The coloured plain variable of the current view (plain or layered);
    /// `None` for a pair's own entry or no view.
    fn current_x(&self) -> Option<(u64, Arc<NcFile>, VarInfo)> {
        let v = self.view.as_ref()?;
        if v.base.is_none() && v.vector.is_some() {
            return None;
        }
        let x = v.nc.meta.var(v.plain_path())?.clone();
        Some((v.file_id, v.nc.clone(), x))
    }

    /// Brings the current view in line with the arrow slot: a plain or
    /// layered view of the coloured variable, or a pair's own entry.
    fn show_slot(&mut self) {
        let Some(v) = &self.view else { return };
        if let Some((file, nc, x)) = self.current_x() {
            let target = match self.slot_pair(file) {
                Some(p) if nc.layer_fits(&x.path, &p) => layer_path(&x.path, &p.path),
                _ => x.path.clone(),
            };
            let hidden = self.slot_pair(file).is_some() && target == x.path;
            let shown_hidden = v.notice.as_deref().is_some_and(|n| n.contains(&hidden_note(&x.name)));
            if target != v.var.path || hidden != shown_hidden {
                let new = self.x_view(file, nc, x);
                self.replace_view(new);
            }
            return;
        }
        // A pair's own entry: it shows the slot's pair, or no arrows when Off.
        let (file, nc) = (v.file_id, v.nc.clone());
        let Some(l) = self.arrow_layer.as_ref().filter(|l| l.file == file) else { return };
        let on = l.mode != ArrowMode::Off;
        match l.pair.clone() {
            Some(p) if on && p.path != v.var.path => {
                let new = VarView::new_vector(&self.ctx, file, nc, p);
                self.replace_view(new);
            }
            _ => {
                if let Some(a) = self.view.as_mut().and_then(|v| v.arrows.as_mut()) {
                    a.on = on;
                }
            }
        }
    }

    /// Pairs the slot's picks in its mode and shows the result. With
    /// `replace`, the slot's previous pair is unregistered (with its
    /// rotation variants) once the new one is made. Any failure leaves the
    /// slot without a pair and says why.
    fn pair_from_picks(&mut self, replace: bool) {
        let x = self.current_x();
        let Some(l) = self.arrow_layer.as_mut() else { return };
        let Some(nc) = self.files.iter().find(|f| f.id == l.file).map(|f| f.nc.clone()) else { return };
        let old = l.pair.take();
        let [Some(a), Some(b)] = l.picks.clone() else {
            l.problem = Some(PICK_TWO.into());
            self.show_slot();
            return;
        };
        let mode = l.mode;
        // A registered pair of these two in this form is used as it is
        // (a detected pair keeps its settings), preferring the one on show
        // (a pair's own entry keeps its rotation variant).
        let same = |p: &VectorPair| p.east == a && p.north == b && ArrowMode::of_form(p.form) == mode;
        let shown = self.view.as_ref().and_then(|v| v.vector.clone());
        let existing = shown
            .filter(|p| same(p) && nc.pair(&p.path).is_some())
            .or_else(|| nc.pairs().into_iter().find(|p| same(p)));
        let (pair, registered) = match existing {
            Some(p) => (p, true),
            None => {
                let made = match mode {
                    ArrowMode::EastNorth => pair_by_hand(&nc.meta, &a, &b),
                    ArrowMode::MagDir => pair_magdir_by_hand(&nc.meta, &a, &b),
                    ArrowMode::Off => return,
                };
                match made {
                    Ok(p) => (Arc::new(p), false),
                    Err(e) => {
                        l.problem = Some(e);
                        self.show_slot();
                        return;
                    }
                }
            }
        };
        // Nothing is registered for a pair off the coloured variable's grid.
        if let Some((_, _, x)) = &x
            && !nc.layer_fits(&x.path, &pair)
        {
            l.problem = Some(off_grid(&x.name));
            self.show_slot();
            return;
        }
        if let Some(o) = old.filter(|o| replace && family(&o.path) != family(&pair.path)) {
            remove_family(&nc, &o.path);
        }
        let pair = if registered { pair } else { register(&nc, Arc::unwrap_or_clone(pair)) };
        l.pair = Some(pair);
        l.problem = None;
        l.remember_picks();
        self.show_slot();
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
        let budget = self.cache_budget;
        if v.loading() {
            // The frame on show is still being taken in: read ahead, but hold the step.
            pb.keep_ahead(now, current, len, bytes, budget, &self.service, &v.nc, &requests);
            let after = if pb.waiting { WAIT_RECHECK } else { pb.due_in(now) };
            self.ctx.request_repaint_after(after);
            return;
        }
        let tick = pb.tick(now, current, len, bytes, budget, &self.service, &v.nc, &requests);
        match tick {
            Tick::Show(i) => {
                v.slider_pos[dim] = i;
                let ready = pb.take_ready();
                v.refresh_with(&self.ctx, &self.service, &ready);
                self.slider_changed_at = Some((now, v.slider_pos.clone()));
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

    /// Milliseconds from the last slider change to this upload, once the
    /// slice on show is the one for the positions moved to (an upload of an
    /// earlier step, whose read finished late, is not counted).
    fn step_shown(&mut self, uploaded: bool) -> Option<f64> {
        let (t, pos) = self.slider_changed_at.as_ref()?;
        if !uploaded || !self.view.as_ref()?.shows_position(pos) {
            return None;
        }
        let took = ms(*t);
        self.slider_changed_at = None;
        Some(took)
    }

    /// Shows `v` in place of the current view of the same grid (another
    /// rotation, convention or arrow layer): slider positions, the camera,
    /// the arrow length, the colour lock and the playback settings stay.
    fn replace_view(&mut self, mut v: VarView) {
        self.selection = Some(Selection::Var(v.file_id, v.plain_path().to_string()));
        let Some(old) = self.view.take() else {
            self.show_view(v);
            return;
        };
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
            new.length = prev.length;
            // A pair's own entry keeps its switch (a layered view is on).
            if v.base.is_none() {
                new.on = prev.on;
            }
        }
        // The slider range and colours stay as the user left them.
        let lock = self.display.lock;
        let settings = self.playback.as_ref().map(|p| (p.dim, p.fps, p.looping));
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
                self.arrow_layer.take_if(|l| l.file == id);
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
                    self.slider_changed_at = Some((Instant::now(), v.slider_pos.clone()));
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
            Action::Unpair { file, path } => {
                let Some(f) = self.files.iter().find(|f| f.id == file) else { return };
                for tag in ["", "#on", "#off"] {
                    f.nc.remove_pair(&format!("{path}{tag}"));
                }
                let of_it = |p: &str| p.strip_prefix(&path).is_some_and(|t| t.is_empty() || t.starts_with('#'));
                let shown = matches!(&self.selection, Some(Selection::Var(id, p)) if *id == file && of_it(p));
                if shown {
                    self.view = None;
                    self.playback = None;
                    self.selection = None;
                }
                // The live arrow layer of it turns Off; a layered view of
                // it becomes the plain view of its variable.
                if let Some(l) = self.arrow_layer.as_mut().filter(|l| l.file == file)
                    && l.pair.as_ref().is_some_and(|p| of_it(&p.path))
                {
                    l.mode = ArrowMode::Off;
                    l.pair = None;
                    self.show_slot();
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
                    // The slot follows the pair drawn (rotation is per pair).
                    if let Some(l) = self.arrow_layer.as_mut()
                        && l.pair.as_ref().is_some_and(|p| p.path == pair.path)
                    {
                        l.pair = Some(new.clone());
                    }
                    let (file, layered) = self.view.as_ref().map_or((0, false), |v| (v.file_id, v.base.is_some()));
                    if layered {
                        if let Some((file, nc, x)) = self.current_x() {
                            let v = VarView::new_layered(&self.ctx, file, nc, x, new);
                            self.replace_view(v);
                        }
                    } else {
                        let v = VarView::new_vector(&self.ctx, file, nc, new);
                        self.replace_view(v);
                    }
                }
            }
            Action::SetArrowMode(mode) => {
                let Some(l) = self.arrow_layer.as_mut() else { return };
                if l.mode == mode {
                    return;
                }
                l.mode = mode;
                // The old pair stays registered; the slot no longer draws it.
                l.pair = None;
                if mode == ArrowMode::Off {
                    l.problem = l.missing_picks().then(|| PICK_TWO.to_string());
                    self.show_slot();
                } else {
                    // Each form starts from its own picks.
                    l.load_prefill(mode);
                    self.pair_from_picks(false);
                }
            }
            Action::PickArrowVar { slot, path } => {
                let Some(l) = self.arrow_layer.as_mut() else { return };
                if slot > 1 || (l.picks[slot].as_deref() == Some(path.as_str()) && l.pair.is_some()) {
                    return;
                }
                l.picks[slot] = Some(path);
                // While Off the picks are only kept.
                if l.mode != ArrowMode::Off {
                    self.pair_from_picks(true);
                }
            }
            Action::SetConvention(c) => {
                let Some(l) = self.arrow_layer.as_mut() else { return };
                let Some(old) = l.pair.clone().filter(|p| matches!(p.form, PairForm::MagDir(_))) else { return };
                let Some(nc) = self.files.iter().find(|f| f.id == l.file).map(|f| f.nc.clone()) else { return };
                let new = old.with_convention(c);
                if new == *old {
                    return;
                }
                // The pair is replaced at its new path, never added beside it.
                remove_family(&nc, &old.path);
                l.pair = Some(register(&nc, new));
                self.show_slot();
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
                    Err(e) => v.set_flash(e, NOTICE_TIME),
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
                self.slider_changed_at = Some((Instant::now(), v.slider_pos.clone()));
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
                    self.slider_changed_at = Some((Instant::now(), v.slider_pos.clone()));
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

/// The note on a plain view while the arrow layer cannot sit on `var`.
fn hidden_note(var: &str) -> String {
    format!("Arrows hidden: {var} is on a different grid")
}

/// Why picked variables were not paired over `var`.
fn off_grid(var: &str) -> String {
    format!("These two do not sit on {var}'s grid")
}

/// A pair path without its rotation tag.
fn family(path: &str) -> &str {
    path.strip_suffix("#on").or_else(|| path.strip_suffix("#off")).unwrap_or(path)
}

/// Unregisters the pair at `path` with its rotation variants.
fn remove_family(nc: &NcFile, path: &str) {
    let base = family(path);
    for tag in ["", "#on", "#off"] {
        nc.remove_pair(&format!("{base}{tag}"));
    }
}

/// Registers `p`; a rotation variant also registers its untagged pair,
/// which is the tree's row for it.
fn register(nc: &NcFile, p: VectorPair) -> Arc<VectorPair> {
    if p.rotate != RotateChoice::Auto {
        nc.add_pair(p.with_rotate(RotateChoice::Auto));
    }
    nc.add_pair(p)
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
        let pair_attrs =
            self.view.as_ref().filter(|v| v.vector.is_some() && v.base.is_none()).map(|v| v.var.attrs.clone());
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

        if let Some(took) = self.step_shown(uploaded) {
            self.log.log(format_args!("step shown {took:.1} ms after slider change"));
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
            arrow_layer: None,
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
    use crate::view::arrow_layer::ArrowMode;
    use std::time::Duration;
    use torcdf_core::fixtures::vectors::{write_arome_like, write_nemo_like};
    use torcdf_core::{AngleRef, AngleUnit, DirConvention, Guessed, SliceRequest, Towards, layer_path};

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
        app.apply(Action::Select(Selection::Var(1, "/h".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!(slot.problem.as_deref(), Some("Pick two variables"));
        assert!(slot.pair.is_none());
        app.apply(Action::PickArrowVar { slot: 0, path: "/aa".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/bb".into() });
        let p = app.files[0].nc.pairs().remove(0);
        assert_eq!((p.east.as_str(), p.north.as_str()), ("/aa", "/bb"));
        assert_eq!(app.selection, Some(Selection::Var(1, "/h".into())));
        assert_eq!(app.view.as_ref().unwrap().var.path, layer_path("/h", &p.path));
        // Its own tree entry, then Unpair: the view goes with the pair.
        app.apply(Action::Select(Selection::Var(1, p.path.clone())));
        app.apply(Action::Unpair { file: 1, path: p.path.clone() });
        assert!(app.files[0].nc.pairs().is_empty());
        assert!(app.view.is_none() && app.selection.is_none());
        // A variable cannot be paired with itself: the slot says why.
        app.apply(Action::Select(Selection::Var(1, "/h".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        app.apply(Action::PickArrowVar { slot: 0, path: "/aa".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/aa".into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert!(slot.problem.is_some() && slot.pair.is_none());
        assert!(app.files[0].nc.pairs().is_empty());
        assert!(app.errors.is_empty());
    }

    const AROME_PAIR: &str = "/wind_speed@wind_direction~from.cwn.deg";

    /// AROME-like file 1, `/wind_speed` coloured, its own magnitude/direction
    /// pair as arrows (picked in the Plot tab).
    fn arome_layer(dir: &Path) -> TorApp {
        let mut app = TorApp::headless(&write_arome_like(dir));
        app.apply(Action::Select(Selection::Var(1, "/wind_speed".into())));
        app.apply(Action::SetArrowMode(ArrowMode::MagDir));
        app.apply(Action::PickArrowVar { slot: 0, path: "/wind_speed".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/wind_direction".into() });
        app
    }

    #[test]
    fn magdir_layer_from_picks() {
        let dir = tempfile::tempdir().unwrap();
        let app = arome_layer(dir.path());
        let v = app.view.as_ref().expect("view");
        assert_eq!(v.var.path, format!("/wind_speed|{AROME_PAIR}"));
        assert_eq!(v.base.as_deref(), Some("/wind_speed"));
        assert_eq!(v.plain_path(), "/wind_speed");
        assert!(v.arrows.as_ref().is_some_and(|a| a.on));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!(slot.mode, ArrowMode::MagDir);
        assert_eq!(slot.pair.as_ref().map(|p| p.path.as_str()), Some(AROME_PAIR));
        assert_eq!(slot.problem, None);
        assert_eq!(app.selection, Some(Selection::Var(1, "/wind_speed".into())));
    }

    /// Each form has its own pre-fill: x & y has none on this file, and
    /// switching to it never pairs speed/direction as east/north.
    #[test]
    fn mode_switch_uses_that_forms_picks() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_arome_like(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/wind_speed".into())));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!((slot.file, slot.mode), (1, ArrowMode::Off));
        assert_eq!(slot.picks, [Some("/wind_speed".to_string()), Some("/wind_direction".to_string())]);
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.picks, [None, None]);
        assert_eq!(slot.problem.as_deref(), Some("Pick two variables"));
        assert!(slot.pair.is_none());
        assert_eq!(app.view.as_ref().unwrap().var.path, "/wind_speed");
        assert_eq!(app.files[0].nc.pairs().len(), 1);
        app.apply(Action::SetArrowMode(ArrowMode::MagDir));
        assert_eq!(app.view.as_ref().unwrap().var.path, format!("/wind_speed|{AROME_PAIR}"));
        app.apply(Action::SetArrowMode(ArrowMode::Off));
        let v = app.view.as_ref().unwrap();
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/wind_speed", None));
        assert!(v.vector.is_none());
        assert_eq!(app.arrow_layer.as_ref().unwrap().mode, ArrowMode::Off);
    }

    #[test]
    fn layer_over_own_component() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        let pos = app.view.as_ref().unwrap().slider_pos.clone();
        wait_shown(&mut app, &pos);
        let v = app.view.as_ref().unwrap();
        let shown = v.shown.as_ref().expect("shown");
        assert!(shown.slice.vector.is_some(), "the layer carries the arrows");
        let plain = SliceRequest { var: "/wind_speed".into(), ..shown.req.clone() };
        let speed = v.nc.read_slice(&plain).expect("plain read");
        let same = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
        };
        assert!(same(&shown.slice.data, &speed.data), "colours are wind_speed's");
    }

    #[test]
    fn layer_kept_on_same_grid_hidden_on_other() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        let pair = app.files[0].nc.pairs().remove(0);
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        let layered = layer_path("/thetao", &pair.path);
        assert_eq!(app.view.as_ref().unwrap().var.path, layered);
        // uo sits on the u-grid: plain, with the note; the slot stays.
        app.apply(Action::Select(Selection::Var(1, "/uo".into())));
        let v = app.view.as_ref().unwrap();
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/uo", None));
        assert!(v.vector.is_none());
        assert_eq!(v.notice.as_deref(), Some("Arrows hidden: uo is on a different grid"));
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.mode, ArrowMode::EastNorth);
        assert_eq!(slot.pair.as_ref().map(|p| p.path.clone()), Some(pair.path.clone()));
        // Back on the centre grid: the arrows come back.
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        assert_eq!(app.view.as_ref().unwrap().var.path, layered);
    }

    /// A pair whose centre is not the coloured variable's grid is refused.
    #[test]
    fn picks_off_grid_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        let before: Vec<String> = app.files[0].nc.pairs().iter().map(|p| p.path.clone()).collect();
        app.apply(Action::Select(Selection::Var(1, "/uo".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        assert_eq!(app.arrow_layer.as_ref().unwrap().problem.as_deref(), Some("Pick two variables"));
        app.apply(Action::PickArrowVar { slot: 0, path: "/uo".into() });
        app.apply(Action::PickArrowVar { slot: 1, path: "/vo".into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.problem.as_deref(), Some("These two do not sit on uo's grid"));
        assert!(slot.pair.is_none());
        let after: Vec<String> = app.files[0].nc.pairs().iter().map(|p| p.path.clone()).collect();
        assert_eq!(after, before);
        assert_eq!(app.view.as_ref().unwrap().var.path, "/uo");
    }

    #[test]
    fn convention_change_replaces_pair() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        let to = DirConvention {
            towards: Towards::To,
            reference: AngleRef::ClockwiseFromNorth,
            unit: AngleUnit::Degrees,
            guessed: Guessed::default(),
        };
        app.apply(Action::SetConvention(to));
        let nc = app.files[0].nc.clone();
        assert!(nc.pair(AROME_PAIR).is_none(), "the old path is gone");
        let new = app.arrow_layer.as_ref().unwrap().pair.clone().expect("pair");
        assert!(new.path.ends_with("~to.cwn.deg"), "{}", new.path);
        assert!(nc.pair(&new.path).is_some());
        assert_eq!(nc.pairs().len(), 1);
        assert_eq!(app.view.as_ref().unwrap().var.path, layer_path("/wind_speed", &new.path));
    }

    #[test]
    fn other_file_clears_slot() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        let nemo = NcFile::open(&write_nemo_like(dir.path())).unwrap();
        app.files.push(OpenFile { id: 2, name: "nemo.nc".into(), nc: Arc::new(nemo) });
        app.apply(Action::Select(Selection::Var(2, "/thetao".into())));
        let slot = app.arrow_layer.as_ref().expect("slot");
        assert_eq!((slot.file, slot.mode), (2, ArrowMode::Off));
        assert!(slot.pair.is_none());
        let v = app.view.as_ref().unwrap();
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/thetao", None));
        assert!(v.vector.is_none());
    }

    #[test]
    fn unpair_turns_layer_off() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = arome_layer(dir.path());
        app.apply(Action::Unpair { file: 1, path: AROME_PAIR.into() });
        let slot = app.arrow_layer.as_ref().unwrap();
        assert_eq!(slot.mode, ArrowMode::Off);
        assert!(slot.pair.is_none());
        let v = app.view.as_ref().expect("the plain view");
        assert_eq!((v.var.path.as_str(), v.base.as_deref()), ("/wind_speed", None));
        assert!(v.vector.is_none());
        assert_eq!(app.selection, Some(Selection::Var(1, "/wind_speed".into())));
    }

    /// The AROME-like file has no time or vertical axis to pop up along, so
    /// this uses the NEMO-like `thetao` (with `uo`/`vo` arrows).
    #[test]
    fn popup_on_layer_reads_plain_series() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = TorApp::headless(&write_nemo_like(dir.path()));
        app.apply(Action::Select(Selection::Var(1, "/thetao".into())));
        app.apply(Action::SetArrowMode(ArrowMode::EastNorth));
        assert!(app.view.as_ref().unwrap().base.is_some(), "layered");
        app.apply(Action::OpenPopup { iy: 1, ix: 1, lonlat: None });
        assert_eq!(app.popups.list.len(), 1);
        let p = &app.popups.list[0];
        assert_eq!(p.var_path, "/thetao");
        assert!(!p.has_components());
        let t0 = Instant::now();
        while app.popups.list[0].values.iter().any(Option::is_none) && app.popups.list[0].error.is_none() {
            assert!(t0.elapsed().as_secs() < 20, "the series never arrived");
            std::thread::sleep(Duration::from_millis(2));
            app.popups.poll();
        }
        let p = &app.popups.list[0];
        assert_eq!(p.error, None);
        // thetao = 10 + 0.01 (p + q) at every time step.
        assert_eq!(p.values.len(), 2);
        assert!(p.values.iter().all(|v| v.is_some_and(|x| (x - 10.02).abs() < 1e-4)), "{:?}", p.values);
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

    fn wait_shown(app: &mut TorApp, pos: &[usize]) {
        let t0 = Instant::now();
        while !app.view.as_ref().unwrap().shows_position(pos) {
            assert!(t0.elapsed().as_secs() < 20, "the slice for {pos:?} never arrived");
            std::thread::sleep(std::time::Duration::from_millis(2));
            app.poll();
        }
    }

    #[test]
    fn step_log_times_the_step_moved_to() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        wait_shown(&mut app, &[0, 0, 0]);
        // The slider moved to step 1, but the upload is of step 0 (a read
        // that finished late): not logged, and the change stays pending.
        app.slider_changed_at = Some((Instant::now(), vec![1, 0, 0]));
        assert_eq!(app.step_shown(true), None);
        assert!(app.slider_changed_at.is_some());
        app.apply(Action::JumpTo { file: 1, dim_name: "time".into(), index: 1 });
        wait_shown(&mut app, &[1, 0, 0]);
        assert_eq!(app.step_shown(false), None, "no upload this frame");
        assert!(app.step_shown(true).is_some(), "step 1 is uploaded");
        assert!(app.slider_changed_at.is_none());
        assert_eq!(app.step_shown(true), None, "logged once");
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
        eprintln!("FLASH {:?}", app.view.as_ref().unwrap().flash);
        assert_eq!(app.popups.list.len(), 2);
        // Off the grid: nothing opens, a passing message appears.
        app.apply(Action::OpenPopup { iy: 99, ix: 99, lonlat: None });
        assert_eq!(app.popups.list.len(), 2);
        assert!(app.view.as_ref().unwrap().flash.is_some());
        let id = app.popups.list[1].id;
        app.apply(Action::ClosePopup(id));
        assert_eq!(app.popups.list.len(), 1);
    }

    #[test]
    fn failed_popup_click_keeps_the_notice() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = uo_app(dir.path());
        let why = "Shown as an image: this variable cannot be drawn as a map because of a test.";
        app.view.as_mut().unwrap().notice = Some(why.into());
        let clicked = Instant::now();
        app.apply(Action::OpenPopup { iy: 99, ix: 99, lonlat: None });
        let v = app.view.as_mut().unwrap();
        assert_eq!(v.notice.as_deref(), Some(why));
        let (text, _) = v.flash.clone().expect("click error shown");
        assert_eq!(text, "That point is outside the grid.");
        // Still there within 4 s; gone after, and the notice stays.
        assert!(v.expire_flash(clicked + Duration::from_millis(3900)).is_some());
        assert!(v.flash.is_some());
        assert!(v.expire_flash(Instant::now() + NOTICE_TIME).is_none());
        assert!(v.flash.is_none());
        assert_eq!(v.notice.as_deref(), Some(why));
    }
}
