//! Right panel: a Plot | Attributes tab row, the tab's content, and the
//! value under the cursor pinned at the bottom.

use eframe::egui;
use torcdf_core::cf::Geo;
use torcdf_core::{
    AngleRef, AngleUnit, Attr, DirConvention, Guessed, PairForm, RotateChoice, Storage, Towards, VectorPair,
    en_to_magdir,
};

use super::{attr_text, dtype_name, fmt_value, slider_label};
use crate::map::projection::Projection;
use crate::app::{Action, hidden_note};
use crate::render::colormap::Colormap;
use crate::view::arrow_layer::{ArrowLayer, ArrowMode, PICK_TWO};
use crate::view::arrow_state::LENGTH_RANGE;
use crate::view::defaults::ViewKind;
use crate::view::var_view::{Display, VarView};

/// Height of the Cursor area pinned at the bottom of the panel (its rows
/// scroll when they do not fit).
const CURSOR_HEIGHT: f32 = 220.0;

/// The right panel's tabs (one choice per window).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PanelTab {
    #[default]
    Plot,
    Attributes,
}

/// What the panel shows besides the view and the colour settings.
pub struct PanelInput<'a> {
    pub tab: PanelTab,
    pub var_selected: bool,
    /// Title and attributes of the selected item.
    pub attrs: Option<(String, &'a [Attr])>,
    /// The arrow slot of the view's file and the variables its drop-downs
    /// offer in its mode.
    pub arrows: Option<(&'a ArrowLayer, &'a [String])>,
}

pub fn show(
    ui: &mut egui::Ui,
    input: PanelInput<'_>,
    display: &mut Display,
    mut view: Option<&mut VarView>,
    actions: &mut Vec<Action>,
) {
    let edge = egui::Frame::NONE.inner_margin(egui::Margin::symmetric(0, 4));
    egui::Panel::top("panel tabs").frame(edge).show(ui, |ui| {
        ui.horizontal(|ui| {
            for (tab, name) in [(PanelTab::Plot, "Plot"), (PanelTab::Attributes, "Attributes")] {
                if ui.selectable_label(input.tab == tab, name).clicked() && input.tab != tab {
                    actions.push(Action::SetPanelTab(tab));
                }
            }
        });
    });
    egui::Panel::bottom("cursor area").resizable(false).exact_size(CURSOR_HEIGHT).frame(edge).show(ui, |ui| {
        ui.heading("Cursor");
        egui::ScrollArea::vertical().id_salt("cursor rows").auto_shrink(false).show(ui, |ui| {
            let view = view.as_deref();
            if view.is_some_and(|v| v.kind == ViewKind::Map) {
                map_cursor(ui, view);
            } else {
                cursor(ui, view);
            }
        });
    });
    egui::ScrollArea::vertical().id_salt("panel tab").auto_shrink(false).show(ui, |ui| match input.tab {
        PanelTab::Plot => {
            ui.heading("Colour");
            if let Some(v) = view.as_deref() {
                ui.strong(&v.var.name);
            }
            settings(ui, display, view.as_deref());
            if let Some(v) = view.as_deref_mut() {
                map_settings(ui, v);
            }
            if let (Some(v), Some((layer, candidates))) = (view.as_deref_mut(), input.arrows) {
                arrows_section(ui, v, layer, candidates, actions);
            }
        }
        PanelTab::Attributes => attributes(ui, view.as_deref(), input.var_selected, input.attrs),
    });
}

fn settings(ui: &mut egui::Ui, display: &mut Display, view: Option<&VarView>) {
    egui::ComboBox::from_id_salt("colormap")
        .selected_text(display.colormap.name())
        .show_ui(ui, |ui| {
            for cm in Colormap::all() {
                ui.selectable_value(&mut display.colormap, *cm, cm.name());
            }
        });
    ui.checkbox(&mut display.log, "Log scale");

    let stats = view.and_then(|v| v.shown.as_ref()).map(|s| &s.stats);
    let effective = display.effective(stats);
    if let Some((lo, hi)) = effective {
        let (mut new_lo, mut new_hi) = (lo, hi);
        let speed = ((hi - lo).abs() as f64 / 300.0).max(1e-9);
        ui.horizontal(|ui| {
            ui.label("Min");
            ui.add(range_field(&mut new_lo, speed));
            ui.label("Max");
            ui.add(range_field(&mut new_hi, speed));
        });
        // Editing a bound keeps it: the range becomes locked.
        if (new_lo, new_hi) != (lo, hi) {
            display.range = (new_lo, new_hi);
            display.lock = true;
        }
    } else {
        ui.weak("Range: waiting for data");
    }
    let mut lock = display.lock;
    if ui.checkbox(&mut lock, "Lock range").changed() {
        if let (true, Some(r)) = (lock, effective) {
            display.range = r;
        }
        display.lock = lock;
    }

    if stats.is_some_and(|s| s.auto.is_none()) {
        ui.colored_label(ui.visuals().warn_fg_color, "All values missing");
    }
    if let (true, Some(r)) = (display.log, effective) {
        match log_range(r, stats.and_then(|s| s.min_positive)) {
            Err(msg) => {
                ui.colored_label(ui.visuals().warn_fg_color, msg);
            }
            Ok((lo, _)) if lo != r.0 => {
                ui.weak(format!("Log scale starts at {} (smallest positive value)", fmt_value(lo as f64)));
            }
            Ok(_) => {}
        }
    }
}

/// The Arrows section: how the arrows are made (or Off), the two variables,
/// a magnitude/direction pair's convention, why nothing is drawn, then the
/// arrow length, the grid notes and the rotation switch of the pair drawn.
fn arrows_section(
    ui: &mut egui::Ui,
    v: &mut VarView,
    layer: &ArrowLayer,
    candidates: &[String],
    actions: &mut Vec<Action>,
) {
    if v.axes.is_none() && v.vector.is_none() {
        return;
    }
    ui.separator();
    ui.heading("Arrows");
    let mut mode = layer.mode;
    ui.radio_value(&mut mode, ArrowMode::Off, "Off");
    ui.radio_value(&mut mode, ArrowMode::EastNorth, "x & y components");
    ui.radio_value(&mut mode, ArrowMode::MagDir, "Magnitude & direction");
    if mode != layer.mode {
        actions.push(Action::SetArrowMode(mode));
    }
    let labels = match layer.mode {
        ArrowMode::Off => None,
        ArrowMode::EastNorth => Some(["x (east)", "y (north)"]),
        ArrowMode::MagDir => Some(["Magnitude", "Direction"]),
    };
    if let Some(labels) = labels {
        egui::Grid::new("arrow picks").num_columns(2).show(ui, |ui| {
            for (slot, label) in labels.into_iter().enumerate() {
                ui.label(label);
                let picked = layer.picks[slot].as_deref();
                egui::ComboBox::from_id_salt(("arrow pick", slot))
                    .selected_text(picked.map_or("—", leaf))
                    .show_ui(ui, |ui| {
                        for c in candidates {
                            if ui.selectable_label(picked == Some(c.as_str()), leaf(c)).clicked() {
                                actions.push(Action::PickArrowVar { slot, path: c.clone() });
                            }
                        }
                    });
                ui.end_row();
            }
        });
        if let (ArrowMode::MagDir, Some(PairForm::MagDir(c))) = (layer.mode, layer.pair.as_ref().map(|p| p.form)) {
            convention(ui, c, actions);
        }
        let problem = layer.problem.as_deref().or(layer.missing_picks().then_some(PICK_TWO));
        if let Some(p) = problem {
            ui.add(egui::Label::new(egui::RichText::new(p).color(ui.visuals().warn_fg_color)).wrap());
        } else if layer.pair.is_some() && v.vector.is_none() {
            // On, but this variable is not on the pair's grid.
            ui.add(egui::Label::new(hidden_note(&v.var.name)).wrap());
        }
    }
    if v.vector.is_none() {
        return;
    }
    if let Some(a) = &mut v.arrows {
        ui.add_enabled(
            a.on,
            egui::Slider::new(&mut a.length, LENGTH_RANGE).logarithmic(true).text("Arrow length"),
        );
    }
    let Some(r) = &v.resolved else {
        ui.weak("Working out the grid...");
        return;
    };
    for note in &r.notes {
        ui.add(egui::Label::new(note).wrap());
    }
    let mut rotated = r.rotated;
    if ui.checkbox(&mut rotated, "Rotate to east/north").changed() {
        actions.push(Action::SetRotate(if rotated { RotateChoice::On } else { RotateChoice::Off }));
    }
}

/// A variable path's last component, as the drop-downs list it.
fn leaf(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

/// Direction convention of a magnitude/direction pair: going from / to,
/// the angle reference and unit, each marked as from the file or guessed.
/// A change sends the new convention with that setting's guess cleared.
fn convention(ui: &mut egui::Ui, c: DirConvention, actions: &mut Vec<Action>) {
    let note = |ui: &mut egui::Ui, guessed: bool| ui.weak(if guessed { "guessed" } else { "from the file" });
    let mut new = c;
    ui.horizontal(|ui| {
        ui.radio_value(&mut new.towards, Towards::From, "Coming from");
        ui.radio_value(&mut new.towards, Towards::To, "Going to");
        note(ui, c.guessed.towards);
    });
    let reference_name = |r: AngleRef| match r {
        AngleRef::ClockwiseFromNorth => "Clockwise from north",
        AngleRef::CounterClockwiseFromEast => "Counter-clockwise from east",
    };
    let unit_name = |u: AngleUnit| match u {
        AngleUnit::Degrees => "Degrees",
        AngleUnit::Radians => "Radians",
    };
    egui::Grid::new("arrow convention").num_columns(3).show(ui, |ui| {
        ui.label("Angle");
        egui::ComboBox::from_id_salt("arrow angle").selected_text(reference_name(c.reference)).show_ui(ui, |ui| {
            for r in [AngleRef::ClockwiseFromNorth, AngleRef::CounterClockwiseFromEast] {
                ui.selectable_value(&mut new.reference, r, reference_name(r));
            }
        });
        note(ui, c.guessed.reference);
        ui.end_row();
        ui.label("Units");
        egui::ComboBox::from_id_salt("arrow units").selected_text(unit_name(c.unit)).show_ui(ui, |ui| {
            for u in [AngleUnit::Degrees, AngleUnit::Radians] {
                ui.selectable_value(&mut new.unit, u, unit_name(u));
            }
        });
        note(ui, c.guessed.unit);
        ui.end_row();
    });
    if new != c {
        let g = c.guessed;
        new.guessed = Guessed {
            towards: g.towards && new.towards == c.towards,
            reference: g.reference && new.reference == c.reference,
            unit: g.unit && new.unit == c.unit,
        };
        actions.push(Action::SetConvention(new));
    }
}

/// The cursor grid's "arrow" row when the view draws arrows: the arrow at
/// the pointer in the pair's form, "—" where there is none.
fn arrow_row(ui: &mut egui::Ui, v: &VarView) {
    let Some(pair) = v.vector.as_ref().filter(|_| v.arrows.as_ref().is_some_and(|a| a.on)) else { return };
    let units = pair.units.as_deref().unwrap_or("");
    let line = match v.hover.and_then(|h| h.arrow) {
        Some((east, north)) => arrow_line(pair, east, north, units),
        None => "—".into(),
    };
    ui.label("arrow");
    ui.label(line);
    ui.end_row();
}

/// The cursor's arrow line for the east/north drawn at a cell, in `pair`'s
/// form: "east 1.2, north -0.4 m/s", or "11.3 m/s from 247°" ("towards"
/// for a going-to convention; radians as "rad"). A missing part gives "—".
pub fn arrow_line(pair: &VectorPair, east: f32, north: f32, units: &str) -> String {
    if east.is_nan() || north.is_nan() {
        return "—".into();
    }
    let with_units = |s: String| if units.is_empty() { s } else { format!("{s} {units}") };
    match pair.form {
        PairForm::EastNorth => {
            with_units(format!("east {}, north {}", fmt_value(east as f64), fmt_value(north as f64)))
        }
        PairForm::MagDir(c) => {
            let (mag, dir) = en_to_magdir(east as f64, north as f64, &c);
            let way = match c.towards {
                Towards::From => "from",
                Towards::To => "towards",
            };
            // `+ 0.0` turns a -0 angle into 0.
            let dir = dir + 0.0;
            let angle = match c.unit {
                // Whole degrees; just under a full turn reads 0°, not 360°.
                AngleUnit::Degrees => format!("{}°", dir.round().rem_euclid(360.0) + 0.0),
                AngleUnit::Radians => format!("{dir:.2} rad"),
            };
            format!("{} {way} {angle}", with_units(fmt_value(mag)))
        }
    }
}

fn range_field(v: &mut f32, speed: f64) -> egui::DragValue<'_> {
    egui::DragValue::new(v).speed(speed).custom_formatter(|x, _| fmt_value(x))
}

fn units(v: &VarView) -> &str {
    v.var.attrs.iter().find(|a| a.name == "units").and_then(|a| a.value.as_str()).unwrap_or("")
}

/// Projection and overlay switches of the map view. They only change
/// state; re-projection happens in background jobs.
fn map_settings(ui: &mut egui::Ui, v: &mut VarView) {
    let (ViewKind::Map, Some(map)) = (v.kind, v.map.as_mut()) else { return };
    ui.separator();
    ui.heading("Map");
    let mut proj = map.camera.proj;
    egui::ComboBox::from_id_salt("projection").selected_text(proj.name()).show_ui(ui, |ui| {
        for p in Projection::all() {
            ui.selectable_value(&mut proj, *p, p.name());
        }
    });
    map.set_projection(proj);
    ui.checkbox(&mut map.coast, "Coastlines");
    ui.checkbox(&mut map.borders, "Borders");
    ui.checkbox(&mut map.grid, "Grid lines");
}

/// Value, lon/lat and slider positions under the pointer on the map; "—"
/// where there is no data (or no globe).
fn map_cursor(ui: &mut egui::Ui, view: Option<&VarView>) {
    let Some((v, map)) = view.and_then(|v| Some((v, v.map.as_ref()?))) else { return };
    if !map.hovering {
        ui.weak("Hover over the map to see values.");
        return;
    }
    const NONE: &str = "—";
    egui::Grid::new("map cursor").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("value");
        match v.hover {
            Some(h) => ui.label(format!("{} {}", fmt_value(h.value as f64), units(v))),
            None => ui.label(NONE),
        };
        ui.end_row();
        arrow_row(ui, v);
        let (lon, lat) = match map.pointer {
            Some((lo, la)) => (format!("{lo:.4}°"), format!("{la:.4}°")),
            None => (NONE.into(), NONE.into()),
        };
        ui.label("lon");
        ui.label(lon);
        ui.end_row();
        ui.label("lat");
        ui.label(lat);
        ui.end_row();
        if let (Some(h), Some(a)) = (v.hover, v.axes) {
            ui.label("index");
            ui.label(format!("{} {}, {} {}", v.var.dims[a.y], h.iy, v.var.dims[a.x], h.ix));
            ui.end_row();
        }
        let coord = |d: usize, i: usize| v.coords.dims.get(d)?.as_ref()?.get(i).copied();
        for d in v.slider_dims() {
            let pos = v.slider_pos[d];
            let time = if v.time_dim == Some(d) { v.time_units.as_ref() } else { None };
            ui.label(&v.var.dims[d]);
            ui.label(slider_label(pos, coord(d, pos), time));
            ui.end_row();
        }
    });
}

fn cursor(ui: &mut egui::Ui, view: Option<&VarView>) {
    let (Some(v), Some(h), Some(a)) =
        (view, view.and_then(|v| v.hover), view.and_then(|v| v.axes))
    else {
        ui.weak("Hover over an image to see values.");
        return;
    };
    egui::Grid::new("cursor").num_columns(2).striped(true).show(ui, |ui| {
        ui.label("value");
        ui.label(format!("{} {}", fmt_value(h.value as f64), units(v)));
        ui.end_row();
        arrow_row(ui, v);
        let coord = |d: usize, i: usize| v.coords.dims.get(d)?.as_ref()?.get(i).copied();
        match &v.geo {
            Geo::Regular { .. } => {
                if let (Some(lat), Some(lon)) = (coord(a.y, h.iy), coord(a.x, h.ix)) {
                    ui.label("lat");
                    ui.label(format!("{lat:.4}"));
                    ui.end_row();
                    ui.label("lon");
                    ui.label(format!("{lon:.4}"));
                    ui.end_row();
                }
            }
            Geo::Curvilinear { .. } => {
                let i = h.iy * v.var.shape[a.x] + h.ix;
                if let Some((lat, lon)) = &v.coords.curv
                    && let (Some(la), Some(lo)) = (lat.get(i), lon.get(i))
                {
                    ui.label("lat");
                    ui.label(format!("{la:.4}"));
                    ui.end_row();
                    ui.label("lon");
                    ui.label(format!("{lo:.4}"));
                    ui.end_row();
                }
            }
            Geo::None => {
                for (d, i) in [(a.y, h.iy), (a.x, h.ix)] {
                    if let Some(c) = coord(d, i) {
                        ui.label(&v.var.dims[d]);
                        ui.label(fmt_value(c));
                        ui.end_row();
                    }
                }
            }
        }
        ui.label("index");
        ui.label(format!("{} {}, {} {}", v.var.dims[a.y], h.iy, v.var.dims[a.x], h.ix));
        ui.end_row();
        for d in v.slider_dims() {
            let pos = v.slider_pos[d];
            let time = if v.time_dim == Some(d) { v.time_units.as_ref() } else { None };
            ui.label(&v.var.dims[d]);
            ui.label(slider_label(pos, coord(d, pos), time));
            ui.end_row();
        }
    });
}

fn attributes(
    ui: &mut egui::Ui,
    view: Option<&VarView>,
    var_selected: bool,
    attrs: Option<(String, &[Attr])>,
) {
    let Some((title, attrs)) = attrs else {
        ui.weak("Select a file, group or variable.");
        return;
    };
    ui.strong(title);
    if let Some(v) = view.filter(|_| var_selected) {
        let dims: Vec<String> =
            v.var.dims.iter().zip(&v.var.shape).map(|(d, n)| format!("{d}={n}")).collect();
        ui.label(format!("{}  ({})", dtype_name(&v.var.dtype), dims.join(", ")));
        let storage = match &v.var.storage {
            Storage::Contiguous => "contiguous".to_string(),
            Storage::Compact => "compact".to_string(),
            Storage::Chunked { chunk, compressed } => {
                let c: Vec<String> = chunk.iter().map(usize::to_string).collect();
                let z = if *compressed { ", compressed" } else { "" };
                format!("chunked [{}]{z}", c.join(" × "))
            }
            Storage::Unknown => "unknown".to_string(),
        };
        ui.weak(format!("storage: {storage}"));
    }
    if attrs.is_empty() {
        ui.weak("(no attributes)");
        return;
    }
    egui::Grid::new("attrs").num_columns(2).striped(true).show(ui, |ui| {
        for a in attrs {
            ui.label(egui::RichText::new(&a.name).strong());
            ui.add(egui::Label::new(attr_text(&a.value)).wrap());
            ui.end_row();
        }
    });
}

/// Range centred on zero when it straddles zero (diverging and topography
/// maps put their midpoint at zero); unchanged otherwise.
pub fn centre_on_zero(range: (f32, f32)) -> (f32, f32) {
    let (lo, hi) = range;
    if lo < 0.0 && hi > 0.0 {
        let m = hi.max(-lo);
        (-m, m)
    } else {
        range
    }
}

/// Topography range: always symmetric about zero so the colour map splits at
/// sea level, even when the data are all above or all below it.
pub fn sea_level_range(range: (f32, f32)) -> (f32, f32) {
    let m = range.0.abs().max(range.1.abs());
    (-m, m)
}

/// The range to draw in log mode: a non-positive low end is raised to the
/// smallest positive value of the slice. `Err` explains why log scale cannot
/// be drawn.
pub fn log_range(range: (f32, f32), min_positive: Option<f32>) -> Result<(f32, f32), &'static str> {
    let (lo, hi) = range;
    // Deliberate: a NaN maximum must also take the error path.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(hi > 0.0) {
        return Err("log scale needs a positive range maximum");
    }
    let lo = if lo > 0.0 { lo } else { min_positive.ok_or("log scale: no positive values")? };
    if lo >= hi {
        return Err("log scale: no positive values inside the range");
    }
    Ok((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torcdf_core::{AngleSource, Frame};

    fn pair(form: PairForm) -> VectorPair {
        VectorPair {
            path: "/p".into(),
            name: "p".into(),
            units: None,
            east: "/a".into(),
            north: "/b".into(),
            centre_dims: vec!["y".into(), "x".into()],
            centre_shape: vec![2, 2],
            horiz: [0, 1],
            centre_lat: None,
            centre_lon: None,
            frame: Frame::Geographic,
            angle: AngleSource::FromLatLon,
            rotate: RotateChoice::Auto,
            by_hand: true,
            form,
        }
    }

    fn magdir(towards: Towards, unit: AngleUnit) -> PairForm {
        PairForm::MagDir(DirConvention {
            towards,
            reference: AngleRef::ClockwiseFromNorth,
            unit,
            guessed: Guessed::default(),
        })
    }

    #[test]
    fn arrow_line_formats() {
        let from = pair(magdir(Towards::From, AngleUnit::Degrees));
        assert_eq!(arrow_line(&from, 10.0, 0.0, "m/s"), "10 m/s from 270°");
        let to = pair(magdir(Towards::To, AngleUnit::Degrees));
        assert_eq!(arrow_line(&to, 0.0, 10.0, "m/s"), "10 m/s towards 0°");
        // Just under a whole turn rounds to 0°, never 360°.
        assert_eq!(arrow_line(&to, -0.001, 10.0, "m/s"), "10 m/s towards 0°");
        // A wind from the north: never "-0°".
        assert_eq!(arrow_line(&from, 0.0, -10.0, "m/s"), "10 m/s from 0°");
        assert_eq!(arrow_line(&to, -0.0, 10.0, "m/s"), "10 m/s towards 0°");
        let rad = pair(magdir(Towards::To, AngleUnit::Radians));
        assert_eq!(arrow_line(&rad, 10.0, 0.0, "m/s"), "10 m/s towards 1.57 rad");
        let en = pair(PairForm::EastNorth);
        assert_eq!(arrow_line(&en, 1.25, -0.5, "m/s"), "east 1.25, north -0.5 m/s");
        assert_eq!(arrow_line(&en, 1.25, -0.5, ""), "east 1.25, north -0.5");
        assert_eq!(arrow_line(&from, 11.3, 0.0, ""), "11.3 from 270°");
        for p in [&from, &en] {
            assert_eq!(arrow_line(p, f32::NAN, 1.0, "m/s"), "—");
            assert_eq!(arrow_line(p, 1.0, f32::NAN, "m/s"), "—");
        }
    }

    #[test]
    fn centring() {
        assert_eq!(centre_on_zero((-5000.0, 3000.0)), (-5000.0, 5000.0));
        assert_eq!(centre_on_zero((-1.0, 4.0)), (-4.0, 4.0));
        assert_eq!(centre_on_zero((2.0, 4.0)), (2.0, 4.0));
        assert_eq!(centre_on_zero((-4.0, -2.0)), (-4.0, -2.0));
        assert_eq!(centre_on_zero((0.0, 4.0)), (0.0, 4.0));
    }

    #[test]
    fn topo_always_splits_at_sea_level() {
        assert_eq!(sea_level_range((-5000.0, 3000.0)), (-5000.0, 5000.0));
        assert_eq!(sea_level_range((2.0, 4.0)), (-4.0, 4.0));
        assert_eq!(sea_level_range((-4.0, -2.0)), (-4.0, 4.0));
        assert_eq!(sea_level_range((0.0, 4.0)), (-4.0, 4.0));
    }

    #[test]
    fn log_ranges() {
        assert_eq!(log_range((1.0, 100.0), Some(0.5)), Ok((1.0, 100.0)));
        assert_eq!(log_range((-3.0, 100.0), Some(0.5)), Ok((0.5, 100.0)));
        assert_eq!(log_range((0.0, 100.0), Some(2.0)), Ok((2.0, 100.0)));
        assert!(log_range((-3.0, 100.0), None).is_err());
        assert!(log_range((-3.0, -1.0), Some(0.5)).is_err());
        // Smallest positive value above the range top.
        assert!(log_range((-3.0, 1.0), Some(5.0)).is_err());
    }
}
