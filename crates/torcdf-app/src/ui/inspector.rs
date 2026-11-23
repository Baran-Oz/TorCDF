//! Right panel: plot settings, value under the cursor, attributes.

use eframe::egui;
use torcdf_core::cf::Geo;
use torcdf_core::{Attr, RotateChoice, Storage};

use super::{attr_text, dtype_name, fmt_value, slider_label};
use crate::map::projection::Projection;
use crate::app::Action;
use crate::render::colormap::Colormap;
use crate::view::defaults::ViewKind;
use crate::view::var_view::{Display, VarView};

pub fn show(
    ui: &mut egui::Ui,
    display: &mut Display,
    mut view: Option<&mut VarView>,
    var_selected: bool,
    attrs: Option<(String, &[Attr])>,
    actions: &mut Vec<Action>,
) {
    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
        ui.heading("Plot");
        settings(ui, display, view.as_deref());
        if let Some(v) = view.as_deref_mut() {
            map_settings(ui, v);
        }
        if let Some(v) = view.as_deref() {
            vector_section(ui, v, actions);
        }
        let view = view.as_deref();
        ui.separator();
        ui.heading("Cursor");
        if view.is_some_and(|v| v.kind == ViewKind::Map) {
            map_cursor(ui, view);
        } else {
            cursor(ui, view);
        }
        ui.separator();
        ui.heading("Attributes");
        attributes(ui, view, var_selected, attrs);
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

/// Components of a vector view, what is known about their direction, and
/// the rotation switch.
fn vector_section(ui: &mut egui::Ui, v: &VarView, actions: &mut Vec<Action>) {
    let Some(pair) = &v.vector else { return };
    ui.separator();
    ui.heading("Vector");
    let leaf = |p: &str| p.rsplit('/').next().unwrap_or(p).to_string();
    ui.label(format!("{} + {}", leaf(&pair.east), leaf(&pair.north)));
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
    ui.add_space(4.0);
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
