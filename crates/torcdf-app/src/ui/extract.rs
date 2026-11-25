//! The Extract window: choose variables and ranges, then save them as a
//! smaller netCDF file.

use eframe::egui;
use torcdf_core::NcFile;
use torcdf_core::extract::DimKey;

use crate::app::{Action, SelWin};
use crate::view::extract::ExtractForm;
use crate::view::selection::SelectionForm;

/// A variable offered for ticking.
pub struct Offered {
    pub path: String,
    /// The name (calculated variables without their "ƒ").
    pub name: String,
    /// "time, height, y, x".
    pub dims: String,
    pub units: String,
    pub long_name: String,
    pub derived: bool,
}

/// The data variables a selection window offers: not coordinate variables,
/// not scalars, and not variables other variables name in their
/// `coordinates`, `bounds` or `grid_mapping` (those are coordinates and
/// helpers, kept automatically when saving). Calculated variables last.
pub(crate) fn offered(nc: &NcFile, numeric_only: bool) -> Vec<Offered> {
    let all = nc.meta.all_vars();
    let mut helpers: std::collections::HashSet<String> = std::collections::HashSet::new();
    for v in &all {
        for a in v.attrs.iter().filter(|a| matches!(a.name.as_str(), "coordinates" | "bounds" | "grid_mapping")) {
            if let Some(text) = a.value.as_str() {
                helpers.extend(text.split_whitespace().map(|t| t.trim_end_matches(':').to_string()));
            }
        }
    }
    let attr = |attrs: &[torcdf_core::Attr], n: &str| {
        attrs.iter().find(|a| a.name == n).and_then(|a| a.value.as_str()).unwrap_or("").to_string()
    };
    let mut out: Vec<Offered> = all
        .into_iter()
        .filter(|v| !v.dims.is_empty() && !(v.dims.len() == 1 && v.dims[0] == v.name))
        .filter(|v| !helpers.contains(&v.name) && (!numeric_only || v.dtype.is_numeric()))
        .map(|v| Offered {
            path: v.path.clone(),
            name: v.name.clone(),
            dims: v.dims.join(", "),
            units: attr(&v.attrs, "units"),
            long_name: attr(&v.attrs, "long_name"),
            derived: false,
        })
        .collect();
    out.extend(nc.derived_vars().iter().map(|d| Offered {
        path: d.path.clone(),
        name: d.name.clone(),
        dims: d.info.dims.join(", "),
        units: d.units.clone().unwrap_or_default(),
        long_name: d.formula.clone(),
        derived: true,
    }));
    out
}

/// Draws the window. `nc`: the file, while it is open. `sliders`: the
/// dimensions the view has a slider for (rows of those get "◀ slider").
pub fn show(
    ctx: &egui::Context,
    form: &mut ExtractForm,
    nc: Option<&NcFile>,
    sliders: &[DimKey],
    actions: &mut Vec<Action>,
) {
    let mut open = true;
    let shown = egui::Window::new("Extract")
        .id(egui::Id::NULL.with("extract"))
        .open(&mut open)
        .default_width(680.0)
        .min_width(560.0)
        .resizable(true)
        .collapsible(false)
        .show(ctx, |ui| contents(ui, form, nc, sliders, actions));
    if shown.is_some_and(|r| r.response.contains_pointer()) && ctx.input(|i| i.pointer.any_pressed()) {
        actions.push(Action::TouchSelection(SelWin::Extract));
    }
    if !open {
        actions.push(Action::CloseExtract);
    }
}

fn contents(ui: &mut egui::Ui, form: &mut ExtractForm, nc: Option<&NcFile>, sliders: &[DimKey], actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        ui.label("File:");
        ui.strong(&form.file_name);
    });
    let running = form.job.is_some();
    if let Some(nc) = nc {
        let heading = "Variables (their coordinates are always kept)";
        ui.add_enabled_ui(!running, |ui| selection_fields(ui, &mut form.sel, offered(nc, false), sliders, SelWin::Extract, heading, actions));
    } else {
        ui.weak("The file was closed.");
    }

    ui.add_space(6.0);
    match &form.check {
        Ok(plan) => {
            let mb = plan.bytes as f64 / 1e6;
            ui.label(format!("About {} before compression", if mb < 1.0 { format!("{:.2} MB", mb) } else { format!("{mb:.0} MB") }));
            if !plan.skipped.is_empty() {
                ui.colored_label(ui.visuals().warn_fg_color, format!("Will not be saved: {}", plan.skipped.join(", ")));
            }
        }
        Err(e) => {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
    }
    ui.horizontal(|ui| {
        let can_save = nc.is_some() && form.check.is_ok() && !form.busy();
        if ui.add_enabled(can_save, egui::Button::new("Save netCDF…")).clicked() {
            actions.push(Action::SaveExtract);
        }
        if ui.button("Close").clicked() {
            actions.push(Action::CloseExtract);
        }
    });
    if let Some(job) = &form.job {
        ui.horizontal(|ui| {
            let p = job.progress();
            ui.add(egui::ProgressBar::new(p).desired_width(320.0).show_percentage());
            if ui.button("Cancel").clicked() {
                actions.push(Action::CancelExtract);
            }
        });
    }
    if let Some(s) = &form.status {
        let colour = if s.starts_with("Saved") { ui.visuals().text_color() } else { ui.visuals().error_fg_color };
        ui.add(egui::Label::new(egui::RichText::new(s).color(colour)).wrap());
    }
}

/// The selection rows shared by the Extract and Statistics windows: variable
/// ticks (`offered`: path, label), From/To rows, the region and its buttons.
pub fn selection_fields(
    ui: &mut egui::Ui,
    form: &mut SelectionForm,
    offered: Vec<Offered>,
    sliders: &[DimKey],
    win: SelWin,
    heading: &str,
    actions: &mut Vec<Action>,
) {
    ui.horizontal(|ui| {
        ui.label(heading);
        if ui.small_button("All").clicked() {
            form.ticked.extend(offered.iter().map(|o| o.path.clone()));
            actions.push(Action::SelectionEdited(win));
        }
        if ui.small_button("None").clicked() {
            form.ticked.clear();
            actions.push(Action::SelectionEdited(win));
        }
    });
    // A filter for files with many variables.
    let filter_id = egui::Id::NULL.with(("selection filter", win));
    let mut filter: String = ui.data(|d| d.get_temp(filter_id)).unwrap_or_default();
    if offered.len() > 8 {
        ui.add(egui::TextEdit::singleline(&mut filter).hint_text("filter variables").desired_width(220.0));
        ui.data_mut(|d| d.insert_temp(filter_id, filter.clone()));
    }
    let needle = filter.trim().to_lowercase();
    let shown = offered.iter().filter(|o| {
        needle.is_empty() || o.name.to_lowercase().contains(&needle) || o.long_name.to_lowercase().contains(&needle)
    });
    egui::ScrollArea::vertical().id_salt(("selection vars", win)).max_height(160.0).show(ui, |ui| {
        egui::Grid::new(("selection var grid", win)).num_columns(3).spacing([12.0, 2.0]).striped(true).show(ui, |ui| {
            for o in shown {
                let mut on = form.ticked.contains(&o.path);
                let label = if o.derived { format!("ƒ {}", o.name) } else { o.name.clone() };
                let hover = format!("{}\n{}", o.path, o.long_name);
                if ui.checkbox(&mut on, egui::RichText::new(label).strong()).on_hover_text(hover).changed() {
                    if on {
                        form.ticked.insert(o.path.clone());
                    } else {
                        form.ticked.remove(&o.path);
                    }
                    actions.push(Action::SelectionEdited(win));
                }
                ui.weak(format!("({})", o.dims));
                let units = if o.units.is_empty() { String::new() } else { format!("[{}]", o.units) };
                let text = [units, o.long_name.clone()].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("  ");
                ui.add(egui::Label::new(egui::RichText::new(text).weak()).truncate());
                ui.end_row();
            }
        });
    });
    ui.add_space(4.0);
    // Edits are checked (and snapped) when a box loses focus or Enter is pressed.
    let mut edited = false;
    let kept: Vec<Option<usize>> = form.dims.iter().map(|r| form.kept(&r.key)).collect();
    egui::Grid::new("extract rows").num_columns(5).spacing([6.0, 6.0]).show(ui, |ui| {
        for (i, row) in form.dims.iter_mut().enumerate() {
            let slider = sliders.contains(&row.key);
            ui.label(&row.label);
            for (end, text) in [(false, &mut row.from), (true, &mut row.to)] {
                ui.horizontal(|ui| {
                    ui.label(if end { "To" } else { "From" });
                    edited |= ui.add_sized([150.0, 20.0], egui::TextEdit::singleline(text)).lost_focus();
                    if slider && ui.small_button("◀ slider").on_hover_text("Use the slider's step").clicked() {
                        actions.push(Action::FromSlider { win, row: i, end });
                    }
                });
            }
            match kept.get(i).copied().flatten() {
                Some(n) => ui.weak(format!("{n} kept")),
                None => ui.weak(""),
            };
            ui.end_row();
        }
        if let Some(r) = &mut form.region {
            ui.label("Latitude");
            for (label, text) in [("From", &mut r.south), ("To", &mut r.north)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    edited |= ui.add_sized([100.0, 20.0], egui::TextEdit::singleline(text)).lost_focus();
                });
            }
            ui.end_row();
            ui.label("Longitude");
            for (label, text) in [("West", &mut r.west), ("East", &mut r.east)] {
                ui.horizontal(|ui| {
                    ui.label(label);
                    edited |= ui.add_sized([100.0, 20.0], egui::TextEdit::singleline(text)).lost_focus();
                });
            }
            ui.end_row();
        }
    });
    if form.region.is_some() {
        ui.horizontal(|ui| {
            if ui.button("Use current map view").clicked() {
                actions.push(Action::UseView(win));
            }
            if ui.button("Whole grid").clicked() {
                actions.push(Action::WholeGrid(win));
            }
        });
    }
    if edited {
        actions.push(Action::SelectionEdited(win));
    }
    ui.weak("Tip: Shift+drag on the map to draw the region; Shift+drag on a time series to set the time range.");
}

// ---- On-screen selection: Shift+drag and overlays --------------------------

/// The accent colour of Extract boxes and spans.
fn accent(ui: &egui::Ui) -> egui::Color32 {
    ui.visuals().selection.stroke.color
}

/// Shift+drag rubber band on a map or image. `start` holds the press
/// point while a band is drawn. Returns the band while dragging, and the
/// finished band (at least 3 points each way) on release.
pub fn band(
    ui: &egui::Ui,
    resp: &egui::Response,
    start: &mut Option<egui::Pos2>,
    rect: egui::Rect,
) -> (Option<egui::Rect>, Option<egui::Rect>) {
    if resp.drag_started() && ui.input(|i| i.modifiers.shift) {
        *start = ui.input(|i| i.pointer.press_origin());
    }
    let Some(a) = *start else { return (None, None) };
    let b = ui.input(|i| i.pointer.latest_pos()).unwrap_or(a);
    let r = egui::Rect::from_two_pos(a, b).intersect(rect);
    if resp.drag_stopped() || !ui.input(|i| i.pointer.primary_down()) {
        *start = None;
        return (None, (r.width() >= 3.0 && r.height() >= 3.0).then_some(r));
    }
    (Some(r), None)
}

/// Draws a band being dragged.
pub fn paint_band(ui: &egui::Ui, r: egui::Rect) {
    let c = accent(ui);
    let p = ui.painter();
    p.rect_filled(r, 0.0, c.gamma_multiply(0.15));
    p.rect_stroke(r, 0.0, egui::Stroke::new(1.5, c), egui::StrokeKind::Inside);
}

/// Draws the outline of a polygon given in screen points; `None` points
/// (off the projection) break it. With `wrap`, a jump wider than half the
/// panel (round the globe) breaks it too.
pub fn paint_outline(ui: &egui::Ui, clip: egui::Rect, pts: &[Option<egui::Pos2>], wrap: bool) {
    let painter = ui.painter().with_clip_rect(clip);
    let stroke = egui::Stroke::new(2.0, accent(ui));
    // A jump wider than half the panel is a wrap round the globe, not an edge.
    let far = clip.width().max(clip.height()) / 2.0;
    for w in pts.windows(2) {
        if let (Some(a), Some(b)) = (w[0], w[1])
            && (!wrap || (a - b).length() < far)
        {
            painter.line_segment([a, b], stroke);
        }
    }
}

/// The region box's outline in lon/lat: each edge sampled at 16 points,
/// east of west (a box across the seam runs past 180).
pub fn region_outline(s: f64, n: f64, w: f64, e: f64) -> Vec<(f64, f64)> {
    let e = if e < w { e + 360.0 } else { e };
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let mut out = Vec::new();
    for (a, b) in [((w, s), (e, s)), ((e, s), (e, n)), ((e, n), (w, n)), ((w, n), (w, s))] {
        for k in 0..16 {
            let t = k as f64 / 16.0;
            out.push((lerp(a.0, b.0, t), lerp(a.1, b.1, t)));
        }
    }
    out.push((w, s));
    out
}

/// Whether a Shift+drag is (or may start) on the plot with `id`: panning
/// is switched off then.
pub fn span_drag_active(ui: &egui::Ui, id: egui::Id) -> bool {
    ui.input(|i| i.modifiers.shift) || ui.data(|d| d.get_temp::<f64>(id).is_some())
}

/// Shift+drag across an egui_plot along x. Shades `shown` (the window's
/// time range) and the span being dragged; returns the finished (from, to).
pub fn plot_span(
    ui: &egui::Ui,
    id: egui::Id,
    out: &egui_plot::PlotResponse<()>,
    shown: Option<(f64, f64)>,
) -> Option<(f64, f64)> {
    let x_at = |p: egui::Pos2| out.transform.value_from_position(p).x;
    if out.response.drag_started()
        && ui.input(|i| i.modifiers.shift)
        && let Some(p) = ui.input(|i| i.pointer.press_origin())
    {
        ui.data_mut(|d| d.insert_temp(id, x_at(p)));
    }
    let frame = *out.transform.frame();
    let shade = |a: f64, b: f64, alpha: f32| {
        let (xa, xb) = (out.transform.position_from_point_x(a), out.transform.position_from_point_x(b));
        let r = egui::Rect::from_x_y_ranges(xa.min(xb)..=xa.max(xb).max(xa.min(xb) + 1.0), frame.y_range());
        ui.painter().with_clip_rect(frame).rect_filled(r, 0.0, accent(ui).gamma_multiply(alpha));
    };
    if let Some((a, b)) = shown {
        shade(a, b, 0.12);
    }
    let start = ui.data(|d| d.get_temp::<f64>(id))?;
    let now = ui.input(|i| i.pointer.latest_pos()).map(x_at).unwrap_or(start);
    if out.response.drag_stopped() || !ui.input(|i| i.pointer.primary_down()) {
        ui.data_mut(|d| d.remove::<f64>(id));
        return Some((start, now));
    }
    shade(start, now, 0.25);
    ui.ctx().request_repaint();
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[Offered]) -> Vec<String> {
        v.iter().map(|o| o.name.clone()).collect()
    }

    #[test]
    fn only_data_variables_are_offered() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&torcdf_core::fixtures::vectors::write_arome_like(dir.path())).unwrap();
        let v = offered(&nc, true);
        assert_eq!(names(&v), ["wind_speed", "wind_direction"], "coordinates and time are not listed");
        assert_eq!(v[0].dims, "time, height, y, x");
        assert_eq!(v[0].units, "m/s");

        let nc = NcFile::open(&torcdf_core::fixtures::write_nc4(dir.path())).unwrap();
        nc.add_derived(nc.check_formula("s", "uo*2", None, None).unwrap());
        assert_eq!(names(&offered(&nc, true)), ["uo", "vo", "s"]);
        assert_eq!(names(&offered(&nc, false)), ["uo", "vo", "label", "s"]);
        assert!(offered(&nc, true)[2].derived);
    }
}
