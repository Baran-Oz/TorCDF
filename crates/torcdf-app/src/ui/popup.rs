//! Pop-up windows of a clicked point's time series or vertical profile, and
//! the ring markers that show where each point is on the map or image.

use eframe::egui::{self, Color32, Pos2, Stroke};
use egui_plot::{HLine, Line, Plot, PlotPoints, VLine};

use super::{fmt_value, slider_label};
use crate::app::Action;
use crate::view::popup::{Popup, PopupKind, PopupSet};

/// Radius of a marker ring (points).
const MARKER_RADIUS: f32 = 4.0;
const EAST: Color32 = Color32::from_rgb(214, 90, 60);
const NORTH: Color32 = Color32::from_rgb(60, 140, 214);
const MAIN: Color32 = Color32::from_rgb(70, 70, 70);
const NOW: Color32 = Color32::from_gray(150);

/// Index of the value in `axis` closest to `v` (the first of equals).
pub fn nearest_index(axis: &[f64], v: f64) -> Option<usize> {
    if !v.is_finite() {
        return None;
    }
    axis.iter()
        .enumerate()
        .filter(|(_, a)| a.is_finite())
        .min_by(|(_, a), (_, b)| (*a - v).abs().total_cmp(&(*b - v).abs()))
        .map(|(i, _)| i)
}

/// Where the current view stands: file id, dimension names, slider positions.
pub type Current<'a> = (u64, &'a [String], &'a [usize]);

/// Draws every pop-up as a window; what the user did is added to `actions`.
pub fn show_all(
    ctx: &egui::Context,
    set: &mut PopupSet,
    current: Option<Current<'_>>,
    actions: &mut Vec<Action>,
) {
    for p in &mut set.list {
        let mut open = true;
        let shown = egui::Window::new(p.title.clone())
            .id(egui::Id::NULL.with(("popup", p.id)))
            .default_size([440.0, 330.0])
            .open(&mut open)
            .show(ctx, |ui| contents(ui, p, current, actions));
        p.rect = shown.map(|r| r.response.rect);
        if !open {
            actions.push(Action::ClosePopup(p.id));
        }
    }
}

fn contents(ui: &mut egui::Ui, p: &mut Popup, current: Option<Current<'_>>, actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        if p.kinds.len() > 1 {
            for (kind, name) in [(PopupKind::Series, "Series"), (PopupKind::Profile, "Profile")] {
                if p.kinds.contains(&kind) && ui.radio(p.kind == kind, name).clicked() && p.kind != kind {
                    actions.push(Action::SetPopupKind(p.id, kind));
                }
            }
            ui.separator();
        }
        ui.toggle_value(&mut p.pinned, "Pin").on_hover_text("Keep this pop-up when another point is clicked");
        if ui.button("Save CSV…").clicked() {
            actions.push(Action::SavePopupCsv(p.id));
        }
        if ui.button("PNG").clicked() {
            actions.push(Action::SavePopupPng(p.id));
        }
        if p.has_components() {
            ui.separator();
            ui.checkbox(&mut p.show_east, "East");
            ui.checkbox(&mut p.show_north, "North");
        }
    });
    let (done, total) = p.progress();
    if done < total {
        ui.add(egui::ProgressBar::new(done as f32 / total as f32).desired_height(6.0));
    }
    if let Some(e) = &p.error {
        ui.colored_label(ui.visuals().error_fg_color, e);
    }
    plot(ui, p, current, actions);
}

/// The axis position of the current slider, when this pop-up's file and
/// dimension are the ones on show.
fn current_axis_value(p: &Popup, current: Option<Current<'_>>) -> Option<f64> {
    let (file, dims, pos) = current?;
    if file != p.file_id {
        return None;
    }
    let i = dims.iter().position(|d| d == p.along_name())?;
    p.axis.get(*pos.get(i)?).copied()
}

/// Plot points of the finite values; profiles have the value on x and the
/// level on y, negated when it runs `down`.
fn points(axis: &[f64], values: &[Option<f32>], profile: bool, down: bool) -> Vec<[f64; 2]> {
    axis.iter()
        .zip(values)
        .filter_map(|(&a, v)| {
            let v = (*v)?;
            if !v.is_finite() || !a.is_finite() {
                return None;
            }
            let a = if down { -a } else { a };
            Some(if profile { [v as f64, a] } else { [a, v as f64] })
        })
        .collect()
}

fn plot(ui: &mut egui::Ui, p: &Popup, current: Option<Current<'_>>, actions: &mut Vec<Action>) {
    let profile = p.kind == PopupKind::Profile;
    let down = profile && p.down;
    let units = p.units.as_ref().map(|u| format!(" [{u}]")).unwrap_or_default();
    let value_label = format!("{}{units}", p.var_name);
    let along = p.along_name().to_string();
    let time = p.time.clone();
    // Text of a position on the along axis (decoded when it is time).
    let axis_text = move |v: f64| match &time {
        Some(t) => slider_label(0, Some(v), Some(t)),
        None => fmt_value(v),
    };
    let signed = move |v: f64| if down { -v } else { v };
    let (text_x, text_y, hover) = (axis_text.clone(), axis_text.clone(), axis_text);
    let mut plot = Plot::new(("popup-plot", p.id, profile)).height(240.0);
    plot = if profile {
        plot.x_axis_label(value_label)
            .y_axis_label(along.clone())
            .y_axis_formatter(move |m, _| text_y(signed(m.value)))
    } else {
        plot.x_axis_label(along.clone()).y_axis_label(value_label).x_axis_formatter(move |m, _| text_x(m.value))
    };
    plot = plot.label_formatter(move |h| {
        let (egui_plot::HoverPosition::NearDataPoint { position: v, .. }
        | egui_plot::HoverPosition::Elsewhere { position: v }) = h;
        let (a, val) = if profile { (signed(v.y), v.x) } else { (v.x, v.y) };
        Some(format!("{along} = {}\nvalue = {}", hover(a), fmt_value(val)))
    });
    let mark = current_axis_value(p, current);
    let out = plot.show(ui, |plot_ui| {
        let mut line = |name: &str, values: &[Option<f32>], colour: Color32| {
            let pts = points(&p.axis, values, profile, down);
            plot_ui.line(Line::new(name.to_string(), PlotPoints::from(pts)).color(colour));
        };
        line(&p.var_name, &p.values, MAIN);
        if p.show_east {
            line("East", &p.east, EAST);
        }
        if p.show_north {
            line("North", &p.north, NORTH);
        }
        if let Some(a) = mark {
            if profile {
                plot_ui.hline(HLine::new("now", signed(a)).color(NOW));
            } else {
                plot_ui.vline(VLine::new("now", a).color(NOW));
            }
        }
    });
    if out.response.clicked()
        && let Some(pos) = out.response.interact_pointer_pos()
    {
        let v = out.transform.value_from_position(pos);
        let at = if profile { signed(v.y) } else { v.x };
        if let Some(index) = nearest_index(&p.axis, at) {
            actions.push(Action::JumpTo { file: p.file_id, dim_name: p.along_name().to_string(), index });
        }
    }
}

/// A ring on every pop-up's point of the variable on show. `to_screen` gives
/// the point's place in the current view, or `None` when it is not there.
pub fn draw_markers(
    painter: &egui::Painter,
    set: &PopupSet,
    file_id: u64,
    var_path: &str,
    to_screen: &dyn Fn(&Popup) -> Option<Pos2>,
) {
    for p in set.list.iter().filter(|p| p.file_id == file_id && p.var_path == var_path) {
        if let Some(pos) = to_screen(p) {
            painter.circle_stroke(pos, MARKER_RADIUS, Stroke::new(3.0, Color32::BLACK));
            painter.circle_stroke(pos, MARKER_RADIUS, Stroke::new(1.5, Color32::WHITE));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_index_on_axis() {
        assert_eq!(nearest_index(&[0.0, 10.0, 20.0], 13.0), Some(1));
        assert_eq!(nearest_index(&[], 1.0), None);
        assert_eq!(nearest_index(&[0.0, 10.0], f64::NAN), None);
        assert_eq!(nearest_index(&[20.0, 10.0, 0.0], 1.0), Some(2));
    }

    #[test]
    fn profile_points_flip_for_down() {
        let axis = [1.0, 2.0, 3.0];
        let v = [Some(5.0), Some(f32::NAN), Some(7.0)];
        assert_eq!(points(&axis, &v, true, true), vec![[5.0, -1.0], [7.0, -3.0]]);
        assert_eq!(points(&axis, &v, false, false), vec![[1.0, 5.0], [3.0, 7.0]]);
    }
}
