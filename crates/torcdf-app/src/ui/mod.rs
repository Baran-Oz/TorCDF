//! Drawing of the main window's panels, plus small formatting helpers.

pub mod colorbar;
pub mod inspector;
pub mod map_view;
pub mod plot_area;
pub mod popup;
pub mod shortcuts;
pub mod theme;
pub mod tree;
pub mod windows;

use eframe::egui;
use torcdf_core::cf::TimeUnits;
use torcdf_core::{AttrValue, DType};

use crate::app::Action;
use crate::view::defaults::ViewKind;
use crate::view::playback::{FPS_RANGE, Playback};
use crate::view::var_view::VarView;
use shortcuts::{Command, keys_of};
use std::path::PathBuf;
use windows::Win;

/// A menu entry showing its shortcut on the right.
fn item(text: &str, cmd: Command) -> egui::Button<'_> {
    egui::Button::new(text).shortcut_text(keys_of(cmd))
}

/// Menu bar: File, View and Help (Process and Edit come in later stages).
pub fn menu_bar(
    ui: &mut egui::Ui,
    view: Option<&VarView>,
    recent: &[PathBuf],
    current_file: Option<u64>,
    actions: &mut Vec<Action>,
) {
    egui::MenuBar::new().ui(ui, |ui| {
        ui.label(egui::RichText::new("TorCDF").strong().color(ui.visuals().selection.stroke.color));
        ui.menu_button("File", |ui| {
            if ui.add(item("Open…", Command::Open)).clicked() {
                actions.push(Action::OpenDialog);
                ui.close();
            }
            ui.menu_button("Open recent", |ui| {
                if recent.is_empty() {
                    ui.weak("No recent files");
                }
                for p in recent {
                    let name = p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned());
                    if ui.button(name).on_hover_text(p.display().to_string()).clicked() {
                        actions.push(Action::OpenRecent(p.clone()));
                        ui.close();
                    }
                }
            });
            if ui.add_enabled(current_file.is_some(), item("Close file", Command::CloseFile)).clicked() {
                if let Some(id) = current_file {
                    actions.push(Action::CloseFile(id));
                }
                ui.close();
            }
            // Ctrl+S stays free for saving edited files (stage 3).
            ui.add_enabled_ui(view.is_some(), |ui| {
                ui.menu_button("Export", |ui| {
                    if ui.add(item("Plot as PNG…", Command::ExportPng)).clicked() {
                        actions.push(Action::SaveImage);
                        ui.close();
                    }
                });
            });
            ui.separator();
            if ui.add(item("Settings…", Command::Settings)).clicked() {
                actions.push(Action::ShowWindow(Win::Settings));
                ui.close();
            }
            ui.separator();
            if ui.button("Exit").clicked() {
                actions.push(Action::Exit);
                ui.close();
            }
        });
        ui.menu_button("View", |ui| {
            let Some(v) = view else {
                ui.weak("Select a variable first");
                return;
            };
            for kind in v.kinds() {
                let name = match kind {
                    ViewKind::Map => "Map",
                    ViewKind::Image => "Image",
                    ViewKind::Line => "Line plot",
                    ViewKind::Table => "Table",
                };
                if ui.radio(v.kind == kind, name).clicked() {
                    actions.push(Action::SetKind(kind));
                    ui.close();
                }
            }
            ui.separator();
            let label = if v.kind == ViewKind::Map { "Fit map to window" } else { "Fit image to window" };
            let fittable = matches!(v.kind, ViewKind::Image | ViewKind::Map);
            let fit = ui.add_enabled(fittable, item(label, Command::Fit));
            if fit.clicked() {
                actions.push(Action::FitView);
                ui.close();
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.add(item("Keyboard shortcuts", Command::ShortcutHelp)).clicked() {
                actions.push(Action::ShowWindow(Win::Shortcuts));
                ui.close();
            }
            if ui.button("About TorCDF").clicked() {
                actions.push(Action::ShowWindow(Win::About));
                ui.close();
            }
        });
    });
}

/// One dismissible line per error.
pub fn error_banners(ui: &mut egui::Ui, errors: &[String], actions: &mut Vec<Action>) {
    for (i, e) in errors.iter().enumerate() {
        ui.horizontal(|ui| {
            if ui.small_button("Dismiss").clicked() {
                actions.push(Action::DismissError(i));
            }
            ui.colored_label(ui.visuals().error_fg_color, e);
        });
    }
}

/// One slider with step buttons per non-plotted dimension longer than 1,
/// and a row of play controls.
pub fn slider_strip(
    ui: &mut egui::Ui,
    v: &mut VarView,
    playback: Option<&Playback>,
    actions: &mut Vec<Action>,
) {
    ui.add_space(4.0);
    for d in v.slider_dims() {
        let len = v.var.shape[d];
        let mut pos = v.slider_pos[d].min(len - 1);
        ui.horizontal(|ui| {
            ui.add_sized([90.0, 18.0], egui::Label::new(&v.var.dims[d]).truncate());
            if ui.add_enabled(pos > 0, egui::Button::new("<")).clicked() {
                pos -= 1;
            }
            ui.spacing_mut().slider_width = (ui.available_width() - 290.0).max(80.0);
            ui.add(egui::Slider::new(&mut pos, 0..=len - 1).show_value(false));
            if ui.add_enabled(pos + 1 < len, egui::Button::new(">")).clicked() {
                pos += 1;
            }
            let coord = v.coords.dims.get(d).and_then(|c| c.as_ref()?.get(pos).copied());
            let time = if v.time_dim == Some(d) { v.time_units.as_ref() } else { None };
            ui.label(slider_label(pos, coord, time));
            ui.weak(format!("{}/{len}", pos + 1));
        });
        if pos != v.slider_pos[d] {
            v.slider_pos[d] = pos;
            actions.push(Action::SliderMoved);
        }
    }
    if let Some(pb) = playback {
        play_controls(ui, v, pb, actions);
    }
    ui.add_space(2.0);
}

/// Play/Pause, the dimension played (when there is a choice), speed, loop,
/// and "loading…" while the next frame is still being read.
fn play_controls(ui: &mut egui::Ui, v: &VarView, pb: &Playback, actions: &mut Vec<Action>) {
    ui.horizontal(|ui| {
        let (label, action) = if pb.playing { ("Pause", Action::Pause) } else { ("Play", Action::Play) };
        if ui.add_sized([60.0, 18.0], egui::Button::new(label)).clicked() {
            actions.push(action);
        }
        let dims = v.slider_dims();
        if dims.len() > 1 {
            let name = |d: usize| v.var.dims.get(d).cloned().unwrap_or_default();
            egui::ComboBox::from_id_salt("play_dim").selected_text(name(pb.dim)).show_ui(ui, |ui| {
                for d in dims {
                    if ui.selectable_label(pb.dim == d, name(d)).clicked() && pb.dim != d {
                        actions.push(Action::SetPlayDim(d));
                    }
                }
            });
        }
        let mut fps = pb.fps;
        let speed = egui::Slider::new(&mut fps, FPS_RANGE).step_by(1.0).suffix(" fps");
        if ui.add(speed).changed() {
            actions.push(Action::SetFps(fps));
        }
        let mut looping = pb.looping;
        if ui.checkbox(&mut looping, "Loop").changed() {
            actions.push(Action::SetLoop(looping));
        }
        if pb.playing && pb.waiting {
            ui.weak(match pb.effective_fps {
                Some(f) => format!("loading\u{2026} {f:.1} fps"),
                None => "loading\u{2026}".to_string(),
            });
        }
    });
}

/// "Drop to open" over the whole window while files are dragged over it.
pub fn drop_overlay(ctx: &egui::Context) {
    if ctx.input(|i| i.raw.hovered_files.is_empty()) {
        return;
    }
    let painter =
        ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop")));
    let rect = ctx.content_rect();
    painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(160));
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "Drop to open",
        egui::FontId::proportional(28.0),
        egui::Color32::WHITE,
    );
}

/// Short lower-case type name ("f32", "string", "compound").
pub fn dtype_name(t: &DType) -> String {
    match t {
        DType::Char => "char".into(),
        DType::Str => "string".into(),
        DType::Other(s) => s.clone(),
        other => format!("{other:?}").to_lowercase(),
    }
}

/// An attribute value as one line of text (long lists are shortened).
pub fn attr_text(v: &AttrValue) -> String {
    const MAX: usize = 16;
    fn list<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
        let mut s: Vec<String> = items.iter().take(MAX).map(f).collect();
        if items.len() > MAX {
            s.push(format!("... ({} values)", items.len()));
        }
        s.join(", ")
    }
    match v {
        AttrValue::Text(s) => s.clone(),
        AttrValue::Texts(t) => list(t, |s| format!("\"{s}\"")),
        AttrValue::Ints(i) => list(i, |x| x.to_string()),
        AttrValue::Floats(f) => list(f, |x| fmt_value(*x)),
    }
}

/// File size in binary units with Explorer-style labels ("6.95 GB").
pub fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["KB", "MB", "GB", "TB", "PB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut v = bytes as f64 / 1024.0;
    let mut unit = 0;
    while v >= 1024.0 && unit + 1 < UNITS.len() {
        v /= 1024.0;
        unit += 1;
    }
    // Three significant digits.
    let decimals = if v < 10.0 {
        2
    } else if v < 100.0 {
        1
    } else {
        0
    };
    format!("{v:.decimals$} {}", UNITS[unit])
}

/// A data or coordinate value for display: up to 6 significant digits,
/// scientific notation for very large or small magnitudes, "missing" for NaN.
pub fn fmt_value(v: f64) -> String {
    if v.is_nan() {
        return "missing".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf" } else { "-inf" }.into();
    }
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if !(1e-4..1e6).contains(&a) {
        let s = format!("{v:.5e}");
        let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
        return format!("{}e{exp}", trim_zeros(mantissa));
    }
    let decimals = (5 - a.log10().floor() as i32).max(0) as usize;
    trim_zeros(&format!("{v:.decimals$}")).to_string()
}

/// "1.500" -> "1.5", "2.000" -> "2"; integers are unchanged.
fn trim_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

/// Label of a slider position: the decoded time when the dimension is a CF
/// time axis, else its coordinate value, else the index.
pub fn slider_label(index: usize, coord: Option<f64>, time: Option<&TimeUnits>) -> String {
    match (coord, time) {
        (Some(v), Some(t)) => t.decode(v).to_string(),
        (Some(v), None) => fmt_value(v),
        (None, _) => index.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_sizes() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(1023), "1023 B");
        assert_eq!(human_size(1024), "1.00 KB");
        assert_eq!(human_size(1536), "1.50 KB");
        assert_eq!(human_size(23 * 1024 * 1024 + 300_000), "23.3 MB");
        // GEBCO: 7,464,960,000 bytes is 6.95 GiB.
        assert_eq!(human_size(7_464_960_000), "6.95 GB");
        assert_eq!(human_size(3 << 40), "3.00 TB");
    }

    #[test]
    fn value_formatting() {
        assert_eq!(fmt_value(f64::NAN), "missing");
        assert_eq!(fmt_value(0.0), "0");
        assert_eq!(fmt_value(-4321.0), "-4321");
        assert_eq!(fmt_value(0.125), "0.125");
        assert_eq!(fmt_value(1.0 / 3.0), "0.333333");
        assert_eq!(fmt_value(273.15), "273.15");
        assert_eq!(fmt_value(1.5e20), "1.5e20");
        assert_eq!(fmt_value(-2.5e-7), "-2.5e-7");
        assert_eq!(fmt_value(f64::INFINITY), "inf");
    }

    #[test]
    fn slider_labels() {
        let t = TimeUnits::parse("hours since 2025-11-30 00:00:00", None).unwrap();
        assert_eq!(slider_label(0, Some(0.5), Some(&t)), "2025-11-30 00:30");
        assert_eq!(slider_label(3, Some(1.5), None), "1.5");
        assert_eq!(slider_label(7, None, Some(&t)), "7");
        assert_eq!(slider_label(7, None, None), "7");
        // Unknown calendar: TimeUnits::parse gives None, raw value is shown.
        let none = TimeUnits::parse("days since 2000-01-01", Some("martian"));
        assert_eq!(slider_label(1, Some(10.0), none.as_ref()), "10");
    }
}
