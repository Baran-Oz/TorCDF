//! Left panel: open files with their dimensions, groups and variables.

use eframe::egui;
use egui::collapsing_header::CollapsingState;
use torcdf_core::{Group, RotateChoice};

use super::{dtype_name, human_size};
use crate::app::{Action, OpenFile, Opening, Selection};

pub fn show(
    ui: &mut egui::Ui,
    files: &[OpenFile],
    opening: &[Opening],
    sel: Option<&Selection>,
    scroll_to: bool,
    actions: &mut Vec<Action>,
) {
    egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
        for f in files {
            let id = ui.make_persistent_id(("file", f.id));
            CollapsingState::load_with_default_open(ui.ctx(), id, true)
                .show_header(ui, |ui| {
                    let text = format!("{}  {}", f.name, human_size(f.nc.meta.file_size));
                    let selected = sel == Some(&Selection::File(f.id));
                    let r = ui.selectable_label(selected, egui::RichText::new(text).strong());
                    let r = r.on_hover_text(f.nc.meta.path.display().to_string());
                    if r.clicked() {
                        actions.push(Action::Select(Selection::File(f.id)));
                    }
                    r.context_menu(|ui| {
                        if ui.button("Calculate new variable…").clicked() {
                            actions.push(Action::OpenCalculate { file: f.id, edit: None });
                            ui.close();
                        }
                        if ui.button("Close file").clicked() {
                            actions.push(Action::CloseFile(f.id));
                            ui.close();
                        }
                    });
                })
                .body(|ui| group_body(ui, f, &f.nc.meta.root, sel, scroll_to, actions));
        }
        for o in opening {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("Opening {}...", o.name));
            });
        }
        if files.is_empty() && opening.is_empty() {
            ui.weak("No files open.\nUse File > Open or drop files on the window.");
        }
    });
}

fn group_body(
    ui: &mut egui::Ui,
    file: &OpenFile,
    g: &Group,
    sel: Option<&Selection>,
    scroll_to: bool,
    actions: &mut Vec<Action>,
) {
    let file_id = file.id;
    if !g.dims.is_empty() {
        egui::CollapsingHeader::new(format!("Dimensions ({})", g.dims.len()))
            .id_salt(("dims", file_id, &g.path))
            .show(ui, |ui| {
                for d in &g.dims {
                    let unlimited = if d.unlimited { "  unlimited" } else { "" };
                    ui.label(format!("{} ({}){unlimited}", d.name, d.len));
                }
            });
    }
    for sub in &g.groups {
        let id = ui.make_persistent_id(("group", file_id, &sub.path));
        CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                let s = Selection::Group(file_id, sub.path.clone());
                if ui.selectable_label(sel == Some(&s), format!("{}/", sub.name)).clicked() {
                    actions.push(Action::Select(s));
                }
            })
            .body(|ui| group_body(ui, file, sub, sel, scroll_to, actions));
    }
    for v in &g.vars {
        let s = Selection::Var(file_id, v.path.clone());
        let text = format!("{}  ({})  {}", v.name, v.dims.join(", "), dtype_name(&v.dtype));
        let shape: Vec<String> = v.shape.iter().map(usize::to_string).collect();
        let r = ui
            .selectable_label(sel == Some(&s), text)
            .on_hover_text(format!("{}  shape [{}]", v.path, shape.join(" × ")));
        accent_edge(ui, &r, sel == Some(&s), scroll_to);
        if r.clicked() {
            actions.push(Action::Select(s));
        }
        r.context_menu(|ui| extract_item(ui, file, &v.path, actions));
    }
    // Calculated variables of this group.
    for d in file.nc.derived_vars().iter().filter(|d| parent(&d.path) == g.path) {
        let s = Selection::Var(file_id, d.path.clone());
        let text = format!("ƒ {}  ({})", d.name, d.info.dims.join(", "));
        let r = ui.selectable_label(sel == Some(&s), text).on_hover_text(format!("= {}", d.formula));
        accent_edge(ui, &r, sel == Some(&s), scroll_to);
        if r.clicked() {
            actions.push(Action::Select(s));
        }
        r.context_menu(|ui| {
            if ui.button("Edit…").clicked() {
                actions.push(Action::OpenCalculate { file: file_id, edit: Some(d.path.clone()) });
                ui.close();
            }
            extract_item(ui, file, &d.path, actions);
            if ui.button("Delete").clicked() {
                actions.push(Action::DeleteDerived { file: file_id, path: d.path.clone() });
                ui.close();
            }
        });
    }
    // Registered pairs of this group. A rotation variant (tagged path) is
    // the same row as its pair.
    for p in file.nc.pairs().iter().filter(|p| p.rotate == RotateChoice::Auto) {
        if file.nc.meta.group_of(&p.east).path != g.path {
            continue;
        }
        let base = p.path.clone();
        let selected = matches!(sel, Some(Selection::Var(f, path))
            if *f == file_id && path.strip_prefix(&base).is_some_and(|t| t.is_empty() || t.starts_with('#')));
        let r = ui
            .selectable_label(selected, &p.name)
            .on_hover_text(format!("vector: {} + {}", p.east, p.north));
        accent_edge(ui, &r, selected, scroll_to);
        if r.clicked() {
            actions.push(Action::Select(Selection::Var(file_id, base.clone())));
        }
        r.context_menu(|ui| {
            if ui.button("Unpair").clicked() {
                actions.push(Action::Unpair { file: file_id, path: base.clone() });
                ui.close();
            }
        });
    }
}

/// The group path of a variable path (`"/"` for the root).
fn parent(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) | None => "/",
        Some(i) => &path[..i],
    }
}

/// The 2 px accent bar on the left of a selected row; scrolls it into
/// view when the selection just moved by keyboard.
fn accent_edge(ui: &egui::Ui, r: &egui::Response, selected: bool, scroll_to: bool) {
    if selected {
        let rect = egui::Rect::from_min_size(r.rect.min, egui::vec2(2.0, r.rect.height()));
        ui.painter().rect_filled(rect, 0.0, ui.visuals().selection.stroke.color);
        if scroll_to {
            r.scroll_to_me(None);
        }
    }
}

/// Every variable and vector-pair row, in the order the tree shows them
/// (groups expanded or not): for ↑/↓.
pub fn var_order(files: &[OpenFile]) -> Vec<Selection> {
    fn walk(file: &OpenFile, g: &Group, out: &mut Vec<Selection>) {
        for sub in &g.groups {
            walk(file, sub, out);
        }
        out.extend(g.vars.iter().map(|v| Selection::Var(file.id, v.path.clone())));
        for d in file.nc.derived_vars().iter().filter(|d| parent(&d.path) == g.path) {
            out.push(Selection::Var(file.id, d.path.clone()));
        }
        for p in file.nc.pairs().iter().filter(|p| p.rotate == RotateChoice::Auto) {
            if file.nc.meta.group_of(&p.east).path == g.path {
                out.push(Selection::Var(file.id, p.path.clone()));
            }
        }
    }
    let mut out = Vec::new();
    for f in files {
        walk(f, &f.nc.meta.root, &mut out);
    }
    out
}

/// "Extract…" for the variable at `path` (netCDF files only).
fn extract_item(ui: &mut egui::Ui, file: &OpenFile, path: &str, actions: &mut Vec<Action>) {
    let ok = file.nc.is_netcdf();
    let r = ui.add_enabled(ok, egui::Button::new("Extract…")).on_disabled_hover_text("Extract works on netCDF files");
    if r.clicked() {
        actions.push(Action::OpenExtract { file: file.id, var: Some(path.to_string()) });
        ui.close();
    }
}
