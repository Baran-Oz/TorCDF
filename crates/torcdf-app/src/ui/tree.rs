//! Left panel: open files with their dimensions, groups and variables.

use eframe::egui;
use egui::collapsing_header::CollapsingState;
use torcdf_core::{Dataset, Group, RotateChoice, VarInfo, can_pair};

use super::{dtype_name, human_size};
use crate::app::{Action, OpenFile, Opening, Selection};

pub fn show(
    ui: &mut egui::Ui,
    files: &[OpenFile],
    opening: &[Opening],
    sel: Option<&Selection>,
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
                        if ui.button("Close file").clicked() {
                            actions.push(Action::CloseFile(f.id));
                            ui.close();
                        }
                    });
                })
                .body(|ui| group_body(ui, f, &f.nc.meta.root, sel, actions));
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
            .body(|ui| group_body(ui, file, sub, sel, actions));
    }
    for v in &g.vars {
        let s = Selection::Var(file_id, v.path.clone());
        let text = format!("{}  ({})  {}", v.name, v.dims.join(", "), dtype_name(&v.dtype));
        let shape: Vec<String> = v.shape.iter().map(usize::to_string).collect();
        let r = ui
            .selectable_label(sel == Some(&s), text)
            .on_hover_text(format!("{}  shape [{}]", v.path, shape.join(" × ")));
        if r.clicked() {
            actions.push(Action::Select(s));
        }
        if v.dtype.is_numeric() {
            r.context_menu(|ui| {
                let others = partners(&file.nc.meta, &v.path);
                for (title, items) in pair_menus(file_id, &v.path, &others) {
                    ui.add_enabled_ui(!items.is_empty(), |ui| {
                        ui.menu_button(title, |ui| {
                            for (label, action) in items {
                                if ui.button(label).clicked() {
                                    actions.push(action);
                                    ui.close();
                                }
                            }
                        });
                    });
                }
            });
        }
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

/// The two "Pair …" submenus of variable `var_path`: (title, [(item label,
/// action)]). Each names the role of both variables, so which one becomes
/// east is never left implicit (spec §4).
fn pair_menus(file: u64, var_path: &str, others: &[String]) -> [(String, Vec<(String, Action)>); 2] {
    let leaf = |p: &str| p.rsplit('/').next().unwrap_or(p).to_string();
    let me = leaf(var_path);
    let menu = |me_east: bool| {
        let (role, other_role) = if me_east { ("east", "north") } else { ("north", "east") };
        let items = others
            .iter()
            .map(|p| {
                let (east, north) =
                    if me_east { (var_path.to_string(), p.clone()) } else { (p.clone(), var_path.to_string()) };
                (format!("{} (as {other_role})", leaf(p)), Action::PairWith { file, east, north })
            })
            .collect();
        (format!("Pair {me} as {role} with"), items)
    };
    [menu(true), menu(false)]
}

/// Variables of the same group that `can_pair` accepts with `var_path`,
/// sorted by name. Coordinate variables (dimension coordinates and any
/// variable named by a `coordinates` attribute) and non-numeric variables
/// never appear.
pub fn partners(ds: &Dataset, var_path: &str) -> Vec<String> {
    let group = ds.group_of(var_path);
    let named: Vec<&str> = group
        .vars
        .iter()
        .filter_map(|v| v.attrs.iter().find(|a| a.name == "coordinates")?.value.as_str())
        .flat_map(str::split_whitespace)
        .map(|n| n.rsplit('/').next().unwrap_or(n))
        .collect();
    let is_coord = |v: &VarInfo| {
        named.contains(&v.name.as_str()) || (v.dims.len() == 1 && v.dims[0] == v.name)
    };
    let mut out: Vec<&VarInfo> = group
        .vars
        .iter()
        .filter(|v| v.path != var_path && v.dtype.is_numeric() && !is_coord(v))
        .filter(|v| can_pair(ds, var_path, &v.path))
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out.into_iter().map(|v| v.path.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use torcdf_core::{NcFile, fixtures};

    fn open(f: impl Fn(&std::path::Path) -> std::path::PathBuf) -> NcFile {
        let d = tempfile::tempdir().unwrap();
        NcFile::open(&f(d.path())).unwrap()
    }

    #[test]
    fn partners_lists_compatible() {
        let nc4 = open(fixtures::write_nc4);
        assert_eq!(partners(&nc4.meta, "/ocean/uo"), vec!["/ocean/vo".to_string()]);
        let noclue = open(|d| fixtures::vectors::write_noclue(d, false));
        assert_eq!(partners(&noclue.meta, "/u"), vec!["/h".to_string(), "/v".to_string()]);
        // Coordinates and the variable itself never appear.
        let of_h = partners(&noclue.meta, "/h");
        assert!(!of_h.iter().any(|p| p == "/lat" || p == "/lon" || p == "/h"), "{of_h:?}");
    }

    /// The menu says which variable becomes east and which north, and the
    /// action carries exactly that choice.
    #[test]
    fn pair_menu_names_the_roles() {
        let menus = pair_menus(7, "/v", &["/u".to_string()]);
        let got: Vec<(String, String, String, String)> = menus
            .into_iter()
            .flat_map(|(title, items)| {
                items.into_iter().map(move |(label, action)| match action {
                    Action::PairWith { file: 7, east, north } => (title.clone(), label, east, north),
                    _ => panic!("not a pairing"),
                })
            })
            .collect();
        let row = |t: &str, l: &str, e: &str, n: &str| (t.into(), l.into(), e.into(), n.into());
        assert_eq!(
            got,
            vec![
                row("Pair v as east with", "u (as north)", "/v", "/u"),
                row("Pair v as north with", "u (as east)", "/u", "/v"),
            ]
        );
    }
}
