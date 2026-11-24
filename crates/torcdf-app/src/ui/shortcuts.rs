//! Keyboard shortcuts: the one table the menus, the F1 window and the
//! key handling all read.

use eframe::egui;

use crate::app::Selection;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Command {
    Open,
    CloseFile,
    ExportPng,
    Settings,
    StepBack,
    StepForward,
    FirstStep,
    LastStep,
    PlayPause,
    Fit,
    PrevVar,
    NextVar,
    ShortcutHelp,
}

impl Command {
    #[cfg(test)]
    pub const ALL: [Command; 13] = [
        Command::Open,
        Command::CloseFile,
        Command::ExportPng,
        Command::Settings,
        Command::StepBack,
        Command::StepForward,
        Command::FirstStep,
        Command::LastStep,
        Command::PlayPause,
        Command::Fit,
        Command::PrevVar,
        Command::NextVar,
        Command::ShortcutHelp,
    ];
}

pub struct Shortcut {
    pub cmd: Command,
    /// As shown to the user.
    pub keys: &'static str,
    pub label: &'static str,
}

/// Ctrl+S is left free for saving edited files (stage 3).
pub const SHORTCUTS: &[Shortcut] = &[
    Shortcut { cmd: Command::Open, keys: "Ctrl+O", label: "Open file" },
    Shortcut { cmd: Command::CloseFile, keys: "Ctrl+W", label: "Close file" },
    Shortcut { cmd: Command::ExportPng, keys: "Ctrl+E", label: "Export plot as PNG" },
    Shortcut { cmd: Command::Settings, keys: "Ctrl+,", label: "Settings" },
    Shortcut { cmd: Command::StepBack, keys: "←", label: "Previous time step" },
    Shortcut { cmd: Command::StepForward, keys: "→", label: "Next time step" },
    Shortcut { cmd: Command::FirstStep, keys: "Home", label: "First time step" },
    Shortcut { cmd: Command::LastStep, keys: "End", label: "Last time step" },
    Shortcut { cmd: Command::PlayPause, keys: "Space", label: "Play / pause animation" },
    Shortcut { cmd: Command::Fit, keys: "Ctrl+0", label: "Fit map / image to window" },
    Shortcut { cmd: Command::PrevVar, keys: "↑", label: "Previous variable in the tree" },
    Shortcut { cmd: Command::NextVar, keys: "↓", label: "Next variable in the tree" },
    Shortcut { cmd: Command::ShortcutHelp, keys: "F1", label: "This list" },
];

pub fn keys_of(cmd: Command) -> &'static str {
    SHORTCUTS.iter().find(|s| s.cmd == cmd).map_or("", |s| s.keys)
}

/// The commands pressed this frame, consumed so no widget also acts on
/// them. Keys without Ctrl act only while no widget has keyboard focus
/// (a text field being typed in, or a Tab-focused checkbox) and no menu or
/// drop-down list is open (those navigate with the arrows and Space).
/// Clicking does not give focus in egui, so mouse users always get them.
/// Call before the panels draw.
pub fn pressed(ctx: &egui::Context) -> Vec<Command> {
    use egui::{Key, Modifiers};
    let ctrl: [(Key, Command); 5] = [
        (Key::O, Command::Open),
        (Key::W, Command::CloseFile),
        (Key::E, Command::ExportPng),
        (Key::Comma, Command::Settings),
        (Key::Num0, Command::Fit),
    ];
    let plain: [(Key, Command); 7] = [
        (Key::ArrowLeft, Command::StepBack),
        (Key::ArrowRight, Command::StepForward),
        (Key::Home, Command::FirstStep),
        (Key::End, Command::LastStep),
        (Key::Space, Command::PlayPause),
        (Key::ArrowUp, Command::PrevVar),
        (Key::ArrowDown, Command::NextVar),
    ];
    let busy = ctx.memory(|m| m.focused().is_some()) || egui::Popup::is_any_open(ctx);
    ctx.input_mut(|i| {
        let mut out = Vec::new();
        for (k, c) in ctrl {
            if i.consume_key(Modifiers::COMMAND, k) {
                out.push(c);
            }
        }
        if i.consume_key(Modifiers::NONE, Key::F1) {
            out.push(Command::ShortcutHelp);
        }
        if !busy {
            for (k, c) in plain {
                if i.consume_key(Modifiers::NONE, k) {
                    out.push(c);
                }
            }
        }
        out
    })
}

/// The variable `step` places after `current` in `order`, stopping at
/// the ends. With no variable selected, the first (forward) or last.
pub fn neighbour(order: &[Selection], current: Option<&Selection>, step: i32) -> Option<Selection> {
    let last = order.len().checked_sub(1)?;
    let at = current.and_then(|c| order.iter().position(|s| s == c));
    let i = match at {
        Some(i) => (i as i64 + step as i64).clamp(0, last as i64) as usize,
        None if step < 0 => last,
        None => 0,
    };
    order.get(i).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_listed_once() {
        for c in Command::ALL {
            assert_eq!(SHORTCUTS.iter().filter(|s| s.cmd == c).count(), 1, "{c:?}");
        }
        assert_eq!(SHORTCUTS.len(), Command::ALL.len());
    }

    #[test]
    fn no_ctrl_s() {
        assert!(SHORTCUTS.iter().all(|s| !s.keys.contains("Ctrl+S")));
    }



    fn v(name: &str) -> Selection {
        Selection::Var(1, name.into())
    }

    #[test]
    fn neighbour_moves_and_clamps() {
        let order = [v("a"), v("b"), v("c")];
        assert_eq!(neighbour(&order, Some(&v("b")), 1), Some(v("c")));
        assert_eq!(neighbour(&order, Some(&v("c")), 1), Some(v("c")));
        assert_eq!(neighbour(&order, Some(&v("a")), -1), Some(v("a")));
        assert_eq!(neighbour(&order, None, 1), Some(v("a")));
        assert_eq!(neighbour(&order, None, -1), Some(v("c")));
        // A selected file or group: start from the first variable.
        assert_eq!(neighbour(&order, Some(&Selection::File(1)), 1), Some(v("a")));
        assert_eq!(neighbour(&[], None, 1), None);
    }

    fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }
    }

    /// Runs two passes: the first focuses a text field when `focus`; the
    /// second delivers `events` and returns what `pressed` read.
    fn press(events: Vec<egui::Event>, focus: bool) -> (Vec<Command>, usize) {
        let ctx = egui::Context::default();
        let mut text = String::new();
        let draw = |ui: &mut egui::Ui, text: &mut String| {
            let r = ui.text_edit_singleline(text);
            if focus {
                r.request_focus();
            }
        };
        ctx.run_ui(Default::default(), |ui| draw(ui, &mut text)).textures_delta.clear();
        let mut got = Vec::new();
        let mut left = 0;
        let input = egui::RawInput { events, ..Default::default() };
        ctx.run_ui(input, |ui| {
            got = pressed(ui.ctx());
            left = ui.ctx().input(|i| i.events.len());
            draw(ui, &mut text);
        })
        .textures_delta
        .clear();
        (got, left)
    }

    #[test]
    fn focused_button_keeps_plain_keys() {
        // A Tab-focused button or checkbox: Space and arrows stay egui's.
        let ctx = egui::Context::default();
        let draw = |ui: &mut egui::Ui| {
            let r = ui.button("Loop");
            r.request_focus();
        };
        ctx.run_ui(Default::default(), draw).textures_delta.clear();
        let mut got = Vec::new();
        let input = egui::RawInput { events: vec![key(egui::Key::Space, egui::Modifiers::NONE)], ..Default::default() };
        ctx.run_ui(input, |ui| {
            got = pressed(ui.ctx());
            draw(ui);
        })
        .textures_delta
        .clear();
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn open_popup_keeps_plain_keys() {
        // An open menu or drop-down list navigates with the arrows itself.
        let ctx = egui::Context::default();
        let id = egui::Id::unique("menu");
        let draw = |ui: &mut egui::Ui| {
            egui::Popup::open_id(ui.ctx(), id);
        };
        ctx.run_ui(Default::default(), draw).textures_delta.clear();
        let mut got = Vec::new();
        let input =
            egui::RawInput { events: vec![key(egui::Key::ArrowDown, egui::Modifiers::NONE)], ..Default::default() };
        ctx.run_ui(input, |ui| {
            got = pressed(ui.ctx());
            draw(ui);
        })
        .textures_delta
        .clear();
        assert!(got.is_empty(), "{got:?}");
    }

    #[test]
    fn plain_keys_step_without_focus() {
        let (got, _) = press(vec![key(egui::Key::ArrowRight, egui::Modifiers::NONE)], false);
        assert_eq!(got, vec![Command::StepForward]);
        let (got, _) = press(vec![key(egui::Key::Space, egui::Modifiers::NONE)], false);
        assert_eq!(got, vec![Command::PlayPause]);
    }

    #[test]
    fn text_field_focus_blocks_plain_keys() {
        let (got, _) = press(vec![key(egui::Key::ArrowRight, egui::Modifiers::NONE)], true);
        assert!(got.is_empty(), "{got:?}");
        let (got, _) = press(vec![key(egui::Key::O, egui::Modifiers::COMMAND)], true);
        assert_eq!(got, vec![Command::Open]);
    }

    #[test]
    fn ctrl_comma_and_zero() {
        let (got, _) = press(vec![key(egui::Key::Comma, egui::Modifiers::COMMAND)], false);
        assert_eq!(got, vec![Command::Settings]);
        let (got, _) = press(vec![key(egui::Key::Num0, egui::Modifiers::COMMAND)], false);
        assert_eq!(got, vec![Command::Fit]);
    }

    #[test]
    fn ctrl_s_not_consumed() {
        let (got, left) = press(vec![key(egui::Key::S, egui::Modifiers::COMMAND)], false);
        assert!(got.is_empty());
        assert_eq!(left, 1);
    }

    #[test]
    fn keys_of_matches_table() {
        assert_eq!(keys_of(Command::Open), "Ctrl+O");
        assert_eq!(keys_of(Command::Settings), "Ctrl+,");
        assert_eq!(keys_of(Command::Fit), "Ctrl+0");
        assert_eq!(keys_of(Command::ShortcutHelp), "F1");
    }
}
