//! Keyboard shortcuts: the one table the menus, the F1 window and the
//! key handling all read.

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

    #[test]
    fn keys_of_matches_table() {
        assert_eq!(keys_of(Command::Open), "Ctrl+O");
        assert_eq!(keys_of(Command::Settings), "Ctrl+,");
        assert_eq!(keys_of(Command::Fit), "Ctrl+0");
        assert_eq!(keys_of(Command::ShortcutHelp), "F1");
    }
}
