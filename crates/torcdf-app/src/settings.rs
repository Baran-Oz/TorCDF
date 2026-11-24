//! Preferences, recent files and window geometry kept between runs in
//! `%APPDATA%\TorCDF\settings.json`.
//!
//! Every field reads leniently: a missing or unreadable value takes its
//! default without discarding the rest, so files from older or newer
//! versions still load. Only a file that is not JSON at all is set aside.

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};

use crate::map::projection::Projection;
use crate::render::colormap::Colormap;

pub const CURRENT_VERSION: u32 = 1;

/// The value, or the type's default when it does not parse.
fn lenient<'de, D: Deserializer<'de>, T: DeserializeOwned + Default>(d: D) -> Result<T, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(T::deserialize(v).unwrap_or_default())
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Look {
    #[default]
    Light,
    Dark,
    /// Follow the Windows app light/dark setting.
    System,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TextSize {
    Smaller,
    #[default]
    Normal,
    Larger,
}

impl TextSize {
    pub fn zoom(self) -> f32 {
        match self {
            TextSize::Smaller => 0.9,
            TextSize::Normal => 1.0,
            TextSize::Larger => 1.25,
        }
    }
}

/// What a new plot starts with.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct PlotDefaults {
    #[serde(deserialize_with = "lenient")]
    pub colormap: Colormap,
    #[serde(deserialize_with = "lenient")]
    pub projection: Projection,
    #[serde(deserialize_with = "lenient")]
    pub coast: bool,
    #[serde(deserialize_with = "lenient")]
    pub borders: bool,
    #[serde(deserialize_with = "lenient")]
    pub grid: bool,
}

impl Default for PlotDefaults {
    fn default() -> Self {
        PlotDefaults {
            colormap: Colormap::Viridis,
            projection: Projection::Equirectangular,
            coast: true,
            borders: true,
            grid: true,
        }
    }
}

/// Window geometry in logical points.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct WindowGeom {
    /// Outer top-left corner.
    pub pos: Option<[f32; 2]>,
    /// Inner size.
    pub size: [f32; 2],
    pub maximized: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct Settings {
    #[serde(deserialize_with = "lenient")]
    pub version: u32,
    #[serde(deserialize_with = "lenient")]
    pub look: Look,
    #[serde(deserialize_with = "lenient")]
    pub text_size: TextSize,
    #[serde(deserialize_with = "lenient")]
    pub plot: PlotDefaults,
    #[serde(deserialize_with = "lenient_recent_max")]
    pub recent_max: usize,
    #[serde(deserialize_with = "lenient")]
    pub reopen_last: bool,
    /// Newest first.
    #[serde(deserialize_with = "lenient")]
    pub recent: Vec<PathBuf>,
    /// Files open at exit, in tree order.
    #[serde(deserialize_with = "lenient")]
    pub last_open: Vec<PathBuf>,
    #[serde(deserialize_with = "lenient")]
    pub window: Option<WindowGeom>,
    /// Formulas of calculated variables made, newest first (typing help).
    #[serde(deserialize_with = "lenient")]
    pub recent_formulas: Vec<RecentFormula>,
}

/// A formula used before: its name, text and units.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Default)]
#[serde(default)]
pub struct RecentFormula {
    pub name: String,
    pub formula: String,
    pub units: String,
}

/// Most formulas kept in [`Settings::recent_formulas`].
const RECENT_FORMULAS_MAX: usize = 20;

const DEFAULT_RECENT_MAX: usize = 10;

fn lenient_recent_max<'de, D: Deserializer<'de>>(d: D) -> Result<usize, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(usize::deserialize(v).unwrap_or(DEFAULT_RECENT_MAX))
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            version: CURRENT_VERSION,
            look: Look::default(),
            text_size: TextSize::default(),
            plot: PlotDefaults::default(),
            recent_max: DEFAULT_RECENT_MAX,
            reopen_last: false,
            recent: Vec::new(),
            last_open: Vec::new(),
            window: None,
            recent_formulas: Vec::new(),
        }
    }
}

/// How loading went.
#[derive(Debug)]
pub enum Loaded {
    /// No file yet.
    Fresh,
    Ok,
    /// The file was unreadable and was set aside; the text is for a banner.
    Recovered(String),
}

/// Case-insensitive identity of a path, as Windows compares them.
fn path_key(p: &Path) -> String {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().to_lowercase()
}

impl Settings {
    pub fn load_from(path: &Path) -> (Settings, Loaded) {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (Settings::default(), Loaded::Fresh),
            Err(e) => {
                return (Settings::default(), Loaded::Recovered(format!("Settings could not be read ({e}); using defaults.")));
            }
        };
        match serde_json::from_slice::<Settings>(&bytes) {
            Ok(mut s) => {
                s.recent.truncate(s.recent_max);
                (s, Loaded::Ok)
            }
            Err(e) => {
                let mut bad = path.as_os_str().to_owned();
                bad.push(".bad");
                let _ = std::fs::rename(path, PathBuf::from(bad));
                (
                    Settings::default(),
                    Loaded::Recovered(format!("Settings file was damaged ({e}); using defaults. The old file was kept as settings.json.bad.")),
                )
            }
        }
    }

    /// Writes a temporary file beside `path`, then renames it over `path`.
    pub fn save_to(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        let json = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }

    /// Back to default preferences; recent files, open files and window
    /// geometry stay.
    pub fn reset_preferences(&mut self) {
        let d = Settings::default();
        self.look = d.look;
        self.text_size = d.text_size;
        self.plot = d.plot;
        self.recent_max = d.recent_max;
        self.reopen_last = d.reopen_last;
    }

    /// `p` becomes the newest recent file (an earlier entry for the same
    /// file is dropped); the list is cut to `recent_max`.
    pub fn push_recent(&mut self, p: &Path) {
        self.remove_recent(p);
        self.recent.insert(0, p.to_path_buf());
        self.recent.truncate(self.recent_max);
    }

    pub fn remove_recent(&mut self, p: &Path) {
        let key = path_key(p);
        self.recent.retain(|r| path_key(r) != key);
    }

    /// `f` becomes the newest recent formula (an earlier one with the same
    /// name and formula is dropped); at most 20 are kept.
    pub fn push_formula(&mut self, f: RecentFormula) {
        self.recent_formulas.retain(|r| !(r.name == f.name && r.formula == f.formula));
        self.recent_formulas.insert(0, f);
        self.recent_formulas.truncate(RECENT_FORMULAS_MAX);
    }
}

/// `%APPDATA%\TorCDF\settings.json`; `None` without `APPDATA`.
pub fn settings_path() -> Option<PathBuf> {
    let appdata = std::env::var_os("APPDATA").filter(|a| !a.is_empty())?;
    Some(PathBuf::from(appdata).join("TorCDF").join("settings.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::projection::Projection;
    use crate::render::colormap::Colormap;
    use std::path::{Path, PathBuf};

    fn file_in(dir: &Path) -> PathBuf {
        dir.join("settings.json")
    }

    fn formula(name: &str, f: &str, units: &str) -> RecentFormula {
        RecentFormula { name: name.into(), formula: f.into(), units: units.into() }
    }

    #[test]
    fn recent_formulas_roundtrip_and_cap() {
        let mut s = Settings::default();
        for i in 0..25 {
            s.push_formula(formula(&format!("v{i}"), &format!("uo*{i}"), ""));
        }
        assert_eq!(s.recent_formulas.len(), 20);
        assert_eq!(s.recent_formulas[0].name, "v24");
        // A repeat moves to the front, with its new units.
        s.push_formula(formula("v10", "uo*10", "m"));
        assert_eq!(s.recent_formulas.len(), 20);
        assert_eq!(s.recent_formulas[0], formula("v10", "uo*10", "m"));
        assert_eq!(s.recent_formulas.iter().filter(|r| r.name == "v10").count(), 1);
        let dir = tempfile::tempdir().unwrap();
        let path = file_in(dir.path());
        s.save_to(&path).unwrap();
        assert_eq!(Settings::load_from(&path).0.recent_formulas, s.recent_formulas);
        // Older settings files have none.
        std::fs::write(&path, r#"{"version": 1}"#).unwrap();
        assert!(Settings::load_from(&path).0.recent_formulas.is_empty());
    }

    #[test]
    fn missing_file_is_fresh_defaults() {
        let d = tempfile::tempdir().unwrap();
        let (s, l) = Settings::load_from(&file_in(d.path()));
        assert!(matches!(l, Loaded::Fresh));
        assert_eq!(s, Settings::default());
        assert_eq!(s.look, Look::Light);
        assert_eq!(s.recent_max, 10);
        assert_eq!(s.plot.colormap, Colormap::Viridis);
        assert!(!s.reopen_last);
    }

    fn non_default() -> Settings {
        Settings {
            version: CURRENT_VERSION,
            look: Look::Dark,
            text_size: TextSize::Larger,
            plot: PlotDefaults {
                colormap: Colormap::Topo,
                projection: Projection::Mercator,
                coast: false,
                borders: false,
                grid: false,
            },
            recent_max: 5,
            reopen_last: true,
            recent: vec![PathBuf::from(r"C:\a.nc")],
            last_open: vec![PathBuf::from(r"C:\b.nc")],
            window: Some(WindowGeom { pos: Some([10.0, 20.0]), size: [800.0, 600.0], maximized: true }),
            recent_formulas: vec![RecentFormula { name: "s".into(), formula: "u*2".into(), units: "m".into() }],
        }
    }

    #[test]
    fn round_trip() {
        let d = tempfile::tempdir().unwrap();
        let p = file_in(d.path());
        let s = non_default();
        s.save_to(&p).unwrap();
        let (back, l) = Settings::load_from(&p);
        assert!(matches!(l, Loaded::Ok));
        assert_eq!(back, s);
    }

    #[test]
    fn unknown_and_missing_fields() {
        let d = tempfile::tempdir().unwrap();
        let p = file_in(d.path());
        std::fs::write(&p, r#"{"version":1,"look":"Dark","future":5}"#).unwrap();
        let (s, l) = Settings::load_from(&p);
        assert!(matches!(l, Loaded::Ok));
        assert_eq!(s, Settings { look: Look::Dark, ..Settings::default() });
    }

    #[test]
    fn bad_enum_value_falls_back_per_field() {
        let d = tempfile::tempdir().unwrap();
        let p = file_in(d.path());
        std::fs::write(&p, r#"{"look":"Purple","recent_max":5,"plot":{"colormap":"Rainbow","coast":false}}"#)
            .unwrap();
        let (s, _) = Settings::load_from(&p);
        assert_eq!(s.look, Look::Light);
        assert_eq!(s.recent_max, 5);
        assert_eq!(s.plot.colormap, Colormap::Viridis);
        assert!(!s.plot.coast);
    }

    #[test]
    fn newer_version_best_effort() {
        let d = tempfile::tempdir().unwrap();
        let p = file_in(d.path());
        std::fs::write(&p, r#"{"version":99,"reopen_last":true}"#).unwrap();
        let (s, l) = Settings::load_from(&p);
        assert!(matches!(l, Loaded::Ok));
        assert!(s.reopen_last);
    }

    #[test]
    fn truncated_file_recovered() {
        let d = tempfile::tempdir().unwrap();
        let p = file_in(d.path());
        std::fs::write(&p, r#"{"look":"Da"#).unwrap();
        let (s, l) = Settings::load_from(&p);
        assert_eq!(s, Settings::default());
        assert!(matches!(l, Loaded::Recovered(_)));
        assert!(d.path().join("settings.json.bad").exists());
        assert!(!p.exists());
    }

    #[test]
    fn save_is_atomic() {
        let d = tempfile::tempdir().unwrap();
        let p = file_in(d.path());
        non_default().save_to(&p).unwrap();
        Settings::default().save_to(&p).unwrap();
        assert!(!d.path().join("settings.json.tmp").exists());
        let (s, l) = Settings::load_from(&p);
        assert!(matches!(l, Loaded::Ok));
        assert_eq!(s, Settings::default());
    }

    #[test]
    fn save_creates_parent_dir() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("TorCDF").join("settings.json");
        Settings::default().save_to(&p).unwrap();
        assert!(p.exists());
    }

    #[test]
    fn recent_dedup_order_cap() {
        let mut s = Settings { recent_max: 3, ..Settings::default() };
        for p in [r"C:\x\a.nc", r"C:\x\b.nc", r"C:\x\c.nc", r"C:\X\A.nc", r"C:\x\d.nc"] {
            s.push_recent(Path::new(p));
        }
        let want: Vec<PathBuf> = [r"C:\x\d.nc", r"C:\X\A.nc", r"C:\x\c.nc"].iter().map(PathBuf::from).collect();
        assert_eq!(s.recent, want);
    }

    #[test]
    fn recent_max_zero_keeps_none() {
        let mut s = Settings { recent_max: 0, ..Settings::default() };
        s.push_recent(Path::new(r"C:\x\a.nc"));
        assert!(s.recent.is_empty());
    }

    #[test]
    fn remove_recent_ignores_case() {
        let mut s = Settings::default();
        s.push_recent(Path::new(r"C:\x\a.nc"));
        s.push_recent(Path::new(r"C:\x\b.nc"));
        s.remove_recent(Path::new(r"c:\X\A.NC"));
        assert_eq!(s.recent, vec![PathBuf::from(r"C:\x\b.nc")]);
    }

    #[test]
    fn reset_keeps_files_and_window() {
        let mut s = non_default();
        s.reset_preferences();
        let d = Settings::default();
        assert_eq!(s.recent, non_default().recent);
        assert_eq!(s.last_open, non_default().last_open);
        assert_eq!(s.window, non_default().window);
        assert_eq!(s.look, d.look);
        assert_eq!(s.text_size, d.text_size);
        assert_eq!(s.plot, d.plot);
        assert_eq!(s.recent_max, d.recent_max);
        assert_eq!(s.reopen_last, d.reopen_last);
    }

    #[test]
    fn non_ascii_path_round_trip() {
        let d = tempfile::tempdir().unwrap();
        let p = file_in(d.path());
        let mut s = Settings::default();
        s.push_recent(Path::new(r"C:\data\Ölçüm\ağ.nc"));
        s.save_to(&p).unwrap();
        let (back, _) = Settings::load_from(&p);
        assert_eq!(back.recent, vec![PathBuf::from(r"C:\data\Ölçüm\ağ.nc")]);
    }

    #[test]
    fn text_size_zoom() {
        assert_eq!(TextSize::Smaller.zoom(), 0.9);
        assert_eq!(TextSize::Normal.zoom(), 1.0);
        assert_eq!(TextSize::Larger.zoom(), 1.25);
    }
}
