//! The window's arrow slot: which pair draws arrows over the coloured
//! variable (the Plot tab's Arrows section), or Off.
//!
//! One slot per window. It belongs to one file: selecting another file's
//! variable starts a fresh slot. Each form (x & y, magnitude & direction)
//! keeps its own pre-filled picks, so switching forms never pairs one
//! form's variables as the other.

use std::sync::Arc;

use torcdf_core::{Dataset, NcFile, PairForm, VarInfo, VectorPair};

/// The section says this when a form has no two variables picked.
pub const PICK_TWO: &str = "Pick two variables";

/// How the arrows are made, or Off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrowMode {
    Off,
    EastNorth,
    MagDir,
}

impl ArrowMode {
    /// The mode that draws a pair of `form`.
    pub fn of_form(form: PairForm) -> ArrowMode {
        match form {
            PairForm::EastNorth => ArrowMode::EastNorth,
            PairForm::MagDir(_) => ArrowMode::MagDir,
        }
    }

    /// Index into the per-form pre-fill; `None` for Off.
    fn index(self) -> Option<usize> {
        match self {
            ArrowMode::Off => None,
            ArrowMode::EastNorth => Some(0),
            ArrowMode::MagDir => Some(1),
        }
    }
}

/// The arrow slot of the window.
pub struct ArrowLayer {
    pub file: u64,
    pub mode: ArrowMode,
    /// x / magnitude and y / direction variable paths.
    pub picks: [Option<String>; 2],
    /// The registered pair drawn when `mode` is not Off.
    pub pair: Option<Arc<VectorPair>>,
    /// One line for the section, e.g. [`PICK_TWO`] or why pairing failed.
    pub problem: Option<String>,
    /// Picks per form (x & y, magnitude & direction), loaded when that
    /// form is chosen.
    prefill: [Option<[String; 2]>; 2],
    /// The pair a drop-down edit is replacing, kept through failed picks
    /// until a pairing succeeds (then it is unregistered).
    pub(crate) editing: Option<Arc<VectorPair>>,
}

impl ArrowLayer {
    /// Off, for plain variable `x` of `nc`. Each form is pre-filled from the
    /// first registered pair of that form that fits `x`; the picks shown
    /// are those of the first fitting pair of either form.
    pub fn fresh(file: u64, nc: &NcFile, x: &VarInfo) -> ArrowLayer {
        let mut prefill: [Option<[String; 2]>; 2] = [None, None];
        let mut first = None;
        for p in nc.pairs() {
            let Some(i) = ArrowMode::of_form(p.form).index() else { continue };
            if prefill[i].is_some() || !nc.layer_fits(&x.path, &p) {
                continue;
            }
            prefill[i] = Some([p.east.clone(), p.north.clone()]);
            first.get_or_insert(i);
        }
        let mut layer = ArrowLayer {
            file,
            mode: ArrowMode::Off,
            picks: [None, None],
            pair: None,
            problem: None,
            prefill,
            editing: None,
        };
        if let Some(i) = first {
            layer.picks = picks_of(&layer.prefill[i]);
        }
        layer.problem = layer.missing_picks().then(|| PICK_TWO.to_string());
        layer
    }

    /// On, drawing `pair` (a pair's own tree entry).
    pub fn of_pair(file: u64, pair: Arc<VectorPair>) -> ArrowLayer {
        let mode = ArrowMode::of_form(pair.form);
        let picks = [pair.east.clone(), pair.north.clone()];
        let mut prefill: [Option<[String; 2]>; 2] = [None, None];
        if let Some(i) = mode.index() {
            prefill[i] = Some(picks.clone());
        }
        ArrowLayer {
            file,
            mode,
            picks: picks_of(&Some(picks)),
            pair: Some(pair),
            problem: None,
            prefill,
            editing: None,
        }
    }

    /// Takes `mode`'s pre-filled picks (none when it has none).
    pub fn load_prefill(&mut self, mode: ArrowMode) {
        self.picks = mode.index().map_or([None, None], |i| picks_of(&self.prefill[i]));
    }

    /// Keeps the current picks as the current form's pre-fill.
    pub fn remember_picks(&mut self) {
        if let (Some(i), [Some(a), Some(b)]) = (self.mode.index(), &self.picks) {
            self.prefill[i] = Some([a.clone(), b.clone()]);
        }
    }

    /// Fewer than two variables are picked.
    pub fn missing_picks(&self) -> bool {
        self.picks.iter().any(Option::is_none)
    }
}

fn picks_of(p: &Option<[String; 2]>) -> [Option<String>; 2] {
    match p {
        Some([a, b]) => [Some(a.clone()), Some(b.clone())],
        None => [None, None],
    }
}

/// Variables the drop-downs of `mode` offer when `x` is coloured: numeric,
/// non-coordinate variables of `x`'s group. x & y: the same rank, with
/// `x`'s shape or one staggered by ±1 on the last two axes. Magnitude &
/// direction: exactly `x`'s dimensions. Off: none.
#[cfg_attr(not(test), expect(dead_code, reason = "the Plot tab's drop-downs list these"))]
pub fn candidates(ds: &Dataset, x: &VarInfo, mode: ArrowMode) -> Vec<String> {
    let group = ds.group_of(&x.path);
    let named: Vec<&str> = group
        .vars
        .iter()
        .filter_map(|v| v.attrs.iter().find(|a| a.name == "coordinates")?.value.as_str())
        .flat_map(str::split_whitespace)
        .map(|n| n.rsplit('/').next().unwrap_or(n))
        .collect();
    let is_coord = |v: &VarInfo| named.contains(&v.name.as_str()) || (v.dims.len() == 1 && v.dims[0] == v.name);
    let fits = |v: &VarInfo| match mode {
        ArrowMode::Off => false,
        ArrowMode::MagDir => v.dims == x.dims,
        ArrowMode::EastNorth => staggered_or_same(&v.shape, &x.shape),
    };
    group
        .vars
        .iter()
        .filter(|v| v.dtype.is_numeric() && !is_coord(v) && fits(v))
        .map(|v| v.path.clone())
        .collect()
}

/// Same rank; equal on every axis except the last two, which may differ by
/// one either way.
#[cfg_attr(not(test), expect(dead_code, reason = "used by `candidates`"))]
fn staggered_or_same(a: &[usize], b: &[usize]) -> bool {
    let n = a.len();
    if n != b.len() || n < 2 {
        return false;
    }
    a.iter().zip(b).enumerate().all(|(i, (&p, &q))| p == q || (i >= n - 2 && p.abs_diff(q) == 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torcdf_core::fixtures;

    fn open(f: impl Fn(&std::path::Path) -> std::path::PathBuf) -> (tempfile::TempDir, NcFile) {
        let d = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&f(d.path())).unwrap();
        (d, nc)
    }

    #[test]
    fn candidates_per_mode() {
        let (_d, nc) = open(fixtures::vectors::write_arome_like);
        let speed = nc.meta.var("/wind_speed").unwrap();
        let both = vec!["/wind_speed".to_string(), "/wind_direction".to_string()];
        assert_eq!(candidates(&nc.meta, speed, ArrowMode::MagDir), both);
        assert_eq!(candidates(&nc.meta, speed, ArrowMode::EastNorth), both);
        assert!(candidates(&nc.meta, speed, ArrowMode::Off).is_empty());
        // Coordinates are never offered.
        let (_d, nemo) = open(fixtures::vectors::write_nemo_like);
        let thetao = nemo.meta.var("/thetao").unwrap();
        let got = candidates(&nemo.meta, thetao, ArrowMode::EastNorth);
        assert_eq!(got, vec!["/thetao".to_string(), "/uo".to_string(), "/vo".to_string()]);
    }

    #[test]
    fn staggering_tolerance() {
        assert!(staggered_or_same(&[2, 10, 20], &[2, 10, 21]));
        assert!(staggered_or_same(&[2, 11, 20], &[2, 10, 20]));
        assert!(!staggered_or_same(&[3, 10, 20], &[2, 10, 20]));
        assert!(!staggered_or_same(&[2, 10, 22], &[2, 10, 20]));
        assert!(!staggered_or_same(&[10, 20], &[2, 10, 20]));
    }

    #[test]
    fn fresh_prefills_per_form() {
        let (_d, nc) = open(fixtures::vectors::write_arome_like);
        let speed = nc.meta.var("/wind_speed").unwrap();
        let mut l = ArrowLayer::fresh(3, &nc, speed);
        assert_eq!((l.file, l.mode, l.problem.as_deref()), (3, ArrowMode::Off, None));
        l.load_prefill(ArrowMode::EastNorth);
        assert_eq!(l.picks, [None, None]);
        l.load_prefill(ArrowMode::MagDir);
        assert_eq!(l.picks, [Some("/wind_speed".into()), Some("/wind_direction".into())]);
    }
}
