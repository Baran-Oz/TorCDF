//! Vector fields: east/north components at cell centres.
//!
//! A [`VectorPair`] is a virtual variable made of two component variables,
//! found from the file's metadata ([`detect_pairs`]) or paired by hand
//! ([`pair_by_hand`]). Pairings last for the session; nothing is written.

mod detect;
mod magdir;
mod resolve;

pub use detect::{can_pair, detect_pairs, pair_by_hand};
pub use magdir::{
    AngleRef, AngleUnit, DirConvention, Guessed, Towards, can_pair_magdir, convention_of, en_to_magdir,
    magdir_to_en, pair_magdir_by_hand,
};
pub use resolve::{Basis, Resolved, Side, angle_from_latlon, stagger_side};
pub(crate) use resolve::ResolvedCache;

use crate::{Attr, AttrValue, DType, Storage, VarInfo};
use std::sync::Arc;

/// East and north components of a vector field, row-major like the
/// [`Slice`](crate::Slice) they belong to; `NaN` marks missing data.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorParts {
    pub east: Vec<f32>,
    pub north: Vec<f32>,
}

/// What the file says about the components' directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Frame {
    /// Eastward/northward standard names: already east/north.
    Geographic,
    /// `x_`/`y_`/`grid_` standard names or a rotated-pole grid mapping:
    /// along the model grid.
    GridRelative,
    /// Nothing either way.
    Undeclared,
}

/// Where the grid's rotation angle comes from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AngleSource {
    /// A variable of angles in radians (ROMS `angle`).
    Angle(String),
    /// Cosine and sine variables (WRF `COSALPHA`, `SINALPHA`).
    CosSin(String, String),
    /// Computed from the centre grid's 2-D latitude/longitude.
    FromLatLon,
}

/// The user's rotation choice for a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RotateChoice {
    #[default]
    Auto,
    On,
    Off,
}

/// How a pair's two variables are to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PairForm {
    /// East and north components.
    EastNorth,
    /// Magnitude (stored in `east`) and direction (stored in `north`),
    /// converted to east/north when read.
    MagDir(DirConvention),
}

/// A vector variable made of two component variables.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorPair {
    /// `"<group path>/<east name>+<north name>"`, plus `"#on"` / `"#off"`
    /// when `rotate` is not `Auto`.
    pub path: String,
    /// Display name: `"<stem> (<east name>, <north name>)"`.
    pub name: String,
    pub units: Option<String>,
    /// Component variable paths (x-like, y-like).
    pub east: String,
    pub north: String,
    /// Dimensions and shape of the centre grid the pair is read on.
    pub centre_dims: Vec<String>,
    pub centre_shape: Vec<usize>,
    /// Positions of the centre's (y, x) dims in `centre_dims`.
    pub horiz: [usize; 2],
    /// Centre latitude/longitude variables: those of a scalar variable on
    /// the centre dims, or the components' own when they are unstaggered;
    /// `None` when they must be averaged from the components' coordinates.
    pub centre_lat: Option<String>,
    pub centre_lon: Option<String>,
    pub frame: Frame,
    pub angle: AngleSource,
    pub rotate: RotateChoice,
    pub by_hand: bool,
    pub form: PairForm,
}

impl VectorPair {
    /// The virtual variable on the centre grid.
    pub fn var_info(&self) -> VarInfo {
        let text = |name: &str, v: &str| Attr { name: name.into(), value: AttrValue::Text(v.into()) };
        let mut attrs = Vec::new();
        if let Some(u) = &self.units {
            attrs.push(text("units", u));
        }
        attrs.push(text("long_name", &self.name));
        if let (Some(lat), Some(lon)) = (&self.centre_lat, &self.centre_lon) {
            attrs.push(text("coordinates", &format!("{} {}", leaf(lat), leaf(lon))));
        }
        VarInfo {
            name: self.name.clone(),
            path: self.path.clone(),
            dims: self.centre_dims.clone(),
            shape: self.centre_shape.clone(),
            dtype: DType::F32,
            attrs,
            storage: Storage::Unknown,
        }
    }

    /// The magnitude variable of a magnitude/direction pair (the `east` field).
    pub fn magnitude(&self) -> &str {
        &self.east
    }

    /// The direction variable of a magnitude/direction pair (the `north` field).
    pub fn direction(&self) -> &str {
        &self.north
    }

    /// This magnitude/direction pair read with convention `c`; the path tag
    /// changes with it, the rotation tag stays last. Other pairs are returned
    /// unchanged.
    pub fn with_convention(&self, c: DirConvention) -> VectorPair {
        if !matches!(self.form, PairForm::MagDir(_)) {
            return self.clone();
        }
        let (base, rot) = match self.rotate {
            RotateChoice::Auto => (self.path.as_str(), ""),
            RotateChoice::On => (self.path.strip_suffix("#on").unwrap_or(&self.path), "#on"),
            RotateChoice::Off => (self.path.strip_suffix("#off").unwrap_or(&self.path), "#off"),
        };
        let base = base.rsplit_once('~').map_or(base, |(b, _)| b);
        VectorPair { path: format!("{base}~{}{rot}", c.tag()), form: PairForm::MagDir(c), ..self.clone() }
    }

    /// This pair with rotation choice `r`; the path tag changes with it.
    pub fn with_rotate(&self, r: RotateChoice) -> VectorPair {
        let base = match self.rotate {
            RotateChoice::Auto => &self.path,
            RotateChoice::On => self.path.strip_suffix("#on").unwrap_or(&self.path),
            RotateChoice::Off => self.path.strip_suffix("#off").unwrap_or(&self.path),
        };
        let tag = match r {
            RotateChoice::Auto => "",
            RotateChoice::On => "#on",
            RotateChoice::Off => "#off",
        };
        VectorPair { path: format!("{base}{tag}"), rotate: r, ..self.clone() }
    }
}

/// The last segment of a variable path.
fn leaf(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// The virtual path of an arrow layer: `var`'s values with `pair`'s arrows.
pub fn layer_path(var: &str, pair: &str) -> String {
    format!("{var}|{pair}")
}

/// Splits an arrow layer path into `(var, pair)` at the first `|`.
pub fn split_layer(path: &str) -> Option<(&str, &str)> {
    path.split_once('|')
}

impl crate::NcFile {
    /// Whether `var` is a numeric variable on exactly `pair`'s centre grid:
    /// the same dims and shape, and the same horizontal coordinates (so a
    /// u- or v-grid variable of identical shape does not fit).
    pub fn layer_fits(&self, var: &str, pair: &VectorPair) -> bool {
        let Ok(v) = self.numeric_var(var) else { return false };
        if v.dims != pair.centre_dims || v.shape != pair.centre_shape {
            return false;
        }
        match (&pair.centre_lat, &pair.centre_lon) {
            (Some(lat), Some(lon)) => match crate::cf::detect_geo(&self.meta, v) {
                crate::cf::Geo::Regular { lat_var, lon_var, .. }
                | crate::cf::Geo::Curvilinear { lat_var, lon_var, .. } => lat_var == *lat && lon_var == *lon,
                crate::cf::Geo::None => false,
            },
            _ => !(matches!(pair.form, PairForm::EastNorth) && (var == pair.east || var == pair.north)),
        }
    }

    fn pairs_read(&self) -> std::sync::RwLockReadGuard<'_, Vec<Arc<VectorPair>>> {
        self.pairs.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Registered pairs: detected ones in detection order, then hand pairs
    /// in the order they were added.
    pub fn pairs(&self) -> Vec<Arc<VectorPair>> {
        self.pairs_read().clone()
    }

    /// The registered pair at `path`.
    pub fn pair(&self, path: &str) -> Option<Arc<VectorPair>> {
        self.pairs_read().iter().find(|p| p.path == path).cloned()
    }

    /// Registers `p` for this session; a pair already at its path is
    /// replaced in place.
    pub fn add_pair(&self, p: VectorPair) -> Arc<VectorPair> {
        let p = Arc::new(p);
        let mut pairs = self.pairs.write().unwrap_or_else(|e| e.into_inner());
        match pairs.iter_mut().find(|q| q.path == p.path) {
            Some(q) => *q = p.clone(),
            None => pairs.push(p.clone()),
        }
        drop(pairs);
        self.forget_resolved(&p.path);
        p
    }

    /// Unregisters the pair at `path` (no effect if there is none).
    pub fn remove_pair(&self, path: &str) {
        self.pairs.write().unwrap_or_else(|e| e.into_inner()).retain(|p| p.path != path);
        self.forget_resolved(path);
    }
}
