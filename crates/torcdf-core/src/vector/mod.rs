//! Vector fields: east/north components at cell centres.
//!
//! A [`VectorPair`] is a virtual variable made of two component variables,
//! found from the file's metadata ([`detect_pairs`]) or paired by hand
//! ([`pair_by_hand`]). Pairings last for the session; nothing is written.

mod detect;
mod resolve;

pub use detect::{can_pair, detect_pairs, pair_by_hand};
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

impl crate::NcFile {
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
