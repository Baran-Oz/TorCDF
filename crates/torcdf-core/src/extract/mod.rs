//! Extract: choose variables and index ranges, then save them as a new
//! netCDF-4 file that keeps the original's attributes and compression.

mod plan;
pub mod range;
mod write;

pub use write::{ExtractJob, history_line, write_extract};

use crate::calc::Derived;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A dimension: the group it is defined in, and its name.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DimKey {
    pub group: String,
    pub name: String,
}

/// What to extract: ticked variable paths (file or calculated) and an
/// index range (start, count) per dimension; a missing dimension is whole.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Selection {
    pub vars: Vec<String>,
    pub ranges: BTreeMap<DimKey, (usize, usize)>,
}

/// One variable to write, with the indices it is cropped to.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanVar {
    pub path: String,
    /// The calculated variable as it was when planned (`None`: a file variable).
    pub derived: Option<Arc<Derived>>,
    pub dims: Vec<DimKey>,
    pub start: Vec<usize>,
    pub count: Vec<usize>,
}

/// Everything a save writes, worked out from metadata only.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Ticked variables, then the coordinate, `coordinates`, `bounds` and
    /// `grid_mapping` variables they pull in.
    pub vars: Vec<PlanVar>,
    /// Written dimensions: new length and whether unlimited.
    pub dims: Vec<(DimKey, usize, bool)>,
    /// "name (why)" for what cannot be copied.
    pub skipped: Vec<String>,
    /// Stored size before compression.
    pub bytes: u64,
}
