#[cfg(feature = "fixtures")]
pub mod fixtures;
pub mod basemap;
pub mod cf;
mod fast;
mod model;
mod open;
mod read;
mod series;
mod service;
mod vector;

pub use model::*;
pub use open::NcFile;
pub use read::{Slice, SliceRequest};
pub use series::{SeriesBlock, SeriesHandle, SeriesRequest, series_blocks};
pub use service::{DEFAULT_WORKERS, RequestHandle, SliceService, default_cache_budget_bytes};
pub use vector::{
    AngleSource, Frame, RotateChoice, VectorPair, VectorParts, can_pair, detect_pairs, pair_by_hand,
};
