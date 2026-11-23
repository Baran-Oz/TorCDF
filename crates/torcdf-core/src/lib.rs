#[cfg(feature = "fixtures")]
pub mod fixtures;
pub mod basemap;
pub mod cf;
mod fast;
mod h5meta;
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
    AngleRef, AngleSource, AngleUnit, Basis, DirConvention, Frame, Guessed, PairForm, Resolved, RotateChoice,
    Side, Towards, VectorPair, VectorParts, angle_from_latlon,
    can_pair, can_pair_magdir, convention_of, detect_pairs, en_to_magdir, magdir_to_en, pair_by_hand,
    pair_magdir_by_hand, stagger_side, layer_path, split_layer,
};
