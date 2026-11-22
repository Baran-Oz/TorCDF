#[cfg(feature = "fixtures")]
pub mod fixtures;
pub mod basemap;
pub mod cf;
mod model;
mod open;
mod read;
mod service;

pub use model::*;
pub use open::NcFile;
pub use read::{Slice, SliceRequest};
pub use service::{DEFAULT_WORKERS, RequestHandle, SliceService, default_cache_budget_bytes};
