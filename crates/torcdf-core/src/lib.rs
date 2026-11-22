#[cfg(feature = "fixtures")]
pub mod fixtures;
pub mod cf;
mod model;
mod open;
mod read;

pub use model::*;
pub use open::NcFile;
pub use read::{Slice, SliceRequest};
