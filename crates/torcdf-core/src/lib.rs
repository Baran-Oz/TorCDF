#[cfg(feature = "fixtures")]
pub mod fixtures;
mod model;
mod open;

pub use model::*;
pub use open::NcFile;
