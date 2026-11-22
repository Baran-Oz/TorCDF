#[cfg(feature = "fixtures")]
pub mod fixtures;
pub mod cf;
mod model;
mod open;

pub use model::*;
pub use open::NcFile;
