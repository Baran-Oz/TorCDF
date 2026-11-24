//! CF-convention interpretation of variable attributes.

mod coords;
mod packing;
mod time;

pub use coords::{Axes, Geo, detect_axes, detect_geo, wrap_lon};
pub use packing::Unpack;
pub use time::{Calendar, CfDateTime, MAX_TICKS, TICK_STEPS, TickStep, TimeUnits};
