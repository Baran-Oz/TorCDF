//! Embedded Natural Earth 1:50m coastlines and country borders (public domain).
//! See `assets/NATURAL_EARTH.md` for provenance and regeneration.

use std::sync::OnceLock;

pub use torcdf_core::basemap::Polyline;

static COASTLINES: OnceLock<Vec<Polyline>> = OnceLock::new();
static BORDERS: OnceLock<Vec<Polyline>> = OnceLock::new();

fn load(bytes: &[u8]) -> Vec<Polyline> {
    // The assets are compiled in; a decode failure is a build-time data bug.
    // Degrade to "no lines" rather than panic in the UI.
    torcdf_core::basemap::decode(bytes).unwrap_or_default()
}

pub fn coastlines() -> &'static [Polyline] {
    COASTLINES.get_or_init(|| load(include_bytes!("../../assets/ne_50m_coastline.bin")))
}

pub fn borders() -> &'static [Polyline] {
    BORDERS.get_or_init(|| load(include_bytes!("../../assets/ne_50m_borders.bin")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_coastlines_are_sane() {
        let c = coastlines();
        assert!(c.len() > 1000);
        assert!(c.iter().flatten().all(|p| (-180.0..=180.0).contains(&p[0]) && (-90.0..=90.0).contains(&p[1])));
    }

    #[test]
    fn embedded_borders_load() {
        assert!(borders().len() > 100);
    }
}
