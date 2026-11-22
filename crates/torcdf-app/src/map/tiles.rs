//! Which part of a grid a GPU tile holds.

use crate::view::defaults::{overview_stride, OVERVIEW_TARGET};

/// A strided window of the plotted dimensions: tile texel `(ty, tx)` holds the
/// full-resolution cell `(start_y + ty * stride_y, start_x + tx * stride_x)`.
/// All arrays are `[y, x]`, in full-resolution indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileSpec {
    pub start: [usize; 2],
    pub count: [usize; 2],
    pub stride: [usize; 2],
}

impl TileSpec {
    /// The whole `ny` x `nx` grid thinned to at most `OVERVIEW_TARGET` per side.
    pub fn overview(ny: usize, nx: usize) -> TileSpec {
        TileSpec {
            start: [0, 0],
            count: [ny, nx],
            stride: [overview_stride(ny, OVERVIEW_TARGET), overview_stride(nx, OVERVIEW_TARGET)],
        }
    }

    /// Texture size `[ny, nx]` of the tile.
    pub fn shape(&self) -> [usize; 2] {
        [0, 1].map(|a| self.count[a].div_ceil(self.stride[a].max(1)))
    }

    /// Tile texel of full-resolution cell `(iy, ix)`, only for an exact hit
    /// (on the stride lattice and inside the tile).
    pub fn texel_of(&self, iy: usize, ix: usize) -> Option<[usize; 2]> {
        let shape = self.shape();
        let mut out = [0; 2];
        for (a, i) in [iy, ix].into_iter().enumerate() {
            let d = i.checked_sub(self.start[a])?;
            let s = self.stride[a].max(1);
            if d % s != 0 || d / s >= shape[a] {
                return None;
            }
            out[a] = d / s;
        }
        Some(out)
    }

    /// Tile texel nearest to full-resolution cell `(iy, ix)`, if the cell is
    /// inside the tile's extent.
    pub fn nearest_texel(&self, iy: usize, ix: usize) -> Option<[usize; 2]> {
        let shape = self.shape();
        let mut out = [0; 2];
        for (a, i) in [iy, ix].into_iter().enumerate() {
            let d = i.checked_sub(self.start[a])?;
            let s = self.stride[a].max(1);
            if d >= self.count[a] {
                return None;
            }
            out[a] = ((d + s / 2) / s).min(shape[a] - 1);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_spec_overview_and_texels() {
        let t = TileSpec::overview(43200, 86400);
        assert_eq!(t.stride, [22, 43]);
        assert_eq!(t.texel_of(22, 43), Some([1, 1]));
        assert_eq!(t.texel_of(23, 43), None);
        assert_eq!(t.nearest_texel(23, 44), Some([1, 1]));
        assert_eq!(t.nearest_texel(43200, 0), None);
        assert!(t.shape().iter().all(|&n| n <= OVERVIEW_TARGET));
    }
}
