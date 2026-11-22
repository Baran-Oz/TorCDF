//! CPU mesh for curvilinear (2-D lat/lon) grids: a coarse triangle mesh over
//! the tile's texel centres, projected once per projection. The GPU
//! interpolates the tile texel across each triangle (see `map_curv`).

use std::f64::consts::{PI, TAU};

use super::projection::Projection;
use super::tiles::TileSpec;
use super::valid_lonlat as valid;

/// Largest mesh side (vertices) the app builds: at most 257 x 257 vertices.
pub const MESH_MAX_SIDE: usize = 257;

/// Mesh in geographic coordinates. `lonlat` is `[lon, lat]` degrees, `texel`
/// is the vertex's tile-texture coordinate `[x, y]` (column, row; the texel
/// centre, so integral), vertices are row-major over the mesh lattice.
/// Vertices with missing coordinates are kept (NaN) but no triangle uses them.
pub struct CurvMesh {
    pub lonlat: Vec<[f64; 2]>,
    pub texel: Vec<[f32; 2]>,
    pub tris: Vec<[u32; 3]>,
}

/// Tile texel indices of the mesh lattice along an axis of `n` texels: every
/// `k`-th texel with the smallest `k` giving at most `max_side` vertices, plus
/// the last texel.
fn lattice(n: usize, max_side: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let max_side = max_side.max(2);
    let k = (1..).find(|&k| (n - 1).div_ceil(k) + 1 <= max_side).unwrap_or(1);
    let mut v: Vec<usize> = (0..n - 1).step_by(k).collect();
    v.push(n - 1);
    v
}

/// Builds the mesh of `tile` from the full-resolution row-major `lat`/`lon`
/// (`full = [ny, nx]`). A tile reaching outside the arrays gives an empty mesh.
pub fn build_mesh(lat: &[f64], lon: &[f64], full: [usize; 2], tile: &TileSpec, max_side: usize) -> CurvMesh {
    let empty = || CurvMesh { lonlat: Vec::new(), texel: Vec::new(), tris: Vec::new() };
    let [ty, tx] = tile.shape();
    let in_bounds = ty > 0
        && tx > 0
        && lat.len() >= full[0].saturating_mul(full[1])
        && lon.len() >= full[0].saturating_mul(full[1])
        && tile.start[0] + (ty - 1) * tile.stride[0] < full[0]
        && tile.start[1] + (tx - 1) * tile.stride[1] < full[1];
    if !in_bounds {
        return empty();
    }
    let (rows, cols) = (lattice(ty, max_side), lattice(tx, max_side));
    let mut mesh = empty();
    mesh.lonlat.reserve(rows.len() * cols.len());
    mesh.texel.reserve(rows.len() * cols.len());
    for &r in &rows {
        let fy = tile.start[0] + r * tile.stride[0];
        for &c in &cols {
            let fx = tile.start[1] + c * tile.stride[1];
            let i = fy * full[1] + fx;
            mesh.lonlat.push([lon[i], lat[i]]);
            mesh.texel.push([c as f32, r as f32]);
        }
    }
    let ok: Vec<bool> = mesh.lonlat.iter().map(|p| valid(p[0], p[1])).collect();
    let w = cols.len();
    for r in 0..rows.len().saturating_sub(1) {
        for c in 0..w.saturating_sub(1) {
            let a = (r * w + c) as u32;
            let (b, d) = (a + 1, a + w as u32);
            let e = d + 1;
            for t in [[a, b, d], [b, e, d]] {
                if t.iter().all(|&v| ok[v as usize]) {
                    mesh.tris.push(t);
                }
            }
        }
    }
    mesh
}

/// A projected mesh vertex: world position and tile texel `[x, y]`.
/// Matches `map_curv.wgsl` (16 bytes, vertex attributes 0 and 1).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 2],
    pub texel: [f32; 2],
}

/// Projects `mesh` to world units: vertices and triangle indices. Triangles
/// with a vertex the projection rejects are dropped. For cylindrical
/// projections a triangle crossing the seam is kept: its other vertices are
/// shifted by 2*pi to lie within pi of its first vertex (such triangles get
/// their own duplicated vertices); the renderer's wrap copies cover the rest.
pub fn project_mesh(proj: Projection, mesh: &CurvMesh) -> (Vec<MeshVertex>, Vec<u32>) {
    let world: Vec<Option<[f64; 2]>> =
        mesh.lonlat.iter().map(|p| proj.forward(p[0], p[1])).collect();
    let vertex = |w: [f64; 2], i: usize| MeshVertex {
        pos: [w[0] as f32, w[1] as f32],
        texel: mesh.texel[i],
    };
    let mut verts: Vec<MeshVertex> =
        world.iter().enumerate().map(|(i, w)| vertex(w.unwrap_or([0.0; 2]), i)).collect();
    let mut indices = Vec::with_capacity(mesh.tris.len() * 3);
    for tri in &mesh.tris {
        let ids = tri.map(|v| v as usize);
        let (Some(p0), Some(p1), Some(p2)) = (world[ids[0]], world[ids[1]], world[ids[2]]) else {
            continue;
        };
        if !proj.wraps_x() {
            indices.extend_from_slice(tri);
            continue;
        }
        let shift = |p: [f64; 2]| {
            let d = p[0] - p0[0];
            if d > PI {
                -TAU
            } else if d < -PI {
                TAU
            } else {
                0.0
            }
        };
        let shifts = [0.0, shift(p1), shift(p2)];
        if shifts == [0.0; 3] {
            indices.extend_from_slice(tri);
            continue;
        }
        for (k, (p, s)) in [p0, p1, p2].into_iter().zip(shifts).enumerate() {
            indices.push(verts.len() as u32);
            verts.push(vertex([p[0] + s, p[1]], ids[k]));
        }
    }
    (verts, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_tile(ny: usize, nx: usize) -> TileSpec {
        TileSpec { start: [0, 0], count: [ny, nx], stride: [1, 1] }
    }

    #[test]
    fn mesh_vertex_count_bounded() {
        let n = 1000;
        let lat: Vec<f64> = (0..n * n).map(|i| (i / n) as f64 * 0.05).collect();
        let lon: Vec<f64> = (0..n * n).map(|i| (i % n) as f64 * 0.05).collect();
        let m = build_mesh(&lat, &lon, [n, n], &full_tile(n, n), 257);
        assert!(m.lonlat.len() <= 257 * 257, "{}", m.lonlat.len());
        assert_eq!(m.lonlat.len(), m.texel.len());
        assert!(m.texel.contains(&[0.0, 0.0]));
        assert!(m.texel.contains(&[999.0, 999.0]));
        assert!(!m.tris.is_empty());
    }

    #[test]
    fn mesh_skips_missing_coords() {
        let mut lat: Vec<f64> = (0..9).map(|i| (i / 3) as f64).collect();
        let lon: Vec<f64> = (0..9).map(|i| (i % 3) as f64).collect();
        lat[4] = f64::NAN;
        let m = build_mesh(&lat, &lon, [3, 3], &full_tile(3, 3), 257);
        assert_eq!(m.lonlat.len(), 9);
        assert!(m.tris.iter().all(|t| !t.contains(&4)));
        assert_eq!(m.tris.len(), 2, "only the corner triangles away from the centre");
        // Fill values and |lat| > 90 are missing too.
        let lat = vec![0.0, 1e20, 0.0, 1.0];
        let lon = vec![0.0, 1.0, 1e20, 1.0];
        assert!(build_mesh(&lat, &lon, [2, 2], &full_tile(2, 2), 257).tris.is_empty());
    }

    #[test]
    fn bad_tile_gives_empty_mesh() {
        let v = vec![0.0; 4];
        let t = TileSpec { start: [0, 0], count: [3, 3], stride: [1, 1] };
        assert!(build_mesh(&v, &v, [2, 2], &t, 257).lonlat.is_empty());
        assert!(build_mesh(&v, &v[..3], [2, 2], &full_tile(2, 2), 257).lonlat.is_empty());
    }

    #[test]
    fn seam_triangles_are_kept() {
        let lon = [179.0, -179.0, 179.0, -179.0];
        let lat = [0.0, 0.0, 1.0, 1.0];
        let m = build_mesh(&lat, &lon, [2, 2], &full_tile(2, 2), 257);
        let (verts, idx) = project_mesh(Projection::Equirectangular, &m);
        assert_eq!(idx.len(), 6, "two triangles kept");
        for t in idx.chunks(3) {
            let xs: Vec<f32> = t.iter().map(|&i| verts[i as usize].pos[0]).collect();
            let span = xs.iter().cloned().fold(f32::MIN, f32::max)
                - xs.iter().cloned().fold(f32::MAX, f32::min);
            assert!(span < 0.1, "{xs:?}");
        }
    }

    #[test]
    fn polar_drops_other_hemisphere() {
        let lat = [10.0, 10.0, -10.0, -10.0];
        let lon = [0.0, 90.0, 0.0, 90.0];
        let m = build_mesh(&lat, &lon, [2, 2], &full_tile(2, 2), 257);
        let (_, idx) = project_mesh(Projection::NorthPolarStereo, &m);
        assert!(idx.is_empty());
    }
}
