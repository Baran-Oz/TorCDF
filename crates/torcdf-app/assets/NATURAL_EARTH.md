# Natural Earth data

Coastlines and country borders embedded in the app (offline). Natural Earth is
public domain. Made with Natural Earth (https://www.naturalearthdata.com/).

## Sources (1:50m, GeoJSON)

- https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson
  (SHA-256 271F1C4C1908312BAC6B29D158EA1356544BEAFC129F260005300913AA5EA283) -> `ne_50m_coastline.bin`
- https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson
  (SHA-256 2FAAC4F6B34386F3D21B6E018CF151F241F00E5C936D44DD17D7D9BFB147FA48) -> `ne_50m_borders.bin`

Version: nvkelso/natural-earth-vector `master` at commit
`ca96624a56bd078437bca8184e78163e5039ad19`, downloaded 2026-11-22.

## Format

`b"TCNE"`, u32 version (1), u32 n, n x u32 point counts, then all points as
(f32 lon, f32 lat); little-endian. See `torcdf-core/src/basemap.rs`.

Conversion takes every `LineString` and `MultiLineString` geometry. Coordinates
are clamped to lon [-180, 180], lat [-90, 90] (guards against float noise).

## Regenerate

From the repository root (the GeoJSON files in `target/ne-src/` are not committed):

```
curl -L -o target/ne-src/ne_50m_coastline.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson
curl -L -o target/ne-src/ne_50m_admin_0_boundary_lines_land.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson
cargo run -p torcdf-bench -- ne-convert target/ne-src/ne_50m_coastline.geojson crates/torcdf-app/assets/ne_50m_coastline.bin
cargo run -p torcdf-bench -- ne-convert target/ne-src/ne_50m_admin_0_boundary_lines_land.geojson crates/torcdf-app/assets/ne_50m_borders.bin
```
