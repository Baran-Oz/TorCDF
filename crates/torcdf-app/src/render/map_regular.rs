//! GPU map layer for regular lat/lon grids: for every framebuffer pixel the
//! fragment shader inverts the map projection, looks the longitude and latitude
//! up in the grid's coordinate arrays (binary search) and colours the cell.

use std::collections::HashMap;
use std::num::NonZeroU64;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

use super::colormap::Colormap;
use super::gpu_common::{self, CameraUniform};
use super::image::{check_shape, normalisation, UploadError};

/// Handle to a grid uploaded with [`upload_regular`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapTile {
    pub id: u64,
    pub ny: usize,
    pub nx: usize,
}

/// How to draw a [`MapTile`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapParams {
    pub range: (f32, f32),
    pub colormap: Colormap,
    pub log: bool,
    pub camera: CameraUniform,
}

/// Longitudes made monotonic: each step is reduced to (-180, 180] degrees, so
/// 0-360 and -180-180 axes both come out continuous. Not yet validated.
pub(crate) fn unwrap_lon(lon: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(lon.len());
    let mut prev = None;
    for &l in lon {
        let u = match prev {
            None => l,
            Some((raw, un)) => {
                let d: f64 = l - raw;
                un + (d - 360.0 * ((d - 180.0) / 360.0).ceil())
            }
        };
        out.push(u);
        prev = Some((l, u));
    }
    out
}

/// True when `c` is finite and strictly monotonic in one direction.
pub(crate) fn strictly_monotonic(c: &[f64]) -> bool {
    c.iter().all(|v| v.is_finite())
        && (c.windows(2).all(|w| w[1] > w[0]) || c.windows(2).all(|w| w[1] < w[0]))
}

/// Start of the 360 degree longitude window for unwrapped, validated `lon`:
/// the smaller end minus half that end cell's spacing (1 degree cells for a
/// single column).
pub(crate) fn lon_window_start(lon: &[f64]) -> f64 {
    let n = lon.len();
    if n < 2 {
        return lon[0] - 0.5;
    }
    let (min, spacing) = if lon[n - 1] > lon[0] {
        (lon[0], lon[1] - lon[0])
    } else {
        (lon[n - 1], lon[n - 2] - lon[n - 1])
    };
    min - 0.5 * spacing
}

/// Uploads row-major `data` of shape `[ny, nx]` as map tile `id`, with `lat`
/// (`ny` values) and `lon` (`nx` values) the tile's own coordinates. Each axis
/// must be finite and strictly monotonic (either direction); `lon` may jump
/// across the 0/360 or dateline seam. Re-uploading an id replaces its data.
pub fn upload_regular(
    render_state: &egui_wgpu::RenderState,
    id: u64,
    data: &[f32],
    ny: usize,
    nx: usize,
    lat: &[f64],
    lon: &[f64],
) -> Result<MapTile, UploadError> {
    let (device, queue) = (&render_state.device, &render_state.queue);
    let max = device.limits().max_texture_dimension_2d.min(2048) as usize;
    check_shape(data.len(), ny, nx, max)?;
    if lat.len() != ny {
        return Err(UploadError::BadCoords("lat length differs from the row count"));
    }
    if lon.len() != nx {
        return Err(UploadError::BadCoords("lon length differs from the column count"));
    }
    if !strictly_monotonic(lat) {
        return Err(UploadError::BadCoords("lat must be finite and strictly monotonic"));
    }
    if lon.iter().any(|v| !v.is_finite()) {
        return Err(UploadError::BadCoords("lon must be finite"));
    }
    let lon = unwrap_lon(lon);
    if !strictly_monotonic(&lon) {
        return Err(UploadError::BadCoords("lon must be strictly monotonic"));
    }
    let lon_start = lon_window_start(&lon) as f32;
    let lat32: Vec<f32> = lat.iter().map(|&v| v as f32).collect();
    let lon32: Vec<f32> = lon.iter().map(|&v| v as f32).collect();

    let mut renderer = render_state.renderer.write();
    let res = renderer
        .callback_resources
        .entry::<MapResources>()
        .or_insert_with(|| MapResources::new(device, render_state.target_format));
    let reusable = res.tiles.get(&id).is_some_and(|t| (t.ny, t.nx) == (ny, nx));
    if !reusable {
        let tile = TileGpu::new(device, &res.bind_group_layout, ny, nx);
        res.tiles.insert(id, tile);
    }
    if let Some(tile) = res.tiles.get_mut(&id) {
        gpu_common::write_data(queue, &tile.data, data, ny, nx);
        gpu_common::write_data(queue, &tile.lat, &lat32, 1, ny);
        gpu_common::write_data(queue, &tile.lon, &lon32, 1, nx);
        tile.lon_start = lon_start;
    }
    Ok(MapTile { id, ny, nx })
}

/// Draws `tile` into `rect` (points) of `ui`.
/// Paint each tile id at most once per frame: its uniforms are per tile.
pub fn paint_regular(ui: &mut egui::Ui, rect: egui::Rect, tile: &MapTile, p: &MapParams) {
    let callback = MapCallback { id: tile.id, params: *p };
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, callback));
}

/// The `Params` uniform of `map_regular.wgsl`: `Camera` (32 bytes) at 0, `f32`
/// offset at 32, `f32` scale at 36, `u32` log at 40, `f32` lon window start at
/// 44; 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct MapUniform {
    camera: CameraUniform,
    offset: f32,
    scale: f32,
    log: u32,
    lon_start: f32,
}

const UNIFORM_BYTES: u64 = std::mem::size_of::<MapUniform>() as u64;
const _: () = assert!(UNIFORM_BYTES == 48);

/// Everything this module keeps in `CallbackResources`.
struct MapResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tiles: HashMap<u64, TileGpu>,
}

impl MapResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("torcdf map regular"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("colour.wgsl"),
                    include_str!("projection.wgsl"),
                    include_str!("map_regular.wgsl")
                )
                .into(),
            ),
        });
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("torcdf map regular"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(UNIFORM_BYTES),
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("torcdf map regular"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        // Mirror egui: gamma-space colour for Unorm targets, linear for sRGB.
        let fragment_entry = if target_format.is_srgb() {
            "fs_main_linear_framebuffer"
        } else {
            "fs_main_gamma_framebuffer"
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("torcdf map regular"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            // egui's pass (eframe defaults): no depth buffer, no MSAA.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment_entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, bind_group_layout, tiles: HashMap::new() }
    }
}

/// GPU objects of one tile.
struct TileGpu {
    ny: usize,
    nx: usize,
    data: wgpu::Texture,
    lat: wgpu::Texture,
    lon: wgpu::Texture,
    lut: wgpu::Texture,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Start of the longitude window of the uploaded `lon`.
    lon_start: f32,
    /// Colour map currently in `lut`; `None` until first painted.
    colormap: Option<Colormap>,
}

impl TileGpu {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, ny: usize, nx: usize) -> Self {
        // check_shape bounds both sides by the texture limit.
        let data = gpu_common::create_data_texture(device, ny, nx);
        let lat = gpu_common::create_data_texture(device, 1, ny);
        let lon = gpu_common::create_data_texture(device, 1, nx);
        let lut = gpu_common::lut_texture(device);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("torcdf map regular params"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let views: Vec<wgpu::TextureView> =
            [&data, &lut, &lat, &lon].iter().map(|t| t.create_view(&Default::default())).collect();
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }];
        entries.extend(views.iter().enumerate().map(|(i, v)| wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: wgpu::BindingResource::TextureView(v),
        }));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("torcdf map regular"),
            layout,
            entries: &entries,
        });
        Self { ny, nx, data, lat, lon, lut, uniforms, bind_group, lon_start: 0.0, colormap: None }
    }
}

/// One frame's draw of a tile.
struct MapCallback {
    id: u64,
    params: MapParams,
}

impl egui_wgpu::CallbackTrait for MapCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let tile = resources.get_mut::<MapResources>().and_then(|res| res.tiles.get_mut(&self.id));
        if let Some(tile) = tile {
            let p = &self.params;
            let (offset, scale) = normalisation(p.range, p.log);
            let u = MapUniform {
                camera: p.camera,
                offset,
                scale,
                log: u32::from(p.log),
                lon_start: tile.lon_start,
            };
            queue.write_buffer(&tile.uniforms, 0, bytemuck::bytes_of(&u));
            if tile.colormap != Some(p.colormap) {
                gpu_common::write_lut(queue, &tile.lut, &p.colormap.lut());
                tile.colormap = Some(p.colormap);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<MapResources>() else { return };
        let Some(tile) = res.tiles.get(&self.id) else { return };
        // The shader works from @builtin(position) (framebuffer pixels) and the
        // unclamped rect in the camera uniform, so the clamped viewport is fine.
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &tile.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::camera::MapCamera;
    use crate::map::projection::Projection;
    use crate::render::gpu_common::{camera_uniform, headless};
    use std::f64::consts::TAU;

    const W: u32 = 64;
    const H: u32 = 32;

    fn near(got: [u8; 4], want: [u8; 4]) -> bool {
        (0..4).all(|c| (got[c] as i32 - want[c] as i32).abs() <= 1)
    }

    /// Cell k (row-major) holds k + 0.5 and the range is (0, ncells), so cell k
    /// maps to LUT entry 256 * (k + 0.5) / ncells (never near a boundary).
    fn values(n: usize) -> Vec<f32> {
        (0..n).map(|k| k as f32 + 0.5).collect()
    }

    fn lut_of(k: usize, n: usize) -> [u8; 4] {
        Colormap::Viridis.lut()[((k as f32 + 0.5) / n as f32 * 256.0) as usize]
    }

    fn world_cam(proj: Projection) -> MapCamera {
        MapCamera { proj, centre: [0.0, 0.0], scale: W as f64 / TAU }
    }

    fn draw(rs: &egui_wgpu::RenderState, tile: &MapTile, cam: &MapCamera, n: usize) -> Vec<[u8; 4]> {
        headless::render(rs, W, H, |ui, rect| {
            let p = MapParams {
                range: (0.0, n as f32),
                colormap: Colormap::Viridis,
                log: false,
                camera: camera_uniform(cam, rect, 1.0),
            };
            paint_regular(ui, rect, tile, &p);
        })
    }

    /// Pixel under (lon, lat) for `cam`, read from a rendered frame.
    fn at(px: &[[u8; 4]], cam: &MapCamera, lon: f64, lat: f64) -> [u8; 4] {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(W as f32, H as f32));
        let w = cam.proj.forward(lon, lat).expect("projects");
        let p = cam.world_to_screen(w, rect);
        px[(p.y.floor() as u32 * W + p.x.floor() as u32) as usize]
    }

    #[test]
    fn equirect_global_grid_pixels() {
        let rs = headless::render_state();
        let tile =
            upload_regular(&rs, 1, &values(8), 2, 4, &[-45.0, 45.0], &[-135.0, -45.0, 45.0, 135.0])
                .expect("upload");
        let cam = world_cam(Projection::Equirectangular);
        let px = draw(&rs, &tile, &cam, 8);
        let got = at(&px, &cam, 45.0, 45.0);
        assert!(near(got, lut_of(1 * 4 + 2, 8)), "{got:?} vs {:?}", lut_of(6, 8));
        let got = at(&px, &cam, -135.0, -45.0);
        assert!(near(got, lut_of(0, 8)), "{got:?}");
    }

    #[test]
    fn wraps_0_360_data_across_greenwich() {
        let rs = headless::render_state();
        // Two identical rows (a one-row grid would be only 1 degree tall).
        let v = [0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5];
        let tile =
            upload_regular(&rs, 2, &v, 2, 4, &[-45.0, 45.0], &[0.0, 90.0, 180.0, 270.0]).expect("upload");
        let cam = world_cam(Projection::Equirectangular);
        let px = draw(&rs, &tile, &cam, 4);
        let got = at(&px, &cam, -30.0, 0.0);
        assert!(near(got, lut_of(0, 4)), "lon -30: {got:?}");
        let got = at(&px, &cam, -60.0, 0.0);
        assert!(near(got, lut_of(3, 4)), "lon -60: {got:?}");
    }

    #[test]
    fn decreasing_lat_draws_north_up() {
        let rs = headless::render_state();
        let v = [0.5, 0.5, 1.5, 1.5];
        let tile =
            upload_regular(&rs, 3, &v, 2, 2, &[45.0, -45.0], &[-90.0, 90.0]).expect("upload");
        let cam = world_cam(Projection::Equirectangular);
        let px = draw(&rs, &tile, &cam, 2);
        assert!(near(at(&px, &cam, 0.0, 45.0), lut_of(0, 2)));
        assert!(near(at(&px, &cam, 0.0, -45.0), lut_of(1, 2)));
    }

    #[test]
    fn regional_grid_transparent_outside() {
        let rs = headless::render_state();
        let tile =
            upload_regular(&rs, 4, &values(4), 2, 2, &[40.0, 50.0], &[10.0, 20.0]).expect("upload");
        let cam = world_cam(Projection::Equirectangular);
        let px = draw(&rs, &tile, &cam, 4);
        assert_eq!(at(&px, &cam, 100.0, 0.0)[3], 0);
        assert_eq!(at(&px, &cam, 15.0, 45.0)[3], 255);
    }

    #[test]
    fn nan_cell_transparent() {
        let rs = headless::render_state();
        let mut v = values(4);
        v[3] = f32::NAN;
        let tile = upload_regular(&rs, 5, &v, 2, 2, &[-45.0, 45.0], &[-90.0, 90.0]).expect("upload");
        let cam = world_cam(Projection::Equirectangular);
        let px = draw(&rs, &tile, &cam, 4);
        assert_eq!(at(&px, &cam, 90.0, 45.0)[3], 0);
        assert_eq!(at(&px, &cam, -90.0, -45.0)[3], 255);
    }

    #[test]
    fn polar_and_mercator_pixels() {
        let rs = headless::render_state();
        // Polar: row 0 is lat 0, row 1 is lat 85; every column of a row is equal.
        let v = [0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5];
        let tile =
            upload_regular(&rs, 6, &v, 2, 4, &[0.0, 85.0], &[0.0, 90.0, 180.0, 270.0]).expect("upload");
        let cam = MapCamera { proj: Projection::NorthPolarStereo, centre: [0.0, 0.0], scale: 8.0 };
        let px = draw(&rs, &tile, &cam, 2);
        let got = px[(16 * W + 32) as usize];
        assert!(near(got, lut_of(1, 2)), "pole pixel {got:?} vs {:?}", lut_of(1, 2));
        // Outside the disc (rho > 2): transparent.
        assert_eq!(px[(16 * W + 63) as usize][3], 0);

        let tile = upload_regular(
            &rs,
            7,
            &values(8),
            2,
            4,
            &[-45.0, 45.0],
            &[-135.0, -45.0, 45.0, 135.0],
        )
        .expect("upload");
        let cam = world_cam(Projection::Mercator);
        let px = draw(&rs, &tile, &cam, 8);
        let got = at(&px, &cam, 45.0, 45.0);
        assert!(near(got, lut_of(6, 8)), "mercator {got:?} vs {:?}", lut_of(6, 8));
    }

    #[test]
    fn bad_coords_rejected() {
        let rs = headless::render_state();
        let v = values(4);
        assert!(matches!(
            upload_regular(&rs, 8, &v, 2, 2, &[0.0, 0.0], &[0.0, 1.0]),
            Err(UploadError::BadCoords(_))
        ));
        assert!(matches!(
            upload_regular(&rs, 8, &v, 2, 2, &[0.0, 1.0], &[0.0, f64::NAN]),
            Err(UploadError::BadCoords(_))
        ));
        assert!(matches!(
            upload_regular(&rs, 8, &v, 2, 2, &[0.0, 1.0, 2.0], &[0.0, 1.0]),
            Err(UploadError::BadCoords(_))
        ));
    }
}
