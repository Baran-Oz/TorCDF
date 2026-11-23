//! Standalone netCDF read-speed spike (design section 6.9).
//!
//! Subcommands:
//!   inspect <file>
//!   gen <out.nc> --like <file> --var uo (--target-gb N | --steps N)
//!   bench <file> --var V
//!   series <file> --var V   (torcdf-core: library vs fast point series)
//!   step <file> --var V     (torcdf-core: time to the next full slice)
//!
//! `inspect`, `gen` and `bench` are independent of torcdf-core and use `netcdf` /
//! `netcdf-sys` directly; `series` and `step` measure torcdf-core itself.

use std::error::Error;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::time::{Duration, Instant};

use netcdf::types::{FloatType, IntType, NcVariableType};
use netcdf_sys as sys;

type Res<T> = Result<T, Box<dyn Error>>;

// ---------------------------------------------------------------------------
// Pure helpers (unit tested)
// ---------------------------------------------------------------------------

/// Scale (lat, lon) together, keeping their ratio, so that
/// `nvars * time * lat * lon * elem_bytes` is approximately `target_bytes`.
fn scale_grid(
    lat0: usize,
    lon0: usize,
    time: usize,
    nvars: usize,
    elem_bytes: usize,
    target_bytes: u64,
) -> (usize, usize) {
    let base = (nvars * time * lat0 * lon0 * elem_bytes) as f64;
    let s = (target_bytes as f64 / base).sqrt();
    let lat = ((lat0 as f64 * s).round() as usize).max(1);
    let lon = ((lon0 as f64 * s).round() as usize).max(1);
    (lat, lon)
}

/// `[time, lat, lon]` of a generated file. `steps` keeps the source grid and
/// sets the time length; otherwise the grid is scaled to `target_gb` over 8760 steps.
fn gen_shape(src_shape: &[usize], steps: Option<usize>, target_gb: Option<u64>) -> [usize; 3] {
    if let Some(t) = steps {
        return [t, src_shape[1], src_shape[2]];
    }
    let time = 8760;
    let gb = target_gb.expect("gen needs --steps or --target-gb");
    let (lat, lon) = scale_grid(src_shape[1], src_shape[2], time, 2, 4, gb * 1_000_000_000);
    [time, lat, lon]
}

/// Pick at most `max_out` evenly strided columns (stride = ceil(len / max_out)).
fn thin_columns<T: Copy>(row: &[T], max_out: usize) -> Vec<T> {
    let stride = row.len().div_ceil(max_out).max(1);
    row.iter().step_by(stride).copied().collect()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------
// Raw (netcdf-sys) inspection
// ---------------------------------------------------------------------------

fn chk(code: c_int, what: &str) -> Res<()> {
    if code != 0 {
        let msg = unsafe { CStr::from_ptr(sys::nc_strerror(code)) }.to_string_lossy().into_owned();
        return Err(format!("{what}: netCDF error {code}: {msg}").into());
    }
    Ok(())
}

struct RawFile(c_int);

impl RawFile {
    fn open(path: &str) -> Res<Self> {
        let c = CString::new(path)?;
        let mut id = 0;
        chk(unsafe { sys::nc_open(c.as_ptr(), sys::NC_NOWRITE, &mut id) }, "nc_open")?;
        Ok(Self(id))
    }
}

impl Drop for RawFile {
    fn drop(&mut self) {
        unsafe { sys::nc_close(self.0) };
    }
}

#[derive(Debug, Clone)]
struct RawVarInfo {
    name: String,
    xtype: c_int,
    dim_names: Vec<String>,
    shape: Vec<usize>,
    unlimited: Vec<bool>,
    /// None = contiguous
    chunks: Option<Vec<usize>>,
    shuffle: bool,
    deflate: Option<i32>,
    filter_ids: Vec<u32>,
}

fn type_name(x: c_int) -> String {
    match x {
        sys::NC_BYTE => "i8".into(),
        sys::NC_CHAR => "char".into(),
        sys::NC_SHORT => "i16".into(),
        sys::NC_INT => "i32".into(),
        sys::NC_FLOAT => "f32".into(),
        sys::NC_DOUBLE => "f64".into(),
        sys::NC_UBYTE => "u8".into(),
        sys::NC_USHORT => "u16".into(),
        sys::NC_UINT => "u32".into(),
        sys::NC_INT64 => "i64".into(),
        sys::NC_UINT64 => "u64".into(),
        sys::NC_STRING => "string".into(),
        other => format!("type#{other}"),
    }
}

fn name_from(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

fn raw_var_info(f: &RawFile, varid: c_int) -> Res<RawVarInfo> {
    let mut name = vec![0 as c_char; sys::NC_MAX_NAME as usize + 1];
    let mut xtype = 0;
    let mut ndims = 0;
    let mut dimids = vec![0 as c_int; sys::NC_MAX_VAR_DIMS as usize];
    let mut natts = 0;
    chk(
        unsafe { sys::nc_inq_var(f.0, varid, name.as_mut_ptr(), &mut xtype, &mut ndims, dimids.as_mut_ptr(), &mut natts) },
        "nc_inq_var",
    )?;
    let mut unlim = -1;
    unsafe { sys::nc_inq_unlimdim(f.0, &mut unlim) };
    let mut dim_names = vec![];
    let mut shape = vec![];
    let mut unlimited = vec![];
    for &d in &dimids[..ndims as usize] {
        let mut dn = vec![0 as c_char; sys::NC_MAX_NAME as usize + 1];
        let mut len = 0usize;
        chk(unsafe { sys::nc_inq_dim(f.0, d, dn.as_mut_ptr(), &mut len) }, "nc_inq_dim")?;
        dim_names.push(name_from(&dn));
        shape.push(len);
        unlimited.push(d == unlim);
    }

    let mut storage = 0;
    let mut csz = vec![0usize; ndims.max(1) as usize];
    chk(unsafe { sys::nc_inq_var_chunking(f.0, varid, &mut storage, csz.as_mut_ptr()) }, "nc_inq_var_chunking")?;
    let chunks = if storage == sys::NC_CHUNKED { Some(csz[..ndims as usize].to_vec()) } else { None };

    let (mut sh, mut df, mut lvl) = (0, 0, 0);
    chk(unsafe { sys::nc_inq_var_deflate(f.0, varid, &mut sh, &mut df, &mut lvl) }, "nc_inq_var_deflate")?;

    let mut nf = 0usize;
    chk(unsafe { sys::nc_inq_var_filter_ids(f.0, varid, &mut nf, std::ptr::null_mut()) }, "nc_inq_var_filter_ids(count)")?;
    let mut ids = vec![0u32; nf];
    if nf > 0 {
        chk(unsafe { sys::nc_inq_var_filter_ids(f.0, varid, &mut nf, ids.as_mut_ptr()) }, "nc_inq_var_filter_ids")?;
    }

    Ok(RawVarInfo {
        name: name_from(&name),
        xtype,
        dim_names,
        shape,
        unlimited,
        chunks,
        shuffle: sh != 0,
        deflate: if df != 0 { Some(lvl) } else { None },
        filter_ids: ids,
    })
}

fn raw_all_vars(path: &str) -> Res<Vec<RawVarInfo>> {
    let f = RawFile::open(path)?;
    let mut n = 0;
    chk(unsafe { sys::nc_inq_nvars(f.0, &mut n) }, "nc_inq_nvars")?;
    (0..n).map(|i| raw_var_info(&f, i)).collect()
}

fn cmd_inspect(path: &str) -> Res<()> {
    println!("file: {path}");
    for v in raw_all_vars(path)? {
        let dims: Vec<String> = v
            .dim_names
            .iter()
            .zip(&v.shape)
            .zip(&v.unlimited)
            .map(|((n, l), u)| format!("{n}={l}{}", if *u { "(unlimited)" } else { "" }))
            .collect();
        let chunks = match &v.chunks {
            None => "contiguous".to_string(),
            Some(c) => format!("chunked {c:?}"),
        };
        let filters = format!(
            "shuffle={} deflate={} filter_ids={:?}",
            v.shuffle,
            v.deflate.map_or("none".to_string(), |l| l.to_string()),
            v.filter_ids
        );
        println!("  {} {} [{}] shape={:?}", type_name(v.xtype), v.name, dims.join(", "), v.shape);
        println!("      {chunks}; {filters}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// gen
// ---------------------------------------------------------------------------

fn cmd_gen(out: &str, like: &str, var: &str, steps: Option<usize>, target_gb: Option<u64>) -> Res<()> {
    if steps.is_none() && target_gb.is_none() {
        return Err("gen needs --steps N or --target-gb N".into());
    }
    let vars = raw_all_vars(like)?;
    let sibling = if var == "uo" { "vo".to_string() } else { return Err("only --var uo (with sibling vo) supported".into()) };
    let src = vars.iter().find(|v| v.name == var).ok_or("var not found in --like")?;
    if src.xtype != sys::NC_FLOAT || src.shape.len() != 3 {
        return Err(format!("expected 3-D f32 variable, got {} {:?}", type_name(src.xtype), src.shape).into());
    }
    let chunks = src.chunks.clone().ok_or("source variable is not chunked")?;
    let [time_len, lat, lon] = gen_shape(&src.shape, steps, target_gb);
    println!(
        "source {:?} chunks {:?} deflate {:?} shuffle {}; -> time={} lat={} lon={} ({:.1} GB uncompressed)",
        src.shape, chunks, src.deflate, src.shuffle, time_len, lat, lon,
        (2 * time_len * lat * lon * 4) as f64 / 1e9
    );

    // Fill value from source.
    let fill: f32 = {
        let nc = netcdf::open(like)?;
        let v = nc.variable(var).ok_or("var")?;
        v.fill_value::<f32>()?.unwrap_or(f32::from_bits(0x7149f2ca)) // NC default fill 9.96921e36
    };

    let mut file = netcdf::create(out)?;
    if src.unlimited[0] {
        file.add_unlimited_dimension(&src.dim_names[0])?;
    } else {
        file.add_dimension(&src.dim_names[0], time_len)?;
    }
    file.add_dimension(&src.dim_names[1], lat)?;
    file.add_dimension(&src.dim_names[2], lon)?;
    let dnames: Vec<&str> = src.dim_names.iter().map(String::as_str).collect();
    // Chunk shape is copied verbatim, clamped to the new dimension lengths.
    let new_chunks = [chunks[0].min(time_len), chunks[1].min(lat), chunks[2].min(lon)];
    for name in [var, sibling.as_str()] {
        let mut v = file.add_variable::<f32>(name, &dnames)?;
        v.set_chunking(&new_chunks)?;
        if let Some(level) = src.deflate {
            v.set_compression(level, src.shuffle)?;
        }
        v.set_fill_value(fill)?;
    }

    // Land mask: exactly 30 % of cells, spatially coherent.
    let mut score = vec![0f32; lat * lon];
    for i in 0..lat {
        for j in 0..lon {
            let (x, y) = (i as f32, j as f32);
            score[i * lon + j] = (x / 41.0).sin() * (y / 67.0).cos() + 0.6 * ((x + y) / 133.0).sin() + 0.3 * (y / 19.0).cos() * (x / 29.0).sin();
        }
    }
    let mut sorted = score.clone();
    let k = (sorted.len() as f64 * 0.30) as usize;
    let thresh = *sorted.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
    let land: Vec<bool> = score.iter().map(|&s| s < thresh).collect();
    drop(score);
    drop(sorted);
    println!("land fraction {:.4}", land.iter().filter(|&&l| l).count() as f64 / land.len() as f64);

    let lat_s: Vec<f64> = (0..lat).map(|i| (i as f64 / 7.0).sin()).collect();
    let block = new_chunks[0].max(1);
    let t0 = Instant::now();
    let mut buf = vec![0f32; block * lat * lon];
    let mut t = 0;
    let mut last_pct = -1i64;
    while t < time_len {
        let n = block.min(time_len - t);
        for (vi, name) in [var, sibling.as_str()].into_iter().enumerate() {
            let amp = if vi == 0 { 1.0 } else { 0.5 };
            for dt in 0..n {
                let ph = (t + dt) as f64 / 24.0;
                let (sp, cp) = (ph.sin(), ph.cos());
                let slab = &mut buf[dt * lat * lon..(dt + 1) * lat * lon];
                for (i, &ls) in lat_s.iter().enumerate().take(lat) {
                    for j in 0..lon {
                        let idx = i * lon + j;
                        slab[idx] = if land[idx] {
                            fill
                        } else {
                            let a = j as f64 / 11.0;
                            (amp * ls * (a.cos() * cp - a.sin() * sp)) as f32
                        };
                    }
                }
            }
            let mut v = file.variable_mut(name).ok_or("var")?;
            v.put_values(&buf[..n * lat * lon], (t..t + n, .., ..))?;
        }
        t += n;
        let pct = (t * 100 / time_len) as i64;
        if pct != last_pct {
            last_pct = pct;
            let el = t0.elapsed().as_secs_f64();
            println!("gen {pct}% t={t}/{time_len} elapsed {el:.0}s eta {:.0}s", el / t as f64 * (time_len - t) as f64);
        }
    }
    drop(file);
    println!("done in {:.0}s", t0.elapsed().as_secs_f64());
    Ok(())
}

// ---------------------------------------------------------------------------
// gen-curv
// ---------------------------------------------------------------------------

/// Small curvilinear test file for the map checklist: a 600 x 800 regular
/// 0.05 degree grid rotated 30 degrees about (10E, 50N), 2-D `nav_lat`/`nav_lon`
/// and `sst(time=12, y, x)`; a 40-cell-radius disc at the centre is land
/// (fill 1e20 in the coordinates, fill in the data).
fn cmd_gen_curv(out: &str) -> Res<()> {
    const FILL: f32 = 1.0e20;
    let (ny, nx, nt) = (600usize, 800usize, 12usize);
    let (s, c) = 30f64.to_radians().sin_cos();
    let mut file = netcdf::create(out)?;
    file.add_dimension("time", nt)?;
    file.add_dimension("y", ny)?;
    file.add_dimension("x", nx)?;

    let (cy, cx) = ((ny as f64 - 1.0) / 2.0, (nx as f64 - 1.0) / 2.0);
    let land: Vec<bool> = (0..ny * nx)
        .map(|k| {
            let (i, j) = ((k / nx) as f64 - cy, (k % nx) as f64 - cx);
            i * i + j * j <= 40.0 * 40.0
        })
        .collect();
    let mut lat = vec![0f32; ny * nx];
    let mut lon = vec![0f32; ny * nx];
    for k in 0..ny * nx {
        let (x, y) = (((k % nx) as f64 - cx) * 0.05, ((k / nx) as f64 - cy) * 0.05);
        lon[k] = (10.0 + x * c - y * s) as f32;
        lat[k] = (50.0 + x * s + y * c) as f32;
    }
    let mut sst = vec![0f32; nt * ny * nx];
    for t in 0..nt {
        for k in 0..ny * nx {
            sst[t * ny * nx + k] = if land[k] { FILL } else { 12.0 + (lat[k] - 50.0) * 2.0 + t as f32 * 0.5 };
        }
    }
    for k in 0..ny * nx {
        if land[k] {
            lat[k] = FILL;
            lon[k] = FILL;
        }
    }
    for (name, vals, units, std_name) in
        [("nav_lat", &lat, "degrees_north", "latitude"), ("nav_lon", &lon, "degrees_east", "longitude")]
    {
        let mut v = file.add_variable::<f32>(name, &["y", "x"])?;
        v.set_fill_value(FILL)?;
        v.put_attribute("units", units)?;
        v.put_attribute("standard_name", std_name)?;
        v.put_values(vals, ..)?;
    }
    let mut v = file.add_variable::<f32>("sst", &["time", "y", "x"])?;
    v.set_fill_value(FILL)?;
    v.put_attribute("units", "degC")?;
    v.put_attribute("coordinates", "nav_lat nav_lon")?;
    v.put_values(&sst, ..)?;
    drop(file);
    println!("wrote {out}: sst(time={nt}, y={ny}, x={nx}), land disc of 40 cells at the centre");
    Ok(())
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// (a) open + read all metadata (dims, vars, shapes, types, chunking, attributes).
fn bench_open(path: &str) -> Res<Duration> {
    let t = Instant::now();
    let f = netcdf::open(path)?;
    let mut sink = 0usize;
    for d in f.dimensions() {
        sink += d.len() + d.name().len();
    }
    for a in f.attributes() {
        sink += a.name().len();
        let _ = a.value()?;
    }
    for v in f.variables() {
        sink += v.name().len() + v.dimensions().len() + v.len();
        let _ = v.vartype();
        sink += v.chunking()?.map_or(0, |c| c.len());
        for a in v.attributes() {
            sink += a.name().len();
            let _ = a.value()?;
        }
    }
    std::hint::black_box(sink);
    Ok(t.elapsed())
}

/// (b) overview of a 2-D variable: every Nth row in full, columns thinned in Rust to <= 2048 x 1024.
fn bench_overview<T: netcdf::NcTypeDescriptor + Copy>(path: &str, var: &str) -> Res<(Duration, (usize, usize))> {
    let t = Instant::now();
    let f = netcdf::open(path)?;
    let v = f.variable(var).ok_or("var")?;
    let (rows, cols) = (v.dimensions()[0].len(), v.dimensions()[1].len());
    let row_stride = rows.div_ceil(2048).max(1);
    let mut out: Vec<T> = Vec::new();
    let mut out_rows = 0;
    let mut out_cols = 0;
    for r in (0..rows).step_by(row_stride) {
        let row: Vec<T> = v.get_values((r..r + 1, 0..cols))?;
        let thin = thin_columns(&row, 1024);
        out_cols = thin.len();
        out.extend(thin);
        out_rows += 1;
    }
    std::hint::black_box(&out);
    Ok((t.elapsed(), (out_rows, out_cols)))
}

/// (c) full 2-D slice at middle time step, (d) adjacent step right after, on the same open file.
fn bench_slices<T: netcdf::NcTypeDescriptor + Copy>(path: &str, var: &str) -> Res<(Duration, Duration)> {
    let f = netcdf::open(path)?;
    let v = f.variable(var).ok_or("var")?;
    let mid = v.dimensions()[0].len() / 2;
    let t = Instant::now();
    let a: Vec<T> = v.get_values((mid, .., ..))?;
    let c = t.elapsed();
    std::hint::black_box(&a);
    let t = Instant::now();
    let b: Vec<T> = v.get_values((mid + 1, .., ..))?;
    let d = t.elapsed();
    std::hint::black_box(&b);
    Ok((c, d))
}

/// (e) point series through the whole time dimension at the grid centre.
fn bench_series<T: netcdf::NcTypeDescriptor + Copy>(path: &str, var: &str) -> Res<(Duration, usize)> {
    let f = netcdf::open(path)?;
    let v = f.variable(var).ok_or("var")?;
    let (nt, nlat, nlon) = (v.dimensions()[0].len(), v.dimensions()[1].len(), v.dimensions()[2].len());
    let t = Instant::now();
    let s: Vec<T> = v.get_values((0..nt, nlat / 2, nlon / 2))?;
    let e = t.elapsed();
    std::hint::black_box(&s);
    Ok((e, s.len()))
}

fn report(label: &str, cold: Duration, runs: &[Duration]) {
    let med = median(runs.to_vec());
    let list: Vec<String> = runs.iter().map(|d| format!("{:.1}", ms(*d))).collect();
    println!("{label}: median {:.1} ms (runs {} ms) | cold/first-touch {:.1} ms", ms(med), list.join(", "), ms(cold));
}

fn bench_typed<T: netcdf::NcTypeDescriptor + Copy>(path: &str, var: &str, ndims: usize) -> Res<()> {
    // Run 0 is the "cold" (first touch) run, then 3 timed runs.
    let mut a = vec![];
    for _ in 0..4 {
        a.push(bench_open(path)?);
    }
    report("(a) open + metadata", a[0], &a[1..]);

    if ndims == 2 {
        let mut b = vec![];
        let mut shape = (0, 0);
        for _ in 0..4 {
            let (d, s) = bench_overview::<T>(path, var)?;
            b.push(d);
            shape = s;
        }
        report(&format!("(b) overview {}x{}", shape.0, shape.1), b[0], &b[1..]);
    } else if ndims == 3 {
        let (mut c, mut d) = (vec![], vec![]);
        for _ in 0..4 {
            let (cc, dd) = bench_slices::<T>(path, var)?;
            c.push(cc);
            d.push(dd);
        }
        report("(c) first 2-D slice (middle step)", c[0], &c[1..]);
        report("(d) adjacent step after (c)", d[0], &d[1..]);
        let mut e = vec![];
        let mut n = 0;
        for _ in 0..4 {
            let (ee, len) = bench_series::<T>(path, var)?;
            e.push(ee);
            n = len;
        }
        report(&format!("(e) point series, {n} steps"), e[0], &e[1..]);
    }
    Ok(())
}

fn cmd_bench(path: &str, var: &str) -> Res<()> {
    println!("bench {path} var={var}");
    let (vt, nd) = {
        let f = netcdf::open(path)?;
        let v = f.variable(var).ok_or("var not found")?;
        (v.vartype(), v.dimensions().len())
    };
    match vt {
        NcVariableType::Int(IntType::I16) => bench_typed::<i16>(path, var, nd),
        NcVariableType::Float(FloatType::F32) => bench_typed::<f32>(path, var, nd),
        other => Err(format!("unsupported type {other:?}").into()),
    }
}

// ---------------------------------------------------------------------------
// series / step (torcdf-core)
// ---------------------------------------------------------------------------

use std::path::Path;
use std::sync::Arc;
use torcdf_core::{NcFile, SeriesRequest, SliceRequest, SliceService};

/// Opens `path`; returns the file, the variable's full path and its shape.
fn open_core(path: &str, var: &str) -> Res<(Arc<NcFile>, String, Vec<usize>)> {
    let f = NcFile::open(Path::new(path))?;
    let v = f.meta.var(&format!("/{var}")).ok_or("var not found")?;
    let (vp, shape) = (v.path.clone(), v.shape.clone());
    if shape.len() != 3 {
        return Err(format!("expected a 3-D variable, got {shape:?}").into());
    }
    Ok((Arc::new(f), vp, shape))
}

fn quiet_service() -> SliceService {
    SliceService::new(1 << 30, 2, Arc::new(|| {}))
}

fn full_slice(var: &str, shape: &[usize], t: usize) -> SliceRequest {
    SliceRequest { var: var.into(), start: vec![t, 0, 0], count: vec![1, shape[1], shape[2]], stride: vec![1; 3] }
}

/// One pass, each part on a fresh file and service:
/// (library ms for the first 200 steps, fast first block ms, fast total ms, fast_reads).
fn series_pass(path: &str, var: &str) -> Res<(f64, f64, f64, usize)> {
    let (f, vp, shape) = open_core(path, var)?;
    let req = SeriesRequest { var: vp, point: vec![0, shape[1] / 2, shape[2] / 2], along: 0, centre: shape[0] / 2 };
    let t = Instant::now();
    let s = f.read_slice_via_library(&req.full_request(shape[0].min(200)))?;
    let lib = ms(t.elapsed());
    std::hint::black_box(&s);
    drop(f);

    let (f, _, _) = open_core(path, var)?;
    let svc = quiet_service();
    let t = Instant::now();
    let mut h = svc.request_series(&f, req);
    let (mut first, mut got) = (None, 0);
    while h.finished().is_none() {
        let b = h.take_blocks();
        if !b.is_empty() && first.is_none() {
            first = Some(ms(t.elapsed()));
        }
        got += b.iter().map(|b| b.values.len()).sum::<usize>();
        std::thread::sleep(Duration::from_micros(200));
    }
    let total = ms(t.elapsed());
    got += h.take_blocks().iter().map(|b| b.values.len()).sum::<usize>();
    if got != shape[0] {
        return Err(format!("series delivered {got} of {} steps", shape[0]).into());
    }
    Ok((lib, first.unwrap_or(total), total, f.fast_reads()))
}

fn cmd_series(path: &str, var: &str) -> Res<()> {
    let (f, vp, shape) = open_core(path, var)?;
    println!("series {path} var={var} shape={shape:?} fast_eligible={}", f.fast_eligible(&vp));
    drop(f);
    let n = shape[0];
    let m = n.min(200);
    for pass in ["first touch", "warm"] {
        let (lib, first, total, reads) = series_pass(path, var)?;
        println!(
            "{pass}: library {lib:.1} ms for {m} steps (projected {:.0} ms for {n}) | fast first block {first:.1} ms, total {total:.1} ms for {n} steps (projected {:.0} ms for 8760), fast_reads {reads}",
            lib * n as f64 / m as f64,
            total * 8760.0 / n as f64
        );
    }
    Ok(())
}

fn cmd_step(path: &str, var: &str) -> Res<()> {
    let (f, vp, shape) = open_core(path, var)?;
    let nt = shape[0];
    if nt < 12 {
        return Err("need at least 12 time steps".into());
    }
    println!("step {path} var={var} shape={shape:?}");
    // Ten different steps spread over the series; each read is of t+1.
    let steps: Vec<usize> = (0..10).map(|i| i * (nt - 2) / 10 + 1).collect();
    let mut fresh = vec![];
    for &t in &steps {
        let s = Instant::now();
        let out = f.read_slice(&full_slice(&vp, &shape, t))?;
        fresh.push(s.elapsed());
        std::hint::black_box(&out);
    }
    println!("fresh full-slice read_slice (fast path), median of 10: {:.1} ms (fast_reads {})", ms(median(fresh)), f.fast_reads());
    let svc = quiet_service();
    let mut cached = vec![];
    for &t in &steps {
        let r = full_slice(&vp, &shape, t);
        let mut warm = svc.request(&f, r.clone());
        while warm.try_take().is_none() {
            std::thread::sleep(Duration::from_micros(200));
        }
        let s = Instant::now();
        let mut h = svc.request(&f, r);
        loop {
            if let Some(out) = h.try_take() {
                std::hint::black_box(out?);
                break;
            }
        }
        cached.push(s.elapsed());
    }
    println!("cached adjacent slice, request to try_take, median of 10: {:.3} ms", ms(median(cached)));
    Ok(())
}

// ---------------------------------------------------------------------------

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => cmd_inspect(args.get(1).ok_or("usage: inspect <file>")?),
        Some("gen") => cmd_gen(
            args.get(1).ok_or("usage: gen <out.nc> --like <file> --var uo (--steps N | --target-gb N)")?,
            &arg(&args, "--like").ok_or("--like required")?,
            &arg(&args, "--var").unwrap_or_else(|| "uo".into()),
            arg(&args, "--steps").map(|v| v.parse()).transpose()?,
            arg(&args, "--target-gb").map(|v| v.parse()).transpose()?,
        ),
        Some("series") => cmd_series(
            args.get(1).ok_or("usage: series <file> --var V")?,
            &arg(&args, "--var").unwrap_or_else(|| "uo".into()),
        ),
        Some("step") => cmd_step(
            args.get(1).ok_or("usage: step <file> --var V")?,
            &arg(&args, "--var").unwrap_or_else(|| "uo".into()),
        ),
        Some("gen-curv") => cmd_gen_curv(args.get(1).ok_or("usage: gen-curv <out.nc>")?),
        Some("bench") => cmd_bench(
            args.get(1).ok_or("usage: bench <file> --var V")?,
            &arg(&args, "--var").ok_or("--var required")?,
        ),
        Some("ne-convert") => cmd_ne_convert(
            args.get(1).ok_or("usage: ne-convert <in.geojson> <out.bin>")?,
            args.get(2).ok_or("usage: ne-convert <in.geojson> <out.bin>")?,
        ),
        _ => Err("usage: torcdf-bench inspect|gen|gen-curv|bench|series|step|ne-convert ...".into()),
    }
}

// ---------------------------------------------------------------------------
// Natural Earth GeoJSON -> embedded polyline format
// ---------------------------------------------------------------------------

fn coord_pair(v: &serde_json::Value) -> Option<[f32; 2]> {
    let a = v.as_array()?;
    // Clamp: Natural Earth coordinates occasionally exceed the range by float noise.
    let lon = (a.first()?.as_f64()? as f32).clamp(-180.0, 180.0);
    let lat = (a.get(1)?.as_f64()? as f32).clamp(-90.0, 90.0);
    Some([lon, lat])
}

fn line_from(v: &serde_json::Value) -> Option<Vec<[f32; 2]>> {
    v.as_array()?.iter().map(coord_pair).collect()
}

/// Every LineString and MultiLineString geometry of a FeatureCollection.
fn geojson_lines(root: &serde_json::Value) -> Res<Vec<Vec<[f32; 2]>>> {
    let features = root["features"].as_array().ok_or("not a FeatureCollection")?;
    let mut out = Vec::new();
    for f in features {
        let g = &f["geometry"];
        match g["type"].as_str() {
            Some("LineString") => out.extend(line_from(&g["coordinates"])),
            Some("MultiLineString") => {
                for l in g["coordinates"].as_array().into_iter().flatten() {
                    out.extend(line_from(l));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn cmd_ne_convert(input: &str, output: &str) -> Res<()> {
    let root: serde_json::Value = serde_json::from_slice(&std::fs::read(input)?)?;
    let lines = geojson_lines(&root)?;
    let bytes = torcdf_core::basemap::encode(&lines);
    std::fs::write(output, &bytes)?;
    let pts: usize = lines.iter().map(Vec::len).sum();
    println!("{output}: {} polylines, {pts} points, {} bytes", lines.len(), bytes.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_keeps_ratio_and_hits_target() {
        let (lat, lon) = scale_grid(380, 1300, 8760, 2, 4, 100_000_000_000);
        let ratio0 = 380.0 / 1300.0;
        let ratio = lat as f64 / lon as f64;
        assert!((ratio - ratio0).abs() / ratio0 < 0.01);
        let bytes = (2 * 8760 * lat * lon * 4) as f64;
        assert!((bytes - 1e11).abs() / 1e11 < 0.01, "{bytes}");
    }

    #[test]
    fn steps_override_keeps_grid() {
        assert_eq!(gen_shape(&[24, 380, 1016], Some(2000), None), [2000, 380, 1016]);
    }

    #[test]
    fn thin_columns_bounds() {
        let row: Vec<u32> = (0..86400).collect();
        let t = thin_columns(&row, 1024);
        assert!(t.len() <= 1024 && t.len() > 900);
        assert_eq!(t[0], 0);
        assert_eq!(thin_columns(&[1, 2, 3], 1024), vec![1, 2, 3]);
        assert_eq!(thin_columns(&[1, 2, 3, 4], 2), vec![1, 3]);
    }

    #[test]
    fn median_of_three() {
        let d = |n| Duration::from_millis(n);
        assert_eq!(median(vec![d(5), d(1), d(3)]), d(3));
    }
}
