//! Vector pair detection from file metadata (spec 7.2).
//!
//! Only the file's own conventions are used (standard names, common
//! component names, units, dimensions, grid mappings, angle variables),
//! never a list of models. Each variable is in at most one detected pair.

use super::magdir::magdir_pair;
use super::{AngleSource, Frame, PairForm, RotateChoice, VectorPair};
use crate::cf::{Geo, detect_geo};
use crate::{Dataset, Group, VarInfo};
use std::collections::HashSet;

/// Every pair the metadata declares, group by group (depth first): first
/// the standard-name pairs, then the common-name pairs, each in the order
/// of their east component.
pub fn detect_pairs(ds: &Dataset) -> Vec<VectorPair> {
    fn walk(ds: &Dataset, g: &Group, coords: &Coords, out: &mut Vec<VectorPair>) {
        detect_in_group(ds, g, coords, out);
        for sub in &g.groups {
            walk(ds, sub, coords, out);
        }
    }
    let coords = Coords::of(ds);
    let mut out = Vec::new();
    walk(ds, &ds.root, &coords, &mut out);
    out
}

/// Whether `a` (east) and `b` (north) can be paired by hand: two distinct
/// numeric variables of one group with the same or staggered shape.
pub fn can_pair(ds: &Dataset, a: &str, b: &str) -> bool {
    hand_check(ds, a, b).is_ok()
}

/// A pair of `east` and `north` chosen by the user; it goes through the same
/// centre-grid and frame rules as a detected pair.
pub fn pair_by_hand(ds: &Dataset, east: &str, north: &str) -> Result<VectorPair, String> {
    let (e, n) = hand_check(ds, east, north)?;
    let g = ds.group_of(east);
    let (frame, stem) = match std_match(e, n) {
        Some(m) => m,
        None => (mapping_frame(ds, g, e, n), name_match(e, n).unwrap_or_else(|| "vector".into())),
    };
    let units = units(e).or_else(|| units(n)).map(str::to_string);
    let mut p = build(ds, g, &Coords::of(ds), e, n, &stem, frame, units);
    p.by_hand = true;
    Ok(p)
}

fn hand_check<'a>(ds: &'a Dataset, a: &str, b: &str) -> Result<(&'a VarInfo, &'a VarInfo), String> {
    if a == b {
        return Err("a variable cannot be paired with itself".into());
    }
    let get = |p: &str| ds.var(p).ok_or_else(|| format!("{p}: no such variable"));
    let (e, n) = (get(a)?, get(b)?);
    if !e.dtype.is_numeric() || !n.dtype.is_numeric() {
        return Err("both components must be numeric".into());
    }
    if parent(&e.path) != parent(&n.path) {
        return Err("both components must be in the same group".into());
    }
    if !shapes_fit(e, n) {
        return Err("the components need the same or a staggered shape".into());
    }
    Ok((e, n))
}

fn detect_in_group(ds: &Dataset, g: &Group, coords: &Coords, out: &mut Vec<VectorPair>) {
    let cands: Vec<&VarInfo> =
        g.vars.iter().filter(|v| v.dtype.is_numeric() && v.shape.len() >= 2 && !coords.is_coord(v)).collect();
    let mut used = vec![false; cands.len()];
    // Rule 1 (standard names) claims its variables before rule 2 (names).
    for rule in [Rule::StandardName, Rule::CommonName] {
        for i in 0..cands.len() {
            for j in 0..cands.len() {
                if i == j || used[i] || used[j] {
                    continue;
                }
                let (e, n) = (cands[i], cands[j]);
                let matched = match rule {
                    Rule::StandardName => std_match(e, n),
                    Rule::CommonName => name_match(e, n).map(|stem| (mapping_frame(ds, g, e, n), stem)),
                };
                let Some((frame, stem)) = matched else {
                    continue;
                };
                let Some(units) = joint_units(e, n) else {
                    continue;
                };
                if !shapes_fit(e, n) {
                    continue;
                }
                out.push(build(ds, g, coords, e, n, &stem, frame, units));
                used[i] = true;
                used[j] = true;
            }
        }
    }
    // Rule 3 (magnitude/direction) takes what rules 1 and 2 left.
    for (di, d) in cands.iter().enumerate() {
        if used[di] {
            continue;
        }
        let Some(dir_stem) = direction_stem(d) else {
            continue;
        };
        for (i, m) in cands.iter().enumerate() {
            if used[i] || i == di || m.dims != d.dims || m.shape != d.shape {
                continue;
            }
            let Some(mag_stem) = magnitude_stem(m) else {
                continue;
            };
            if direction_stem(m).is_some() || !stem_eq(&mag_stem, &dir_stem) {
                continue;
            }
            let shared = if mag_stem.0 == dir_stem.0 { &dir_stem.0 } else { &dir_stem.1 };
            let stem = if !shared.is_empty() {
                shared.clone()
            } else if [m, d].iter().any(|v| has_std(v, "wind")) {
                "wind".into()
            } else {
                "vector".into()
            };
            out.push(magdir_pair(ds, g, coords, m, d, &stem));
            used[i] = true;
        }
    }
}

fn has_std(v: &VarInfo, word: &str) -> bool {
    attr_str(v, "standard_name").is_some_and(|s| s.to_ascii_lowercase().contains(word))
}

/// A comparable stem: the name without its token, and the standard name
/// without its token (either side may agree).
type Stem = (String, String);

fn stem_eq(a: &Stem, b: &Stem) -> bool {
    a.0 == b.0 || (!a.1.is_empty() && a.1 == b.1)
}

/// `name` without the first matching suffix or prefix token, trimmed of `_`.
fn strip_token(name: &str, suffixes: &[&str], prefixes: &[&str], whole: &[&str]) -> Option<String> {
    if whole.contains(&name) {
        return Some(String::new());
    }
    suffixes
        .iter()
        .find_map(|t| name.strip_suffix(t))
        .or_else(|| prefixes.iter().find_map(|t| name.strip_prefix(t)))
        .map(|s| s.trim_matches('_').to_string())
}

/// `s` without a trailing `_velocity`: `sea_water_velocity_to_direction`
/// and `sea_water_speed` share the stem `sea_water`.
fn velocity_less(s: &str) -> String {
    s.strip_suffix("_velocity").unwrap_or(s).to_string()
}

/// The stem of a direction candidate, or `None` when `v` is not one.
fn direction_stem(v: &VarInfo) -> Option<Stem> {
    let name = v.name.to_ascii_lowercase();
    let std = attr_str(v, "standard_name").map(str::to_ascii_lowercase).unwrap_or_default();
    let std_stem = ["_from_direction", "_to_direction"]
        .iter()
        .find_map(|t| std.strip_suffix(t))
        .or_else(|| std.strip_prefix("direction_of_"))
        .map(velocity_less);
    let by_name = strip_token(
        &name,
        &["_direction", "_dir"],
        &["direction_", "dir_", "wdir_", "wd_"],
        &["wd", "wdir", "dir", "direction"],
    )
    .map(|s| {
        // `x_from_direction` / `x_to_direction`: the stem is `x`.
        velocity_less(s.trim_end_matches("_from").trim_end_matches("_to"))
    });
    if std_stem.is_none() && by_name.is_none() {
        return None;
    }
    Some((by_name.unwrap_or_else(|| name.clone()), std_stem.unwrap_or_default()))
}

/// The stem of a magnitude candidate, or `None` when `v` is not one.
fn magnitude_stem(v: &VarInfo) -> Option<Stem> {
    let name = v.name.to_ascii_lowercase();
    let std = attr_str(v, "standard_name").map(str::to_ascii_lowercase).unwrap_or_default();
    if name.contains("gust") || std.contains("gust") {
        return None;
    }
    let std_stem = std.strip_suffix("_speed").map(str::to_string);
    let by_name =
        strip_token(&name, &["_speed", "_spd"], &["speed_", "spd_", "wspd_", "ws_"], &["ws", "wspd", "speed"]);
    const VELOCITY: [&str; 8] = ["m/s", "m s-1", "m s**-1", "km/h", "knots", "kt", "cm/s", "cm s-1"];
    let by_units = units(v).is_some_and(|u| VELOCITY.contains(&u.to_ascii_lowercase().as_str()));
    if std_stem.is_none() && by_name.is_none() && !by_units {
        return None;
    }
    Some((by_name.unwrap_or(name), std_stem.unwrap_or_default()))
}

/// Coordinate variables, which are never components or centre scalars
/// (ROMS `lat_u`/`lat_v` would otherwise pair by name).
pub(super) struct Coords(HashSet<String>);

impl Coords {
    /// Names listed in any variable's `coordinates` attribute.
    pub(super) fn of(ds: &Dataset) -> Coords {
        let names = ds
            .all_vars()
            .into_iter()
            .filter_map(|v| attr_str(v, "coordinates"))
            .flat_map(str::split_whitespace)
            .map(str::to_string)
            .collect();
        Coords(names)
    }

    /// Named as a coordinate, or a latitude/longitude by standard name or units.
    fn is_coord(&self, v: &VarInfo) -> bool {
        const LAT_LON_UNITS: [&str; 12] = [
            "degrees_north",
            "degree_north",
            "degrees_n",
            "degree_n",
            "degreen",
            "degreesn",
            "degrees_east",
            "degree_east",
            "degrees_e",
            "degree_e",
            "degreee",
            "degreese",
        ];
        self.0.contains(&v.name)
            || attr_str(v, "standard_name")
                .is_some_and(|s| s.eq_ignore_ascii_case("latitude") || s.eq_ignore_ascii_case("longitude"))
            || units(v).is_some_and(|u| LAT_LON_UNITS.contains(&u.to_ascii_lowercase().as_str()))
    }
}

#[derive(Clone, Copy)]
enum Rule {
    StandardName,
    CommonName,
}

pub(super) fn attr_str<'a>(v: &'a VarInfo, name: &str) -> Option<&'a str> {
    v.attrs.iter().find(|a| a.name == name).and_then(|a| a.value.as_str()).map(str::trim).filter(|s| !s.is_empty())
}

pub(super) fn units(v: &VarInfo) -> Option<&str> {
    attr_str(v, "units")
}

/// The pair's units when they agree (`Some(None)` when neither has any);
/// `None` when they differ.
fn joint_units(e: &VarInfo, n: &VarInfo) -> Option<Option<String>> {
    match (units(e), units(n)) {
        (Some(a), Some(b)) if a != b => None,
        (a, b) => Some(a.or(b).map(str::to_string)),
    }
}

/// Rule 1: standard names equal as `_`-separated tokens except
/// `eastward`/`northward` (geographic; grid-relative after `grid`) or
/// `x`/`y` (grid-relative). Returns the frame and the stem (the name
/// without those tokens). `e` must be the east (x) side.
fn std_match(e: &VarInfo, n: &VarInfo) -> Option<(Frame, String)> {
    let (a, b) = (attr_str(e, "standard_name")?, attr_str(n, "standard_name")?);
    let ta: Vec<&str> = a.split('_').collect();
    let tb: Vec<&str> = b.split('_').collect();
    if ta.len() != tb.len() {
        return None;
    }
    let mut diff = (0..ta.len()).filter(|&k| !ta[k].eq_ignore_ascii_case(tb[k]));
    let k = diff.next()?;
    if diff.next().is_some() {
        return None;
    }
    let (x, y) = (ta[k].to_ascii_lowercase(), tb[k].to_ascii_lowercase());
    let (frame, drop) = match (x.as_str(), y.as_str()) {
        ("eastward", "northward") if k > 0 && ta[k - 1].eq_ignore_ascii_case("grid") => {
            (Frame::GridRelative, k - 1..=k)
        }
        ("eastward", "northward") => (Frame::Geographic, k..=k),
        ("x", "y") => (Frame::GridRelative, k..=k),
        _ => return None,
    };
    let rest: Vec<&str> =
        ta.iter().enumerate().filter(|(i, t)| !drop.contains(i) && !t.is_empty()).map(|(_, t)| *t).collect();
    Some((frame, stem_or_vector(&rest.join("_"))))
}

/// Rule 2: common component names, case-insensitive. Returns the stem.
/// `e` must be the u (east) side.
fn name_match(e: &VarInfo, n: &VarInfo) -> Option<String> {
    const LISTED: [(&str, &str); 7] = [
        ("u", "v"),
        ("u10", "v10"),
        ("uas", "vas"),
        ("uo", "vo"),
        ("water_u", "water_v"),
        ("uu", "vv"),
        ("ubar", "vbar"),
    ];
    // Stems of fields that merely end in u/v (masks, map factors, coordinates).
    const NOT_VELOCITY: [&str; 7] = ["mask", "mapfac", "lat", "lon", "xlat", "xlong", "area"];
    let (a, b) = (e.name.to_ascii_lowercase(), n.name.to_ascii_lowercase());
    // ASCII lowercasing keeps byte offsets, so `e.name` slices like `a`.
    // `u_<s>`/`v_<s>`, then `<p>_u`/`<p>_v`, then `<p>u`/`<p>v`.
    let pattern = a
        .strip_prefix("u_")
        .filter(|s| b.strip_prefix("v_") == Some(*s))
        .map(|s| &e.name[e.name.len() - s.len()..])
        .or_else(|| {
            a.strip_suffix("_u")
                .filter(|p| b.strip_suffix("_v") == Some(*p))
                .or_else(|| a.strip_suffix('u').filter(|p| b.strip_suffix('v') == Some(*p)))
                .map(|p| &e.name[..p.len()])
        });
    match pattern {
        Some(stem) if NOT_VELOCITY.contains(&stem.trim_matches('_').to_ascii_lowercase().as_str()) => None,
        Some(stem) => Some(stem_or_vector(stem)),
        None if LISTED.contains(&(a.as_str(), b.as_str())) => Some("vector".into()),
        None => None,
    }
}

/// `s` without surrounding `_`, or `"vector"` if nothing is left.
fn stem_or_vector(s: &str) -> String {
    let s = s.trim_matches('_');
    if s.is_empty() { "vector".into() } else { s.to_string() }
}

/// Grid-relative when either component's `grid_mapping` is a rotated pole.
pub(super) fn mapping_frame(ds: &Dataset, g: &Group, e: &VarInfo, n: &VarInfo) -> Frame {
    let rotated = [e, n].iter().any(|v| {
        attr_str(v, "grid_mapping")
            .and_then(|m| m.split_whitespace().next())
            .map(|m| m.trim_end_matches(':'))
            .and_then(|m| find_named(ds, g, m))
            .and_then(|gm| attr_str(gm, "grid_mapping_name"))
            .is_some_and(|n| n.eq_ignore_ascii_case("rotated_latitude_longitude"))
    });
    if rotated { Frame::GridRelative } else { Frame::Undeclared }
}

/// A variable by name: in `g`, then the root group, then anywhere.
fn find_named<'a>(ds: &'a Dataset, g: &'a Group, name: &str) -> Option<&'a VarInfo> {
    g.vars
        .iter()
        .find(|v| v.name == name)
        .or_else(|| ds.root.vars.iter().find(|v| v.name == name))
        .or_else(|| ds.all_vars().into_iter().find(|v| v.name == name))
}

/// Axes on which the two components' dims differ (name or length).
fn staggered_axes(e: &VarInfo, n: &VarInfo) -> Vec<usize> {
    (0..e.dims.len()).filter(|&k| e.dims[k] != n.dims[k] || e.shape[k] != n.shape[k]).collect()
}

/// Same dims, or staggered: same rank (at least 2) and the same dims except
/// on one or two of the last two axes, where the lengths are equal or differ
/// by exactly one.
fn shapes_fit(e: &VarInfo, n: &VarInfo) -> bool {
    let r = e.dims.len();
    if r < 2 || n.dims.len() != r || e.shape.len() != r || n.shape.len() != r {
        return false;
    }
    staggered_axes(e, n).into_iter().all(|k| k >= r - 2 && e.shape[k].abs_diff(n.shape[k]) <= 1)
}

/// The pair's centre grid, coordinates and angle source. `e` and `n` must
/// satisfy [`shapes_fit`].
#[allow(clippy::too_many_arguments)]
pub(super) fn build(
    ds: &Dataset,
    g: &Group,
    coords: &Coords,
    e: &VarInfo,
    n: &VarInfo,
    stem: &str,
    frame: Frame,
    units: Option<String>,
) -> VectorPair {
    let r = e.dims.len();
    let stag = staggered_axes(e, n);
    let scalar = centre_scalar(ds, g, coords, e, n, &stag);
    let (centre_dims, centre_shape) = match &scalar {
        Some((s, _)) => (s.dims.clone(), s.shape.clone()),
        None => (0..r)
            .map(|k| {
                // The longer dim on a staggered axis (the east one if equal).
                let v = if n.shape[k] > e.shape[k] { n } else { e };
                (v.dims[k].clone(), v.shape[k])
            })
            .unzip(),
    };
    // A scalar's lat/lon; else the components' own when unstaggered and
    // shared by both (NEMO's `nav_lat_u`/`nav_lat_v` are not centres).
    let geo = match scalar {
        Some((_, geo)) if geo != Geo::None => Some(geo),
        _ => {
            let (ge, gn) = (detect_geo(ds, e), detect_geo(ds, n));
            (stag.is_empty() && ge == gn).then_some(ge)
        }
    };
    let (centre_lat, centre_lon, horiz) = match geo {
        Some(Geo::Regular { lat_dim, lon_dim, lat_var, lon_var }) => (Some(lat_var), Some(lon_var), [lat_dim, lon_dim]),
        Some(Geo::Curvilinear { y_dim, x_dim, lat_var, lon_var }) => (Some(lat_var), Some(lon_var), [y_dim, x_dim]),
        _ => (None, None, [r - 2, r - 1]),
    };
    let angle = angle_source(ds, g, &centre_dims, horiz);
    VectorPair {
        path: join(&g.path, &format!("{}+{}", e.name, n.name)),
        name: format!("{stem} ({}, {})", e.name, n.name),
        units,
        east: e.path.clone(),
        north: n.path.clone(),
        centre_dims,
        centre_shape,
        horiz,
        centre_lat,
        centre_lon,
        frame,
        angle,
        rotate: RotateChoice::Auto,
        by_hand: false,
        form: PairForm::EastNorth,
    }
}

/// A numeric variable of `g` (not a component) on the centre grid: its dims
/// equal the components' on unstaggered axes, and its length matches one
/// component's on each staggered axis. Preferred, in order: dims unlike
/// both components' (a true centre grid when staggered), then lat/lon.
fn centre_scalar<'a>(
    ds: &Dataset,
    g: &'a Group,
    coords: &Coords,
    e: &VarInfo,
    n: &VarInfo,
    stag: &[usize],
) -> Option<(&'a VarInfo, Geo)> {
    let r = e.dims.len();
    let mut best: Option<((bool, bool), &VarInfo, Geo)> = None;
    for s in &g.vars {
        if s.path == e.path || s.path == n.path || !s.dtype.is_numeric() || coords.is_coord(s) {
            continue;
        }
        if s.dims.len() != r || s.shape.len() != r {
            continue;
        }
        let fits = (0..r).all(|k| {
            if stag.contains(&k) {
                s.shape[k] == e.shape[k] || s.shape[k] == n.shape[k]
            } else {
                s.dims[k] == e.dims[k]
            }
        });
        if !fits {
            continue;
        }
        let geo = detect_geo(ds, s);
        let centred = stag.is_empty() || (s.dims != e.dims && s.dims != n.dims);
        let key = (centred, geo != Geo::None);
        if best.as_ref().is_none_or(|(k, ..)| key > *k) {
            best = Some((key, s, geo));
        }
    }
    best.map(|(_, s, geo)| (s, geo))
}

/// `angle` on the centre's horizontal dims, else `COSALPHA` + `SINALPHA`,
/// else computed from lat/lon. Variables with leading extra dims (WRF's
/// `Time`) count when their last two dims are the centre's (y, x).
fn angle_source(ds: &Dataset, g: &Group, centre_dims: &[String], horiz: [usize; 2]) -> AngleSource {
    let (Some(y), Some(x)) = (centre_dims.get(horiz[0]), centre_dims.get(horiz[1])) else {
        return AngleSource::FromLatLon;
    };
    let on_plane = |name: &str| {
        g.vars
            .iter()
            .chain(ds.root.vars.iter())
            .find(|v| {
                let r = v.dims.len();
                v.name.eq_ignore_ascii_case(name)
                    && v.dtype.is_numeric()
                    && r >= 2
                    && v.dims[r - 2] == *y
                    && v.dims[r - 1] == *x
            })
            .map(|v| v.path.clone())
    };
    if let Some(a) = on_plane("angle") {
        return AngleSource::Angle(a);
    }
    match (on_plane("COSALPHA"), on_plane("SINALPHA")) {
        (Some(c), Some(s)) => AngleSource::CosSin(c, s),
        _ => AngleSource::FromLatLon,
    }
}

/// The group path of a variable path.
pub(super) fn parent(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) | None => "/",
        Some(i) => &path[..i],
    }
}

pub(super) fn join(group: &str, name: &str) -> String {
    if group == "/" { format!("/{name}") } else { format!("{group}/{name}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{self, vectors};
    use crate::{Attr, AttrValue, DType, Dataset, Dim, FileFormat, Group, NcFile, Storage, VarInfo};

    /// A root-only dataset of f32 variables: (name, [(dim, len)], [(attr, text)]).
    #[allow(clippy::type_complexity)]
    fn ds_with(vars: &[(&str, &[(&str, usize)], &[(&str, &str)])]) -> Dataset {
        let mut dims: Vec<Dim> = Vec::new();
        let vars = vars
            .iter()
            .map(|(name, vdims, attrs)| {
                for (d, n) in vdims.iter() {
                    if !dims.iter().any(|x| x.name == *d) {
                        dims.push(Dim { name: d.to_string(), len: *n, unlimited: false });
                    }
                }
                VarInfo {
                    name: name.to_string(),
                    path: format!("/{name}"),
                    dims: vdims.iter().map(|(d, _)| d.to_string()).collect(),
                    shape: vdims.iter().map(|(_, n)| *n).collect(),
                    dtype: DType::F32,
                    attrs: attrs
                        .iter()
                        .map(|(k, v)| Attr { name: k.to_string(), value: AttrValue::Text(v.to_string()) })
                        .collect(),
                    storage: Storage::Unknown,
                }
            })
            .collect();
        Dataset {
            path: "mem.nc".into(),
            file_size: 0,
            format: FileFormat::Netcdf4,
            root: Group { name: "/".into(), path: "/".into(), dims, vars, attrs: vec![], groups: vec![] },
        }
    }

    const YX: &[(&str, usize)] = &[("y", 10), ("x", 20)];

    #[test]
    fn cf_eastward_pair() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
        let pairs = detect_pairs(&f.meta);
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.path, "/ocean/uo+vo");
        assert_eq!(p.name, "sea_water_velocity (uo, vo)");
        assert_eq!(p.frame, Frame::Geographic);
        assert_eq!(p.units.as_deref(), Some("m s-1"));
        assert_eq!(p.centre_lat.as_deref(), Some("/ocean/lat"));
        assert_eq!(p.centre_lon.as_deref(), Some("/ocean/lon"));
        assert_eq!((p.east.as_str(), p.north.as_str()), ("/ocean/uo", "/ocean/vo"));
        assert_eq!(p.horiz, [1, 2]);
        assert!(!p.by_hand);
        // Registered at open; virtual paths never go to the fast reader.
        assert_eq!(f.pairs().len(), 1);
        assert!(f.pair("/ocean/uo+vo").is_some());
        assert!(!f.fast_eligible("/ocean/uo+vo"));
        let vi = p.var_info();
        assert_eq!(vi.path, "/ocean/uo+vo");
        assert_eq!(vi.dims, ["time", "lat", "lon"]);
        assert_eq!(vi.shape, [24, 10, 20]);
        assert_eq!(vi.dtype, DType::F32);
        let attr = |n: &str| vi.attrs.iter().find(|a| a.name == n).and_then(|a| a.value.as_str());
        assert_eq!(attr("units"), Some("m s-1"));
        assert_eq!(attr("long_name"), Some("sea_water_velocity (uo, vo)"));
        assert_eq!(attr("coordinates"), Some("lat lon"));
    }

    #[test]
    fn cf_x_y_tokens_grid_relative() {
        let ds = ds_with(&[
            ("xw", YX, &[("standard_name", "x_wind")]),
            ("yw", YX, &[("standard_name", "y_wind")]),
            ("ge", YX, &[("standard_name", "grid_eastward_sea_water_velocity")]),
            ("gn", YX, &[("standard_name", "grid_northward_sea_water_velocity")]),
            ("sx", YX, &[("standard_name", "sea_water_x_velocity")]),
            ("sy", YX, &[("standard_name", "sea_water_y_velocity")]),
        ]);
        let pairs = detect_pairs(&ds);
        assert_eq!(pairs.len(), 3);
        assert!(pairs.iter().all(|p| p.frame == Frame::GridRelative));
        assert_eq!(pairs[0].name, "wind (xw, yw)");
        assert_eq!(pairs[0].path, "/xw+yw");
        assert_eq!(pairs[1].name, "sea_water_velocity (ge, gn)");
        assert_eq!(pairs[2].name, "sea_water_velocity (sx, sy)");
    }

    #[test]
    fn north_first_in_file_still_pairs_east_first() {
        let ds = ds_with(&[
            ("vn", YX, &[("standard_name", "northward_wind")]),
            ("ue", YX, &[("standard_name", "eastward_wind")]),
        ]);
        let pairs = detect_pairs(&ds);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].path, "/ue+vn");
        assert_eq!(pairs[0].frame, Frame::Geographic);
    }

    #[test]
    fn common_names_case_insensitive() {
        let ds = ds_with(&[
            ("U10", YX, &[]),
            ("V10", YX, &[]),
            ("water_u", YX, &[]),
            ("water_v", YX, &[]),
            ("u_surf", YX, &[]),
            ("v_surf", YX, &[]),
            ("windu", YX, &[]),
            ("windv", YX, &[]),
            ("temp", YX, &[]),
        ]);
        let pairs = detect_pairs(&ds);
        let names: Vec<&str> = pairs.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            ["vector (U10, V10)", "water (water_u, water_v)", "surf (u_surf, v_surf)", "wind (windu, windv)"]
        );
        assert!(pairs.iter().all(|p| p.frame == Frame::Undeclared));
        assert_eq!(pairs[0].path, "/U10+V10");
    }

    #[test]
    fn rotated_pole_grid_mapping_is_grid_relative() {
        let ds = ds_with(&[
            ("rp", &[], &[("grid_mapping_name", "rotated_latitude_longitude")]),
            ("u", YX, &[("grid_mapping", "rp")]),
            ("v", YX, &[("grid_mapping", "rp")]),
        ]);
        let pairs = detect_pairs(&ds);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].frame, Frame::GridRelative);
    }

    #[test]
    fn each_variable_in_at_most_one_pair() {
        let ds = ds_with(&[("u", YX, &[]), ("v", YX, &[]), ("U", YX, &[]), ("V", YX, &[])]);
        let pairs = detect_pairs(&ds);
        let paths: Vec<&str> = pairs.iter().map(|p| p.path.as_str()).collect();
        assert_eq!(paths, ["/u+v", "/U+V"]);
    }

    #[test]
    fn units_must_match() {
        let ds = ds_with(&[("u", YX, &[("units", "m s-1")]), ("v", YX, &[("units", "K")])]);
        assert!(detect_pairs(&ds).is_empty());
        let ds = ds_with(&[("u", YX, &[("units", "m s-1")]), ("v", YX, &[])]);
        let pairs = detect_pairs(&ds);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].units.as_deref(), Some("m s-1"));
    }

    #[test]
    fn shapes_must_stagger() {
        let ds = ds_with(&[("u", &[("y", 10), ("x", 20)], &[]), ("v", &[("y", 10), ("x2", 22)], &[])]);
        assert!(detect_pairs(&ds).is_empty());
        let ds = ds_with(&[("u", &[("y", 10), ("x", 20)], &[]), ("v", &[("y2", 11), ("x", 20)], &[])]);
        let pairs = detect_pairs(&ds);
        assert_eq!(pairs.len(), 1);
        // No scalar on the centre grid: the longer dim on the staggered axis.
        assert_eq!(pairs[0].centre_dims, ["y2", "x"]);
        assert_eq!(pairs[0].centre_shape, [11, 20]);
        assert_eq!(pairs[0].centre_lat, None);
        // Staggering only on the last two axes.
        let ds =
            ds_with(&[("u", &[("t", 3), ("y", 10), ("x", 20)], &[]), ("v", &[("t2", 4), ("y", 10), ("x", 20)], &[])]);
        assert!(detect_pairs(&ds).is_empty());
        // Different rank, and rank below 2, never pair.
        let ds = ds_with(&[("u", &[("y", 10), ("x", 20)], &[]), ("v", &[("t", 1), ("y", 10), ("x", 20)], &[])]);
        assert!(detect_pairs(&ds).is_empty());
        let ds = ds_with(&[("u", &[("t", 5)], &[]), ("v", &[("t", 5)], &[])]);
        assert!(detect_pairs(&ds).is_empty());
    }

    #[test]
    fn nemo_detected() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_nemo_like(d.path())).unwrap();
        let pairs = detect_pairs(&f.meta);
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.path, "/uo+vo");
        assert_eq!(p.name, "vector (uo, vo)");
        assert_eq!(p.units.as_deref(), Some("m/s"));
        assert_eq!(p.centre_dims, ["time_counter", "y", "x"]);
        assert_eq!(p.centre_shape, [2, 20, 30]);
        assert_eq!(p.horiz, [1, 2]);
        assert_eq!(p.centre_lat.as_deref(), Some("/nav_lat"));
        assert_eq!(p.centre_lon.as_deref(), Some("/nav_lon"));
        assert_eq!(p.frame, Frame::Undeclared);
        assert_eq!(p.angle, AngleSource::FromLatLon);
        assert_eq!(p.rotate, RotateChoice::Auto);
        assert_eq!(f.pairs().len(), 1);
    }

    #[test]
    fn roms_detected() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_roms_like(d.path())).unwrap();
        let pairs = detect_pairs(&f.meta);
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.path, "/u+v");
        assert_eq!(p.units.as_deref(), Some("meter second-1"));
        assert_eq!(p.centre_dims, ["ocean_time", "eta_rho", "xi_rho"]);
        assert_eq!(p.centre_shape, [2, 12, 16]);
        assert_eq!(p.horiz, [1, 2]);
        assert_eq!(p.angle, AngleSource::Angle("/angle".into()));
        assert_eq!(p.centre_lat.as_deref(), Some("/lat_rho"));
        assert_eq!(p.centre_lon.as_deref(), Some("/lon_rho"));
    }

    #[test]
    fn wrf_detected() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_wrf_like(d.path())).unwrap();
        let pairs = detect_pairs(&f.meta);
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.path, "/U+V");
        assert_eq!(p.units.as_deref(), Some("m s-1"));
        assert_eq!(p.centre_dims, ["Time", "south_north", "west_east"]);
        assert_eq!(p.centre_shape, [2, 10, 14]);
        assert_eq!(p.horiz, [1, 2]);
        assert_eq!(p.angle, AngleSource::CosSin("/COSALPHA".into(), "/SINALPHA".into()));
        assert_eq!(p.centre_lat.as_deref(), Some("/XLAT"));
        assert_eq!(p.centre_lon.as_deref(), Some("/XLONG"));
    }

    #[test]
    fn noclue_detected() {
        let d = tempfile::tempdir().unwrap();
        for turned in [false, true] {
            let f = NcFile::open(&vectors::write_noclue(d.path(), turned)).unwrap();
            let names: Vec<&str> = f.meta.root.vars.iter().map(|v| v.name.as_str()).collect();
            assert_eq!(names, ["h", "lat", "lon", "u", "v"]);
            let pairs = f.pairs();
            assert_eq!(pairs.len(), 1);
            let p = &pairs[0];
            assert_eq!(p.path, "/u+v");
            assert_eq!(p.units, None);
            assert_eq!(p.centre_dims, ["y", "x"]);
            assert_eq!(p.centre_lat.as_deref(), Some("/lat"));
            assert_eq!(p.frame, Frame::Undeclared);
            assert_eq!(p.angle, AngleSource::FromLatLon);
        }
    }

    #[test]
    fn hand_pair_and_unpair() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_noclue_renamed(d.path())).unwrap();
        assert!(detect_pairs(&f.meta).is_empty());
        assert!(f.pairs().is_empty());
        assert!(can_pair(&f.meta, "/aa", "/bb"));
        assert!(!can_pair(&f.meta, "/aa", "/aa"));
        assert!(!can_pair(&f.meta, "/aa", "/nope"));
        assert!(pair_by_hand(&f.meta, "/aa", "/nope").is_err());
        let added = f.add_pair(pair_by_hand(&f.meta, "/aa", "/bb").unwrap());
        assert_eq!(added.path, "/aa+bb");
        let p = f.pair("/aa+bb").unwrap();
        assert!(p.by_hand);
        assert_eq!(p.frame, Frame::Undeclared);
        assert_eq!(p.centre_lat.as_deref(), Some("/lat"));
        assert_eq!(f.pairs().len(), 1);
        f.remove_pair("/aa+bb");
        assert!(f.pair("/aa+bb").is_none());
        assert!(f.pairs().is_empty());
    }

    #[test]
    fn current_to_detects_one_pair() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_current_to(d.path())).unwrap();
        let pairs = detect_pairs(&f.meta);
        assert_eq!(pairs.len(), 1, "{pairs:?}");
        assert!(pairs[0].path.ends_with("~to.cwn.deg"), "{}", pairs[0].path);
        let PairForm::MagDir(k) = pairs[0].form else { panic!("not magdir") };
        assert!(!k.guessed.towards && !k.guessed.reference);
    }

    #[test]
    fn arome_detects_one_magdir_pair() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_arome_like(d.path())).unwrap();
        let pairs = detect_pairs(&f.meta);
        assert_eq!(pairs.len(), 1);
        let p = &pairs[0];
        assert_eq!(p.path, "/wind_speed@wind_direction~from.cwn.deg");
        assert_eq!(p.frame, Frame::Geographic);
        let PairForm::MagDir(c) = p.form else { panic!("not a magdir pair") };
        assert_eq!(c.guessed, super::super::Guessed::default());
        assert!(!p.by_hand);
    }

    #[test]
    fn silent_ws_wd_pairs_guessed() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_silent_magdir(d.path())).unwrap();
        let pairs = detect_pairs(&f.meta);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].path, "/ws@wd~from.cwn.deg");
        let PairForm::MagDir(c) = pairs[0].form else { panic!("not a magdir pair") };
        assert_eq!(c.guessed, super::super::Guessed { towards: true, reference: true, unit: true });
    }

    #[test]
    fn direction_pairs_each_height() {
        let ds = ds_with(&[("ws_10m", YX, &[]), ("ws_100m", YX, &[]), ("wd", YX, &[])]);
        assert!(detect_pairs(&ds).is_empty());
        let ds = ds_with(&[
            ("speed_10m", YX, &[]),
            ("speed_100m", YX, &[]),
            ("direction_10m", YX, &[]),
            ("direction_100m", YX, &[]),
        ]);
        let paths: Vec<String> = detect_pairs(&ds).into_iter().map(|p| p.path).collect();
        assert_eq!(paths, ["/speed_10m@direction_10m~from.cwn.deg", "/speed_100m@direction_100m~from.cwn.deg"]);
    }

    #[test]
    fn gust_and_wave_height_not_paired() {
        let ds = ds_with(&[
            ("gust", YX, &[("standard_name", "wind_speed_of_gust"), ("units", "m/s")]),
            ("wd", YX, &[("standard_name", "wind_from_direction")]),
        ]);
        assert!(detect_pairs(&ds).is_empty());
        let ds = ds_with(&[
            ("VHM0", YX, &[("standard_name", "sea_surface_wave_significant_height"), ("units", "m")]),
            ("VMDR", YX, &[("standard_name", "sea_surface_wave_from_direction")]),
        ]);
        assert!(detect_pairs(&ds).is_empty());
    }

    #[test]
    fn mask_and_mapfac_not_paired() {
        let ds = ds_with(&[("mask_u", YX, &[]), ("mask_v", YX, &[]), ("MAPFAC_U", YX, &[]), ("MAPFAC_V", YX, &[])]);
        assert!(detect_pairs(&ds).is_empty());
        let ds = ds_with(&[("wind_u", YX, &[]), ("wind_v", YX, &[])]);
        assert_eq!(detect_pairs(&ds).len(), 1);
    }

    #[test]
    fn uv_rules_still_win() {
        let ds = ds_with(&[
            ("current_speed", YX, &[]),
            ("current_dir", YX, &[]),
            ("uo", YX, &[("units", "m/s")]),
            ("vo", YX, &[("units", "m/s")]),
        ]);
        let paths: Vec<String> = detect_pairs(&ds).into_iter().map(|p| p.path).collect();
        assert_eq!(paths, ["/uo+vo", "/current_speed@current_dir~to.cwn.deg"]);
    }

    #[test]
    fn with_rotate_changes_path() {
        let ds = ds_with(&[("u", YX, &[]), ("v", YX, &[])]);
        let p = detect_pairs(&ds).remove(0);
        let off = p.with_rotate(RotateChoice::Off);
        assert!(off.path.ends_with("#off"), "{}", off.path);
        assert_eq!(off.rotate, RotateChoice::Off);
        let on = off.with_rotate(RotateChoice::On);
        assert_eq!(on.path, "/u+v#on");
        assert_eq!(on.with_rotate(RotateChoice::Auto), p);
        assert_eq!(off.var_info().path, off.path);
    }

    /// The fixtures hold the known field at each component's own position.
    #[test]
    fn fixtures_hold_the_known_field() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&vectors::write_roms_like(d.path())).unwrap();
        let read = |v: &str, shape: [usize; 3]| {
            let r = crate::SliceRequest { var: v.into(), start: vec![0; 3], count: shape.to_vec(), stride: vec![1; 3] };
            f.read_slice(&r).unwrap().data
        };
        let (s, c) = 0.5f64.sin_cos();
        let u = read("/u", [2, 12, 15]);
        // u[1, j=3, i=4] sits at (p, q) = (4.5, 3).
        let want = vectors::known_east(4.5) * c + vectors::known_north(3.0) * s;
        assert!((f64::from(u[15 * 12 + 3 * 15 + 4]) - want).abs() < 1e-5);
        let v = read("/v", [2, 11, 16]);
        // v[0, j=2, i=7] sits at (p, q) = (7, 2.5).
        let want = -vectors::known_east(7.0) * s + vectors::known_north(2.5) * c;
        assert!((f64::from(v[2 * 16 + 7]) - want).abs() < 1e-5);

        let f = NcFile::open(&vectors::write_nemo_like(d.path())).unwrap();
        let r =
            crate::SliceRequest { var: "/uo".into(), start: vec![1, 5, 10], count: vec![1, 1, 1], stride: vec![1; 3] };
        assert!(f.read_slice(&r).unwrap().data[0].is_nan(), "uo[:,5,10] is fill");
    }
}
