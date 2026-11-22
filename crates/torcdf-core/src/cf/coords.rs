//! Detection of geographic, time and vertical coordinates (spec 6.4).

use crate::{Dataset, VarInfo};

#[derive(Debug, Clone, PartialEq)]
pub enum Geo {
    None,
    Regular { lat_dim: usize, lon_dim: usize, lat_var: String, lon_var: String },
    Curvilinear { y_dim: usize, x_dim: usize, lat_var: String, lon_var: String },
}

/// Indices into the variable's `dims`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Axes {
    pub time: Option<usize>,
    pub vertical: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Lat,
    Lon,
}

fn attr_str<'a>(v: &'a VarInfo, name: &str) -> Option<&'a str> {
    v.attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| a.value.as_str())
        .map(str::trim)
}

/// Looks a variable up by name: next to `from`, then in the root group, then anywhere.
fn find_named<'a>(ds: &'a Dataset, from: &VarInfo, name: &str) -> Option<&'a VarInfo> {
    ds.group_of(&from.path)
        .vars
        .iter()
        .find(|c| c.name == name)
        .or_else(|| ds.root.vars.iter().find(|c| c.name == name))
        .or_else(|| ds.all_vars().into_iter().find(|c| c.name == name))
}

/// Dimension coordinate variables (same-named 1-D variables) in `dims` order,
/// then the variables named by the `coordinates` attribute.
fn candidates<'a>(ds: &'a Dataset, v: &VarInfo) -> Vec<&'a VarInfo> {
    let group = ds.group_of(&v.path);
    let mut out: Vec<&VarInfo> = v
        .dims
        .iter()
        .filter_map(|d| {
            group
                .vars
                .iter()
                .chain(ds.root.vars.iter())
                .find(|c| c.name == *d && c.dims.len() == 1 && c.dims[0] == *d)
        })
        .collect();
    if let Some(coords) = attr_str(v, "coordinates") {
        out.extend(coords.split_whitespace().filter_map(|n| find_named(ds, v, n)));
    }
    out
}

fn classify(c: &VarInfo) -> Option<Kind> {
    if let Some(sn) = attr_str(c, "standard_name") {
        match sn.to_ascii_lowercase().as_str() {
            "latitude" => return Some(Kind::Lat),
            "longitude" => return Some(Kind::Lon),
            _ => {}
        }
    }
    if let Some(u) = attr_str(c, "units") {
        match u.to_ascii_lowercase().as_str() {
            "degrees_north" | "degree_north" | "degrees_n" | "degree_n" | "degreen" | "degreesn" => {
                return Some(Kind::Lat);
            }
            "degrees_east" | "degree_east" | "degrees_e" | "degree_e" | "degreee" | "degreese" => {
                return Some(Kind::Lon);
            }
            _ => {}
        }
    }
    // The axis fallback needs no conflicting evidence: a projected x/y (metres,
    // projection_x_coordinate) is not lat/lon even though it has axis X/Y.
    let conflicting_std = attr_str(c, "standard_name").is_some();
    let conflicting_units = attr_str(c, "units")
        .is_some_and(|u| !u.to_ascii_lowercase().starts_with("degree"));
    if !conflicting_std
        && !conflicting_units
        && let Some(axis) = attr_str(c, "axis")
    {
        match axis.to_ascii_uppercase().as_str() {
            "Y" => return Some(Kind::Lat),
            "X" => return Some(Kind::Lon),
            _ => {}
        }
    }
    match c.name.to_ascii_lowercase().as_str() {
        "lat" | "latitude" | "nav_lat" | "lat_rho" | "xlat" => Some(Kind::Lat),
        "lon" | "longitude" | "nav_lon" | "lon_rho" | "xlong" => Some(Kind::Lon),
        _ => None,
    }
}

/// Positions of `c`'s dimensions within `v.dims`; `None` if any is missing.
fn dim_positions(v: &VarInfo, c: &VarInfo) -> Option<Vec<usize>> {
    c.dims.iter().map(|d| v.dims.iter().position(|x| x == d)).collect()
}

pub fn detect_geo(ds: &Dataset, v: &VarInfo) -> Geo {
    let usable: Vec<(Kind, &VarInfo, Vec<usize>)> = candidates(ds, v)
        .into_iter()
        .filter_map(|c| {
            let kind = classify(c)?;
            let pos = dim_positions(v, c)?;
            matches!(pos.len(), 1 | 2).then_some((kind, c, pos))
        })
        .collect();

    let Some((_, lat, lat_pos)) = usable.iter().find(|(k, ..)| *k == Kind::Lat) else {
        return Geo::None;
    };
    // The longitude must have the same shape class (1-D vs 2-D) and, when 2-D,
    // the same dimensions as the latitude.
    let Some((_, lon, lon_pos)) = usable.iter().find(|(k, c, p)| {
        *k == Kind::Lon && p.len() == lat_pos.len() && (p.len() == 1 || c.dims == lat.dims)
    }) else {
        return Geo::None;
    };
    let (lat_var, lon_var) = (lat.path.clone(), lon.path.clone());
    if lat_pos.len() == 1 {
        if lat_pos[0] == lon_pos[0] {
            return Geo::None;
        }
        Geo::Regular { lat_dim: lat_pos[0], lon_dim: lon_pos[0], lat_var, lon_var }
    } else {
        Geo::Curvilinear { y_dim: lat_pos[0], x_dim: lat_pos[1], lat_var, lon_var }
    }
}

pub fn detect_axes(ds: &Dataset, v: &VarInfo) -> Axes {
    let group = ds.group_of(&v.path);
    let mut axes = Axes::default();
    for (i, dim) in v.dims.iter().enumerate() {
        let Some(c) = group
            .vars
            .iter()
            .chain(ds.root.vars.iter())
            .find(|c| c.name == *dim && c.dims.len() == 1)
        else {
            continue;
        };
        if axes.time.is_none() && is_time(c) {
            axes.time = Some(i);
        } else if axes.vertical.is_none() && is_vertical(c) {
            axes.vertical = Some(i);
        }
    }
    axes
}

fn is_time(c: &VarInfo) -> bool {
    attr_str(c, "axis").is_some_and(|a| a.eq_ignore_ascii_case("T"))
        || attr_str(c, "standard_name").is_some_and(|s| s.eq_ignore_ascii_case("time"))
        || attr_str(c, "units").is_some_and(|u| u.to_ascii_lowercase().contains(" since "))
}

fn is_vertical(c: &VarInfo) -> bool {
    attr_str(c, "axis").is_some_and(|a| a.eq_ignore_ascii_case("Z"))
        || c.attrs.iter().any(|a| a.name == "positive")
        || attr_str(c, "standard_name").is_some_and(|s| {
            matches!(
                s.to_ascii_lowercase().as_str(),
                "depth" | "height" | "altitude" | "air_pressure" | "model_level_number"
            )
        })
}

/// Wraps a longitude to `[-180, 180)`.
pub fn wrap_lon(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}
