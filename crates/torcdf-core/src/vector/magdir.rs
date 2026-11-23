//! Magnitude/direction pairs: the direction convention, conversion to
//! east/north, reading the convention from the file and pairing by hand.

use super::detect::{Coords, attr_str, build, join, mapping_frame, parent, units};
use super::{Frame, PairForm, VectorPair};
use crate::{Dataset, Group, VarInfo};

/// Whether the direction is where the vector comes from or goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Towards {
    From,
    To,
}

/// Where the direction angle is measured from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AngleRef {
    ClockwiseFromNorth,
    CounterClockwiseFromEast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AngleUnit {
    Degrees,
    Radians,
}

/// Which parts of a convention were guessed rather than read from the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Guessed {
    pub towards: bool,
    pub reference: bool,
    pub unit: bool,
}

/// How a direction variable is to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DirConvention {
    pub towards: Towards,
    pub reference: AngleRef,
    pub unit: AngleUnit,
    pub guessed: Guessed,
}

impl DirConvention {
    /// Short form for pair paths: `"from.cwn.deg"`, `"to.ccwe.rad"`, ...
    pub fn tag(&self) -> String {
        let t = match self.towards {
            Towards::From => "from",
            Towards::To => "to",
        };
        let r = match self.reference {
            AngleRef::ClockwiseFromNorth => "cwn",
            AngleRef::CounterClockwiseFromEast => "ccwe",
        };
        let u = match self.unit {
            AngleUnit::Degrees => "deg",
            AngleUnit::Radians => "rad",
        };
        format!("{t}.{r}.{u}")
    }

    fn radians(&self, dir: f64) -> f64 {
        match self.unit {
            AngleUnit::Degrees => dir.to_radians(),
            AngleUnit::Radians => dir,
        }
    }
}

/// East and north components of magnitude `m` at direction `dir`. NaN in,
/// NaN out.
pub fn magdir_to_en(m: f64, dir: f64, c: &DirConvention) -> (f64, f64) {
    if m.is_nan() || dir.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let (s, co) = c.radians(dir).sin_cos();
    let (e, n) = match c.reference {
        AngleRef::ClockwiseFromNorth => (m * s, m * co),
        AngleRef::CounterClockwiseFromEast => (m * co, m * s),
    };
    match c.towards {
        Towards::To => (e, n),
        Towards::From => (-e, -n),
    }
}

/// Magnitude and direction (in `[0, 360)` degrees or `[0, 2π)` radians) of
/// the vector east `e`, north `n`; the inverse of [`magdir_to_en`].
pub fn en_to_magdir(e: f64, n: f64, c: &DirConvention) -> (f64, f64) {
    let (e, n) = match c.towards {
        Towards::To => (e, n),
        Towards::From => (-e, -n),
    };
    let angle = match c.reference {
        AngleRef::ClockwiseFromNorth => e.atan2(n),
        AngleRef::CounterClockwiseFromEast => n.atan2(e),
    };
    let dir = match c.unit {
        AngleUnit::Degrees => angle.to_degrees().rem_euclid(360.0),
        AngleUnit::Radians => angle.rem_euclid(std::f64::consts::TAU),
    };
    (e.hypot(n), dir)
}

/// Whether `v`'s CF standard name says its direction is where the vector
/// comes from or goes to (`*_from_direction`, `*_to_direction`,
/// `direction_of_*`).
pub(super) fn cf_direction(v: &VarInfo) -> Option<Towards> {
    let s = attr_str(v, "standard_name")?.to_ascii_lowercase();
    if s.ends_with("_from_direction") {
        Some(Towards::From)
    } else if s.ends_with("_to_direction") || s.starts_with("direction_of_") {
        Some(Towards::To)
    } else {
        None
    }
}

/// The convention the file declares for `direction`, with a guess (flagged
/// in `guessed`) for what it does not say.
pub fn convention_of(magnitude: &VarInfo, direction: &VarInfo) -> DirConvention {
    let cf = cf_direction(direction);
    let declared_towards = cf;
    let text = units(direction).map(str::to_ascii_lowercase).unwrap_or_default();
    let anti = text.contains("anticlockwise") || text.contains("anti-clockwise");
    let declared_ref =
        if text.contains("counterclockwise") || text.contains("counter-clockwise") || text.contains("from east") {
            Some(AngleRef::CounterClockwiseFromEast)
        } else if anti {
            // "anticlockwise from north" cannot be expressed: guess.
            None
        } else if (text.contains("clockwise") && text.contains("north")) || cf.is_some() {
            // CF defines its direction names clockwise from north.
            Some(AngleRef::ClockwiseFromNorth)
        } else {
            None
        };
    let declared_unit = if text.starts_with("rad") {
        Some(AngleUnit::Radians)
    } else if text.starts_with("deg") {
        Some(AngleUnit::Degrees)
    } else {
        None
    };
    let towards = declared_towards.unwrap_or_else(|| {
        let mut names = String::new();
        for v in [magnitude, direction] {
            names.push_str(&v.name.to_ascii_lowercase());
            names.push(' ');
            names.push_str(&attr_str(v, "standard_name").unwrap_or("").to_ascii_lowercase());
            names.push(' ');
        }
        if names.contains("wind") {
            Towards::From
        } else if names.contains("water") || names.contains("current") {
            Towards::To
        } else {
            Towards::From
        }
    });
    DirConvention {
        towards,
        reference: declared_ref.unwrap_or(AngleRef::ClockwiseFromNorth),
        unit: declared_unit.unwrap_or(AngleUnit::Degrees),
        guessed: Guessed {
            towards: declared_towards.is_none(),
            reference: declared_ref.is_none(),
            unit: declared_unit.is_none(),
        },
    }
}

/// A magnitude/direction pair of `m` and `d` (same group and grid) named
/// after `stem`.
pub(super) fn magdir_pair(
    ds: &Dataset,
    g: &Group,
    coords: &Coords,
    m: &VarInfo,
    d: &VarInfo,
    stem: &str,
) -> VectorPair {
    let frame = if cf_direction(d).is_some() { Frame::Geographic } else { mapping_frame(ds, g, m, d) };
    let c = convention_of(m, d);
    let mut p = build(ds, g, coords, m, d, stem, frame, units(m).map(str::to_string));
    p.path = join(&g.path, &format!("{}@{}~{}", m.name, d.name, c.tag()));
    p.name = format!("{stem} ({}, {})", m.name, d.name);
    p.form = PairForm::MagDir(c);
    p
}

fn hand_check<'a>(ds: &'a Dataset, magnitude: &str, direction: &str) -> Result<(&'a VarInfo, &'a VarInfo), String> {
    if magnitude == direction {
        return Err("a variable cannot be paired with itself".into());
    }
    let get = |p: &str| ds.var(p).ok_or_else(|| format!("{p}: no such variable"));
    let (m, d) = (get(magnitude)?, get(direction)?);
    if !m.dtype.is_numeric() || !d.dtype.is_numeric() {
        return Err("both must be numeric".into());
    }
    if parent(&m.path) != parent(&d.path) {
        return Err("both must be in the same group".into());
    }
    if m.dims != d.dims || m.shape != d.shape {
        return Err("magnitude and direction need the same grid".into());
    }
    Ok((m, d))
}

/// Whether `magnitude` and `direction` can be paired by hand.
pub fn can_pair_magdir(ds: &Dataset, magnitude: &str, direction: &str) -> bool {
    hand_check(ds, magnitude, direction).is_ok()
}

/// A magnitude/direction pair chosen by the user, with the convention read
/// from the file (or guessed).
pub fn pair_magdir_by_hand(ds: &Dataset, magnitude: &str, direction: &str) -> Result<VectorPair, String> {
    let (m, d) = hand_check(ds, magnitude, direction)?;
    let g = ds.group_of(magnitude);
    let mut p = magdir_pair(ds, g, &Coords::of(ds), m, d, "vector");
    p.by_hand = true;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::vectors;
    use crate::{Attr, AttrValue, DType, Dim, FileFormat, Group, NcFile, RotateChoice, SliceRequest, Storage};

    fn c(t: Towards, r: AngleRef, u: AngleUnit) -> DirConvention {
        DirConvention { towards: t, reference: r, unit: u, guessed: Guessed::default() }
    }

    fn open(write: fn(&std::path::Path) -> std::path::PathBuf) -> (tempfile::TempDir, NcFile) {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&write(d.path())).unwrap();
        (d, f)
    }

    fn convention(f: &NcFile, m: &str, d: &str) -> DirConvention {
        convention_of(f.meta.var(m).unwrap(), f.meta.var(d).unwrap())
    }

    #[test]
    fn magdir_conversion_table() {
        use {AngleRef::*, AngleUnit::*, Towards::*};
        let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9;
        assert!(near(magdir_to_en(2.0, 90.0, &c(To, ClockwiseFromNorth, Degrees)), (2.0, 0.0)));
        assert!(near(magdir_to_en(2.0, 0.0, &c(To, ClockwiseFromNorth, Degrees)), (0.0, 2.0)));
        assert!(near(magdir_to_en(2.0, 270.0, &c(From, ClockwiseFromNorth, Degrees)), (2.0, 0.0)));
        assert!(near(magdir_to_en(2.0, 90.0, &c(To, CounterClockwiseFromEast, Degrees)), (0.0, 2.0)));
        assert!(near(magdir_to_en(2.0, 0.0, &c(From, CounterClockwiseFromEast, Degrees)), (-2.0, 0.0)));
        assert!(near(magdir_to_en(2.0, std::f64::consts::FRAC_PI_2, &c(To, ClockwiseFromNorth, Radians)), (2.0, 0.0)));
        let (m, d) = en_to_magdir(2.0, 0.0, &c(From, ClockwiseFromNorth, Degrees));
        assert!((m - 2.0).abs() < 1e-9 && (d - 270.0).abs() < 1e-9);
        assert!(magdir_to_en(f64::NAN, 10.0, &c(To, ClockwiseFromNorth, Degrees)).0.is_nan());
    }

    #[test]
    fn convention_from_arome_attrs() {
        let (_d, f) = open(vectors::write_arome_like);
        let k = convention(&f, "/wind_speed", "/wind_direction");
        assert_eq!((k.towards, k.reference, k.unit), (Towards::From, AngleRef::ClockwiseFromNorth, AngleUnit::Degrees));
        assert_eq!(k.guessed, Guessed::default());
    }

    #[test]
    fn convention_guessed_when_silent() {
        let (_d, f) = open(vectors::write_silent_magdir);
        let k = convention(&f, "/ws", "/wd");
        assert_eq!((k.towards, k.reference, k.unit), (Towards::From, AngleRef::ClockwiseFromNorth, AngleUnit::Degrees));
        assert_eq!(k.guessed, Guessed { towards: true, reference: true, unit: true });
    }

    #[test]
    fn convention_current_is_to() {
        let (_d, f) = open(vectors::write_current_to);
        let k = convention(&f, "/sea_water_speed", "/sea_water_velocity_to_direction");
        assert_eq!(k.towards, Towards::To);
        assert_eq!(k.unit, AngleUnit::Degrees);
        assert!(!k.guessed.towards && !k.guessed.unit && !k.guessed.reference);
        assert_eq!(k.reference, AngleRef::ClockwiseFromNorth);
    }

    #[test]
    fn cf_direction_name_is_clockwise_from_north_unless_units_say_otherwise() {
        let (_d, f) = open(vectors::write_current_to);
        let m = f.meta.var("/sea_water_speed").unwrap();
        let mut d = f.meta.var("/sea_water_velocity_to_direction").unwrap().clone();
        d.attrs.retain(|a| a.name != "units");
        d.attrs
            .push(Attr { name: "units".into(), value: AttrValue::Text("degrees counterclockwise from east".into()) });
        let k = convention_of(m, &d);
        assert_eq!(k.reference, AngleRef::CounterClockwiseFromEast);
        assert!(!k.guessed.reference);
    }

    #[test]
    fn anticlockwise_needs_from_east() {
        let (_d, f) = open(vectors::write_current_to);
        let m = f.meta.var("/sea_water_speed").unwrap();
        let mut d = f.meta.var("/sea_water_velocity_to_direction").unwrap().clone();
        let mut with = |text: &str| {
            d.attrs.retain(|a| a.name != "units");
            d.attrs.push(Attr { name: "units".into(), value: AttrValue::Text(text.into()) });
            convention_of(m, &d)
        };
        let k = with("degrees anticlockwise from east");
        assert_eq!(k.reference, AngleRef::CounterClockwiseFromEast);
        assert!(!k.guessed.reference);
        let k = with("degrees anti-clockwise from north");
        assert_eq!(k.reference, AngleRef::ClockwiseFromNorth);
        assert!(k.guessed.reference);
    }

    #[test]
    fn magdir_path_and_with_convention() {
        let (_d, f) = open(vectors::write_arome_like);
        let p = pair_magdir_by_hand(&f.meta, "/wind_speed", "/wind_direction").unwrap();
        assert_eq!(p.path, "/wind_speed@wind_direction~from.cwn.deg");
        assert_eq!(p.name, "vector (wind_speed, wind_direction)");
        assert_eq!((p.magnitude(), p.direction()), ("/wind_speed", "/wind_direction"));
        assert_eq!(p.units.as_deref(), Some("m/s"));
        assert!(p.by_hand);
        let to = c(Towards::To, AngleRef::ClockwiseFromNorth, AngleUnit::Degrees);
        assert_eq!(p.with_convention(to).path, "/wind_speed@wind_direction~to.cwn.deg");
        let on = p.with_rotate(RotateChoice::On).with_convention(to);
        assert_eq!(on.path, "/wind_speed@wind_direction~to.cwn.deg#on");
        assert_eq!(on.with_rotate(RotateChoice::Auto).path, "/wind_speed@wind_direction~to.cwn.deg");
    }

    #[test]
    fn magdir_hand_rejects_other_grid() {
        let var = |name: &str, shape: [usize; 2]| VarInfo {
            name: name.into(),
            path: format!("/{name}"),
            dims: vec!["y".into(), format!("x{}", shape[1])],
            shape: shape.to_vec(),
            dtype: DType::F32,
            attrs: vec![Attr { name: "units".into(), value: AttrValue::Text("m".into()) }],
            storage: Storage::Unknown,
        };
        let ds = Dataset {
            path: "mem.nc".into(),
            file_size: 0,
            format: FileFormat::Netcdf4,
            root: Group {
                name: "/".into(),
                path: "/".into(),
                dims: vec![Dim { name: "y".into(), len: 3, unlimited: false }],
                vars: vec![var("a", [3, 4]), var("b", [3, 5])],
                attrs: vec![],
                groups: vec![],
            },
        };
        assert_eq!(pair_magdir_by_hand(&ds, "/a", "/b").unwrap_err(), "magnitude and direction need the same grid");
        assert_eq!(pair_magdir_by_hand(&ds, "/a", "/a").unwrap_err(), "a variable cannot be paired with itself");
        assert!(!can_pair_magdir(&ds, "/a", "/b"));
    }

    fn arome_plane(f: &NcFile, p: &VectorPair, t: usize) -> crate::Slice {
        let p = f.add_pair(p.clone());
        let req =
            SliceRequest { var: p.path.clone(), start: vec![t, 0, 0, 0], count: vec![1, 1, 4, 5], stride: vec![1; 4] };
        f.read_slice(&req).unwrap()
    }

    fn arome_pair(f: &NcFile) -> VectorPair {
        pair_magdir_by_hand(&f.meta, "/wind_speed", "/wind_direction").unwrap()
    }

    #[test]
    fn arome_reads_known_arrows() {
        let (_d, f) = open(vectors::write_arome_like);
        let p = arome_pair(&f);
        let s = arome_plane(&f, &p, 0);
        let v = s.vector.as_ref().unwrap();
        let k = 5 + 1;
        assert!((v.east[k] - 10.0).abs() < 1e-3 && v.north[k].abs() < 1e-3);
        assert!((s.data[k] - 10.0).abs() < 1e-3);
        let s = arome_plane(&f, &p, 1);
        let v = s.vector.as_ref().unwrap();
        let k = 2 * 5 + 2;
        assert!(v.east[k].abs() < 1e-3 && (v.north[k] + 10.0).abs() < 1e-3);
    }

    #[test]
    fn magdir_wraps_and_unpacks() {
        let (_d, f) = open(vectors::write_arome_like);
        let p = arome_pair(&f);
        let s = arome_plane(&f, &p, 1);
        let v = s.vector.as_ref().unwrap();
        // 270 degrees from: blows east; 90 degrees from: blows west.
        let (a, b) = (5 + 1, 5 + 2);
        assert!((v.east[a] - 10.0).abs() < 1e-3 && v.north[a].abs() < 1e-3);
        assert!((v.east[b] + 10.0).abs() < 1e-3 && v.north[b].abs() < 1e-3);
    }

    #[test]
    fn magdir_missing_either_side() {
        let (_d, f) = open(vectors::write_arome_like);
        let p = arome_pair(&f);
        let s = arome_plane(&f, &p, 0);
        let v = s.vector.as_ref().unwrap();
        for k in [0, 3 * 5 + 4] {
            assert!(s.data[k].is_nan() && v.east[k].is_nan() && v.north[k].is_nan(), "cell {k}");
        }
        assert_eq!(s.valid_count, 18);
    }

    #[test]
    fn current_to_points_east() {
        let (_d, f) = open(vectors::write_current_to);
        let p = pair_magdir_by_hand(&f.meta, "/sea_water_speed", "/sea_water_velocity_to_direction").unwrap();
        let p = f.add_pair(p);
        let req = SliceRequest { var: p.path.clone(), start: vec![0, 0], count: vec![3, 3], stride: vec![1; 2] };
        let s = f.read_slice(&req).unwrap();
        let v = s.vector.as_ref().unwrap();
        assert!(v.east.iter().all(|e| (e - 0.5).abs() < 1e-6));
        assert!(v.north.iter().all(|n| n.abs() < 1e-6));
    }
}
