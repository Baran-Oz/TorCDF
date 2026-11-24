//! Checking a formula against a file's metadata: names to paths, shapes,
//! the new variable's name. No data is read.

use super::{CalcError, Expr, parse};
use crate::vector::VectorPair;
use crate::{Attr, AttrValue, DType, Dataset, VarInfo};
use std::sync::Arc;

/// A calculated variable: a formula over a file's variables, computed on
/// read. It lasts for the session.
#[derive(Debug, Clone, PartialEq)]
pub struct Derived {
    pub name: String,
    /// The group of the formula's first variable, then the name.
    pub path: String,
    /// The formula as typed (trimmed).
    pub formula: String,
    pub units: Option<String>,
    /// Dimensions and shape of the result; 32-bit floats.
    pub info: VarInfo,
    /// The formula with every name replaced by its variable's full path.
    pub expr: Expr,
    /// Paths the formula uses directly, in first-use order.
    pub inputs: Vec<String>,
}

/// What a formula's names can refer to.
pub(crate) struct Known<'a> {
    pub ds: &'a Dataset,
    pub derived: &'a [Arc<Derived>],
    pub pairs: &'a [Arc<VectorPair>],
}

impl Known<'_> {
    /// File variables, then calculated ones.
    fn all(&self) -> Vec<&VarInfo> {
        let mut v = self.ds.all_vars();
        v.extend(self.derived.iter().map(|d| &d.info));
        v
    }

    /// Paths of calculated variables that use `path`, at any depth.
    fn deep_users(&self, path: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut todo = vec![path.to_string()];
        while let Some(p) = todo.pop() {
            for d in self.derived {
                if d.inputs.contains(&p) && !out.contains(&d.path) {
                    out.push(d.path.clone());
                    todo.push(d.path.clone());
                }
            }
        }
        out
    }
}

fn valid_name(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_ascii_alphabetic()) && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn parent(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) | None => "/",
        Some(i) => &path[..i],
    }
}

fn dims_text(v: &VarInfo) -> String {
    format!("({})", v.dims.join(", "))
}

/// Whether `small`'s dims are an order-keeping subsequence of `big`'s,
/// with equal lengths.
fn fits(small: &VarInfo, big: &VarInfo) -> bool {
    let mut j = 0;
    for (d, n) in small.dims.iter().zip(&small.shape) {
        loop {
            let Some(bd) = big.dims.get(j) else { return false };
            j += 1;
            if bd == d {
                if big.shape[j - 1] != *n {
                    return false;
                }
                break;
            }
        }
    }
    true
}

/// Checks `formula` for a new variable `name`. `replacing`: the path being
/// edited; `users`: names of the variables and pairs using it.
pub(crate) fn check(
    known: &Known<'_>,
    name: &str,
    formula: &str,
    units: Option<&str>,
    replacing: Option<&str>,
    users: &[String],
) -> Result<Derived, CalcError> {
    let parsed = parse(formula)?;
    let all = known.all();
    // Variables that would make a loop: the edited one and its users.
    let mut banned: Vec<String> = Vec::new();
    if let Some(r) = replacing {
        banned.push(r.to_string());
        banned.extend(known.deep_users(r));
    }
    let mut inputs: Vec<&VarInfo> = Vec::new();
    let expr = resolve(parsed, known, &all, &banned, &mut inputs)?;
    let Some(first) = inputs.first() else {
        return Err(CalcError::msg("a formula needs at least one variable"));
    };
    // The widest input sets the dimensions (the first one on ties).
    let widest = inputs.iter().fold(*first, |w, v| if v.dims.len() > w.dims.len() { v } else { w });
    if let Some(odd) = inputs.iter().find(|v| !fits(v, widest)) {
        return Err(CalcError::msg(format!(
            "{} and {} have different dimensions: {} vs {}",
            widest.name,
            odd.name,
            dims_text(widest),
            dims_text(odd)
        )));
    }
    if !valid_name(name) {
        return Err(CalcError::msg("the name must start with a letter and use only letters, digits and _"));
    }
    let group = parent(&first.path);
    let path = if group == "/" { format!("/{name}") } else { format!("{group}/{name}") };
    if let Some(r) = replacing
        && r != path
    {
        return Err(CalcError::msg("the name cannot change while editing; create a new variable instead"));
    }
    let clash_file = known.ds.all_vars().iter().any(|v| parent(&v.path) == group && v.name == name);
    let clash_calc = known.derived.iter().any(|d| d.name == name && Some(d.path.as_str()) != replacing);
    if clash_file || clash_calc {
        return Err(CalcError::msg(format!("{name} already exists")));
    }
    if let Some(r) = replacing
        && !users.is_empty()
        && let Some(old) = known.derived.iter().find(|d| d.path == r)
        && (old.info.dims != widest.dims || old.info.shape != widest.shape)
    {
        return Err(CalcError::msg(format!("{name} is used by {}; keep its dimensions", users.join(", "))));
    }
    let units = units.map(str::trim).filter(|u| !u.is_empty()).map(str::to_string);
    let formula = formula.trim().to_string();
    let text = |n: &str, v: &str| Attr { name: n.into(), value: AttrValue::Text(v.into()) };
    let mut attrs = vec![text("long_name", name)];
    if let Some(u) = &units {
        attrs.push(text("units", u));
    }
    attrs.push(text("formula", &formula));
    attrs.push(text("source", "calculated in TorCDF"));
    if let Some(c) = widest.attrs.iter().find(|a| a.name == "coordinates") {
        attrs.push(c.clone());
    }
    let info = VarInfo {
        name: name.to_string(),
        path: path.clone(),
        dims: widest.dims.clone(),
        shape: widest.shape.clone(),
        dtype: DType::F32,
        attrs,
        storage: widest.storage.clone(),
    };
    let inputs = inputs.iter().map(|v| v.path.clone()).collect();
    Ok(Derived { name: name.to_string(), path, formula, units, info, expr, inputs })
}

/// `e` with names replaced by paths (and `pi` by its value); collects the
/// variables used.
fn resolve<'a>(
    e: Expr,
    known: &Known<'_>,
    all: &[&'a VarInfo],
    banned: &[String],
    inputs: &mut Vec<&'a VarInfo>,
) -> Result<Expr, CalcError> {
    let go = |e: Expr, inputs: &mut Vec<&'a VarInfo>| resolve(e, known, all, banned, inputs);
    Ok(match e {
        Expr::Num(v) => Expr::Num(v),
        Expr::Neg(a) => Expr::Neg(Box::new(go(*a, inputs)?)),
        Expr::Not(a) => Expr::Not(Box::new(go(*a, inputs)?)),
        Expr::Bin(op, a, b) => Expr::Bin(op, Box::new(go(*a, inputs)?), Box::new(go(*b, inputs)?)),
        Expr::Call(f, args) => {
            let args = args.into_iter().map(|a| go(a, inputs)).collect::<Result<Vec<_>, _>>()?;
            Expr::Call(f, args)
        }
        Expr::Var(name) => {
            let matches: Vec<&'a VarInfo> = if name.starts_with('/') {
                all.iter().copied().filter(|v| v.path == name).collect()
            } else {
                all.iter().copied().filter(|v| v.name == name).collect()
            };
            let v = match matches[..] {
                [] if name == "pi" => return Ok(Expr::Num(std::f64::consts::PI)),
                [] if known.pairs.iter().any(|p| p.path == name || p.name == name) => {
                    return Err(CalcError::msg(format!("{name} is an arrow pair; use its components")));
                }
                [] => return Err(CalcError::msg(format!("no variable called {name}"))),
                [v] => v,
                [a, b, ..] => {
                    return Err(CalcError::msg(format!("{name} is in {} and {}; write `{}`", a.path, b.path, a.path)));
                }
            };
            if banned.contains(&v.path) {
                return Err(CalcError::msg("a variable cannot use itself"));
            }
            if !v.dtype.is_numeric() {
                return Err(CalcError::msg(format!("{name} is not numeric")));
            }
            if !inputs.iter().any(|i| i.path == v.path) {
                inputs.push(v);
            }
            Expr::Var(v.path.clone())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calc::BinOp;
    use crate::fixtures;
    use crate::{Attr, AttrValue, DType, Dataset, Dim, FileFormat, Group, NcFile, Storage, VarInfo};
    use std::f64::consts::PI;

    fn nc4() -> (tempfile::TempDir, NcFile) {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
        (d, f)
    }

    /// A root-only dataset: (name, dtype, [(dim, len)]).
    #[allow(clippy::type_complexity)]
    fn ds(vars: &[(&str, DType, &[(&str, usize)])]) -> Dataset {
        let mut dims: Vec<Dim> = Vec::new();
        let vars = vars
            .iter()
            .map(|(name, dtype, vdims)| {
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
                    dtype: dtype.clone(),
                    attrs: vec![Attr { name: "units".into(), value: AttrValue::Text("m".into()) }],
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

    fn check_in(ds: &Dataset, name: &str, formula: &str) -> Result<Derived, CalcError> {
        check(&Known { ds, derived: &[], pairs: &[] }, name, formula, None, None, &[])
    }

    fn attr<'a>(v: &'a VarInfo, n: &str) -> Option<&'a str> {
        v.attrs.iter().find(|a| a.name == n).and_then(|a| a.value.as_str())
    }

    #[test]
    fn speed_from_components() {
        let (_d, f) = nc4();
        let d = f.check_formula("speed", "sqrt(uo^2+vo^2)", Some("m s-1"), None).unwrap();
        assert_eq!(d.path, "/ocean/speed");
        assert_eq!(d.info.dims, ["time", "lat", "lon"]);
        assert_eq!(d.info.shape, [24, 10, 20]);
        assert_eq!(d.info.dtype, DType::F32);
        assert_eq!(attr(&d.info, "formula"), Some("sqrt(uo^2+vo^2)"));
        assert_eq!(attr(&d.info, "units"), Some("m s-1"));
        assert_eq!(attr(&d.info, "long_name"), Some("speed"));
        assert_eq!(attr(&d.info, "source"), Some("calculated in TorCDF"));
        assert_eq!(d.inputs, ["/ocean/uo", "/ocean/vo"]);
        assert_eq!(d.units.as_deref(), Some("m s-1"));
        // No units typed: no units attribute.
        let d = f.check_formula("s2", "uo*2", Some("  "), None).unwrap();
        assert_eq!(d.units, None);
        assert_eq!(attr(&d.info, "units"), None);
    }

    #[test]
    fn broadcast_subset_dims() {
        let (_d, f) = nc4();
        let d = f.check_formula("ul", "lat * uo", None, None).unwrap();
        assert_eq!(d.info.dims, ["time", "lat", "lon"]);
        assert_eq!(d.inputs, ["/ocean/lat", "/ocean/uo"]);
        let m = ds(&[
            ("u", DType::F32, &[("time", 3), ("lat", 4), ("lon", 5)]),
            ("h", DType::F32, &[("depth", 2), ("lat", 4), ("lon", 5)]),
        ]);
        let e = check_in(&m, "x", "u + h").unwrap_err();
        assert_eq!(e.msg, "u and h have different dimensions: (time, lat, lon) vs (depth, lat, lon)");
        // Out of order is refused too.
        let m = ds(&[("u", DType::F32, &[("lat", 4), ("lon", 5)]), ("w", DType::F32, &[("lon", 5), ("lat", 4)])]);
        assert!(check_in(&m, "x", "u + w").is_err());
    }

    #[test]
    fn names() {
        let (_d, f) = nc4();
        assert_eq!(f.check_formula("uo", "vo", None, None).unwrap_err().msg, "uo already exists");
        assert_eq!(
            f.check_formula("2x", "vo", None, None).unwrap_err().msg,
            "the name must start with a letter and use only letters, digits and _"
        );
        assert_eq!(f.check_formula("x", "q + 1", None, None).unwrap_err().msg, "no variable called q");
        // Two groups with the same leaf name.
        let mut m = ds(&[("uo", DType::F32, &[("y", 2)])]);
        let mut sub = m.root.clone();
        sub.name = "a".into();
        sub.path = "/a".into();
        sub.vars[0].path = "/a/uo".into();
        m.root.groups.push(sub);
        let e = check_in(&m, "x", "uo").unwrap_err();
        assert_eq!(e.msg, "uo is in /uo and /a/uo; write `/uo`");
        assert_eq!(check_in(&m, "x", "`/a/uo` * 2").unwrap().path, "/a/x");
    }

    #[test]
    fn pi_constant_or_variable() {
        let (_d, f) = nc4();
        let d = f.check_formula("x", "uo*pi", None, None).unwrap();
        assert_eq!(d.expr, Expr::Bin(BinOp::Mul, Box::new(Expr::Var("/ocean/uo".into())), Box::new(Expr::Num(PI))));
        let m = ds(&[("pi", DType::F32, &[("y", 2)])]);
        assert_eq!(check_in(&m, "x", "pi").unwrap().expr, Expr::Var("/pi".into()));
    }

    #[test]
    fn no_variables() {
        let (_d, f) = nc4();
        assert_eq!(f.check_formula("x", "1+2", None, None).unwrap_err().msg, "a formula needs at least one variable");
    }

    #[test]
    fn non_numeric_refused() {
        let m = ds(&[("label", DType::Char, &[("n", 4)]), ("u", DType::F32, &[("n", 4)])]);
        assert_eq!(check_in(&m, "x", "label + u").unwrap_err().msg, "label is not numeric");
        let d = tempfile::tempdir().unwrap();
        let f = fixtures::paired(NcFile::open(&fixtures::write_nc4(d.path())).unwrap());
        assert_eq!(
            f.check_formula("x", "`/ocean/uo+vo` * 2", None, None).unwrap_err().msg,
            "/ocean/uo+vo is an arrow pair; use its components"
        );
    }

    #[test]
    fn registry() {
        let (_d, f) = nc4();
        let first = f.add_derived(f.check_formula("first", "uo*2", None, None).unwrap());
        assert_eq!(f.derived(&first.path).unwrap().name, "first");
        let second = f.add_derived(f.check_formula("second", "first + vo", None, None).unwrap());
        assert_eq!(second.inputs, ["/ocean/first", "/ocean/vo"]);
        assert_eq!(f.derived_vars().iter().map(|d| d.name.clone()).collect::<Vec<_>>(), ["first", "second"]);
        assert_eq!(f.users_of(&first.path), ["second"]);
        assert_eq!(f.remove_derived(&first.path).unwrap_err(), "first is used by second");
        // Editing while used: the dimensions must stay.
        let e = f.check_formula("first", "lat*2", None, Some(&first.path)).unwrap_err();
        assert_eq!(e.msg, "first is used by second; keep its dimensions");
        let edited = f.check_formula("first", "uo*3", None, Some(&first.path)).unwrap();
        f.add_derived(edited);
        assert_eq!(f.derived_vars()[0].formula, "uo*3");
        assert_eq!(f.derived_vars().len(), 2);
        f.remove_derived(&second.path).unwrap();
        f.remove_derived(&first.path).unwrap();
        assert!(f.derived_vars().is_empty());
    }

    #[test]
    fn self_reference_refused() {
        let (_d, f) = nc4();
        let s = f.add_derived(f.check_formula("speed", "uo*2", None, None).unwrap());
        let e = f.check_formula("speed", "speed*2", None, Some(&s.path)).unwrap_err();
        assert_eq!(e.msg, "a variable cannot use itself");
        // A loop through another variable is refused too.
        f.add_derived(f.check_formula("twice", "speed*2", None, None).unwrap());
        assert_eq!(f.check_formula("speed", "twice+1", None, Some(&s.path)).unwrap_err().msg, "a variable cannot use itself");
        assert_eq!(f.affected_by(&s.path), ["/ocean/speed", "/ocean/twice"]);
    }

    #[test]
    fn meta_with_derived_lists_it() {
        let (_d, f) = nc4();
        assert!(matches!(f.meta_with_derived(), std::borrow::Cow::Borrowed(_)));
        f.add_derived(f.check_formula("speed", "uo*2", None, None).unwrap());
        let m = f.meta_with_derived();
        assert!(m.var("/ocean/speed").is_some());
        assert_eq!(m.group_of("/ocean/speed").path, "/ocean");
        assert_eq!(f.var_info_any("/ocean/speed").unwrap().name, "speed");
        assert_eq!(f.var_info_any("/ocean/uo").unwrap().name, "uo");
    }
}
