//! Calculated variables: new variables defined by a formula over a file's
//! variables. They are virtual (computed on read, like vector pairs) and
//! last for the session.

mod check;
pub mod parse;

pub use check::Derived;
pub use parse::{BinOp, Expr, Func, parse};

use crate::{Dataset, Group, NcFile, VarInfo};
use std::borrow::Cow;
use std::sync::{Arc, RwLockReadGuard};

impl NcFile {
    fn derived_read(&self) -> RwLockReadGuard<'_, Vec<Arc<Derived>>> {
        self.derived.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Checks a formula for a new variable `name` (metadata only).
    /// `replacing`: the calculated variable being edited.
    pub fn check_formula(
        &self,
        name: &str,
        formula: &str,
        units: Option<&str>,
        replacing: Option<&str>,
    ) -> Result<Derived, CalcError> {
        let derived = self.derived_vars();
        let pairs = self.pairs();
        let users = replacing.map(|r| self.users_of(r)).unwrap_or_default();
        let known = check::Known { ds: &self.meta, derived: &derived, pairs: &pairs };
        check::check(&known, name, formula, units, replacing, &users)
    }

    /// Registers `d`; one already at its path is replaced in place.
    pub fn add_derived(&self, d: Derived) -> Arc<Derived> {
        let d = Arc::new(d);
        let mut all = self.derived.write().unwrap_or_else(|e| e.into_inner());
        match all.iter_mut().find(|x| x.path == d.path) {
            Some(x) => *x = d.clone(),
            None => all.push(d.clone()),
        }
        d
    }

    /// The calculated variable at `path`.
    pub fn derived(&self, path: &str) -> Option<Arc<Derived>> {
        self.derived_read().iter().find(|d| d.path == path).cloned()
    }

    /// Calculated variables in the order they were made.
    pub fn derived_vars(&self) -> Vec<Arc<Derived>> {
        self.derived_read().clone()
    }

    /// Names of the calculated variables and arrow pairs that use `path`
    /// directly.
    pub fn users_of(&self, path: &str) -> Vec<String> {
        let mut out: Vec<String> =
            self.derived_read().iter().filter(|d| d.inputs.iter().any(|i| i == path)).map(|d| d.name.clone()).collect();
        out.extend(self.pairs().iter().filter(|p| p.east == path || p.north == path).map(|p| p.name.clone()));
        out
    }

    /// Paths whose values change when `path` is edited: `path`, the
    /// calculated variables built on it (at any depth) and the arrow pairs
    /// using any of them.
    pub fn affected_by(&self, path: &str) -> Vec<String> {
        let derived = self.derived_vars();
        let mut out = vec![path.to_string()];
        let mut i = 0;
        while i < out.len() {
            for d in &derived {
                if d.inputs.contains(&out[i]) && !out.contains(&d.path) {
                    out.push(d.path.clone());
                }
            }
            i += 1;
        }
        let pairs: Vec<String> =
            self.pairs().iter().filter(|p| out.contains(&p.east) || out.contains(&p.north)).map(|p| p.path.clone()).collect();
        out.extend(pairs);
        out
    }

    /// Unregisters the calculated variable at `path`; refused while
    /// another calculated variable or an arrow pair uses it.
    pub fn remove_derived(&self, path: &str) -> Result<(), String> {
        let users = self.users_of(path);
        if !users.is_empty() {
            let name = self.derived(path).map_or_else(|| path.to_string(), |d| d.name.clone());
            return Err(format!("{name} is used by {}", users.join(", ")));
        }
        self.derived.write().unwrap_or_else(|e| e.into_inner()).retain(|d| d.path != path);
        Ok(())
    }

    /// A file variable, a calculated variable or an arrow pair at `path`.
    pub fn var_info_any(&self, path: &str) -> Option<VarInfo> {
        if let Some(v) = self.meta.var(path) {
            return Some(v.clone());
        }
        if let Some(d) = self.derived(path) {
            return Some(d.info.clone());
        }
        self.pair(path).map(|p| p.var_info())
    }

    /// The file's metadata with calculated variables listed in their
    /// groups (a copy only when there are any).
    pub fn meta_with_derived(&self) -> Cow<'_, Dataset> {
        let derived = self.derived_read();
        if derived.is_empty() {
            return Cow::Borrowed(&self.meta);
        }
        fn group_mut<'a>(g: &'a mut Group, path: &str) -> Option<&'a mut Group> {
            if g.path == path {
                return Some(g);
            }
            g.groups.iter_mut().find_map(|s| group_mut(s, path))
        }
        let mut ds = self.meta.clone();
        for d in derived.iter() {
            let parent = match d.path.rfind('/') {
                Some(0) | None => "/".to_string(),
                Some(i) => d.path[..i].to_string(),
            };
            if let Some(g) = group_mut(&mut ds.root, &parent) {
                g.vars.push(d.info.clone());
            }
        }
        Cow::Owned(ds)
    }
}

/// A formula problem, with the byte position it was found at when known.
#[derive(Debug, Clone, PartialEq)]
pub struct CalcError {
    pub pos: Option<usize>,
    pub msg: String,
}

impl CalcError {
    pub fn at(pos: usize, msg: impl Into<String>) -> CalcError {
        CalcError { pos: Some(pos), msg: msg.into() }
    }

    pub fn msg(msg: impl Into<String>) -> CalcError {
        CalcError { pos: None, msg: msg.into() }
    }
}

impl std::fmt::Display for CalcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.pos {
            Some(p) => write!(f, "{} (at character {})", self.msg, p + 1),
            None => f.write_str(&self.msg),
        }
    }
}

/// Every function a formula can use, with its one-line help.
pub const FUNCTIONS: &[(&str, &str)] = &[
    ("sqrt", "sqrt(a): square root (missing below 0)"),
    ("abs", "abs(a): absolute value"),
    ("exp", "exp(a): e to the power a"),
    ("log", "log(a): natural logarithm (missing at 0 or below)"),
    ("log10", "log10(a): base-10 logarithm (missing at 0 or below)"),
    ("sin", "sin(a): sine of a in radians"),
    ("cos", "cos(a): cosine of a in radians"),
    ("tan", "tan(a): tangent of a in radians"),
    ("asin", "asin(a): arcsine, in radians"),
    ("acos", "acos(a): arccosine, in radians"),
    ("atan", "atan(a): arctangent, in radians"),
    ("atan2", "atan2(y, x): angle of (x, y) in radians"),
    ("deg", "deg(a): radians to degrees"),
    ("rad", "rad(a): degrees to radians"),
    ("min", "min(a, b): the smaller of a and b"),
    ("max", "max(a, b): the larger of a and b"),
    ("hypot", "hypot(a, b): sqrt(a^2 + b^2)"),
    ("floor", "floor(a): round down"),
    ("ceil", "ceil(a): round up"),
    ("round", "round(a): round to the nearest whole number"),
    ("where", "where(c, a, b): a where c is true (not 0), else b"),
];
