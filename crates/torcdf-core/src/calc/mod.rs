//! Calculated variables: new variables defined by a formula over a file's
//! variables. They are virtual (computed on read, like vector pairs) and
//! last for the session.

pub mod parse;

pub use parse::{BinOp, Expr, Func, parse};

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
