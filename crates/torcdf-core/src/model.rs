//! Plain-data description of a netCDF/HDF5 file's structure.
//!
//! Paths are `/`-separated from the root: root variable `t` is `"/t"`, and
//! `uo` inside group `ocean` is `"/ocean/uo"`.

use std::path::PathBuf;

#[derive(Clone, Debug, thiserror::Error)]
pub enum CoreError {
    #[error("cannot open {}: {msg}", path.display())]
    Open { path: PathBuf, msg: String },
    #[error("cannot read {var}: {msg}")]
    Read { var: String, msg: String },
    #[error("unsupported: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub path: PathBuf,
    pub file_size: u64,
    pub format: FileFormat,
    pub root: Group,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Classic,
    Offset64,
    Cdf5,
    Netcdf4,
    Netcdf4Classic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub name: String,
    pub path: String,
    pub dims: Vec<Dim>,
    pub vars: Vec<VarInfo>,
    pub attrs: Vec<Attr>,
    pub groups: Vec<Group>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dim {
    pub name: String,
    pub len: usize,
    pub unlimited: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarInfo {
    pub name: String,
    pub path: String,
    pub dims: Vec<String>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub attrs: Vec<Attr>,
    pub storage: Storage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Char,
    Str,
    Other(String),
}

impl DType {
    pub fn is_numeric(&self) -> bool {
        !matches!(self, DType::Char | DType::Str | DType::Other(_))
    }

    /// Size in bytes of one element; `None` for non-numeric types.
    pub fn size(&self) -> Option<usize> {
        match self {
            DType::I8 | DType::U8 => Some(1),
            DType::I16 | DType::U16 => Some(2),
            DType::I32 | DType::U32 | DType::F32 => Some(4),
            DType::I64 | DType::U64 | DType::F64 => Some(8),
            DType::Char | DType::Str | DType::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attr {
    pub name: String,
    pub value: AttrValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Text(String),
    Texts(Vec<String>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

impl AttrValue {
    /// Numeric values as `f64`; `None` for text attributes.
    pub fn as_f64s(&self) -> Option<Vec<f64>> {
        match self {
            AttrValue::Ints(v) => Some(v.iter().map(|&i| i as f64).collect()),
            AttrValue::Floats(v) => Some(v.clone()),
            AttrValue::Text(_) | AttrValue::Texts(_) => None,
        }
    }

    /// The string of a single text attribute.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttrValue::Text(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Storage {
    Contiguous,
    Chunked { chunk: Vec<usize>, compressed: bool },
    Compact,
    Unknown,
}

impl Dataset {
    /// Looks up a variable by its full path, e.g. `"/ocean/uo"`.
    pub fn var(&self, path: &str) -> Option<&VarInfo> {
        self.all_vars().into_iter().find(|v| v.path == path)
    }

    /// Every variable in the file, depth first, root group first.
    pub fn all_vars(&self) -> Vec<&VarInfo> {
        fn walk<'a>(g: &'a Group, out: &mut Vec<&'a VarInfo>) {
            out.extend(g.vars.iter());
            for sub in &g.groups {
                walk(sub, out);
            }
        }
        let mut out = Vec::new();
        walk(&self.root, &mut out);
        out
    }

    /// The group containing the variable at `var_path`; the root group if
    /// the path is unknown.
    pub fn group_of(&self, var_path: &str) -> &Group {
        let parent = match var_path.rfind('/') {
            Some(0) | None => "/",
            Some(i) => &var_path[..i],
        };
        fn find<'a>(g: &'a Group, path: &str) -> Option<&'a Group> {
            if g.path == path {
                return Some(g);
            }
            g.groups.iter().find_map(|s| find(s, path))
        }
        find(&self.root, parent).unwrap_or(&self.root)
    }
}
