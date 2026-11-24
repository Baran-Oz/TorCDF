//! Which variables and dimensions an extract writes (metadata only).

use super::{DimKey, Plan, PlanVar, Selection};
use crate::{DType, Group, NcFile, VarInfo};

/// The group at `path` within `root`.
pub(crate) fn find_group<'a>(root: &'a Group, path: &str) -> Option<&'a Group> {
    if root.path == path {
        return Some(root);
    }
    root.groups.iter().find_map(|g| find_group(g, path))
}

/// The parent path of a variable or group path (`"/"` for top level).
pub(crate) fn parent_of(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) | None => "/",
        Some(i) => &path[..i],
    }
}

/// Names listed in an attribute such as `coordinates` or `grid_mapping`
/// (the extended `grid_mapping` form "crs: lat lon" gives "crs").
fn names_in(text: &str, grid_mapping: bool) -> Vec<String> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if grid_mapping && tokens.iter().any(|t| t.ends_with(':')) {
        return tokens.iter().filter_map(|t| t.strip_suffix(':')).map(str::to_string).collect();
    }
    tokens.into_iter().map(str::to_string).collect()
}

/// Every dimension of `g` and its subgroups, in file order.
fn all_dims(g: &Group, out: &mut Vec<DimKey>) {
    out.extend(g.dims.iter().map(|d| DimKey { group: g.path.clone(), name: d.name.clone() }));
    for s in &g.groups {
        all_dims(s, out);
    }
}

impl NcFile {
    /// Whether the file is read through netCDF-C (Extract needs it).
    pub fn is_netcdf(&self) -> bool {
        !self.backend_is_hdf5()
    }

    /// The dimension `dim` as used by the variable at `var_path`: the
    /// nearest enclosing group that defines it.
    pub fn dim_key(&self, var_path: &str, dim: &str) -> Option<DimKey> {
        let mut g = parent_of(var_path);
        loop {
            if let Some(grp) = find_group(&self.meta.root, g)
                && grp.dims.iter().any(|d| d.name == dim)
            {
                return Some(DimKey { group: g.to_string(), name: dim.to_string() });
            }
            if g == "/" {
                return None;
            }
            g = parent_of(g);
        }
    }

    /// The file variable named `name` seen from the variable at `from`: in
    /// its group or the nearest ancestor holding one.
    fn var_by_name(&self, from: &str, name: &str) -> Option<&VarInfo> {
        if name.starts_with('/') {
            return self.meta.var(name);
        }
        let mut g = parent_of(from);
        loop {
            if let Some(v) = find_group(&self.meta.root, g).and_then(|grp| grp.vars.iter().find(|v| v.name == name)) {
                return Some(v);
            }
            if g == "/" {
                return None;
            }
            g = parent_of(g);
        }
    }

    /// The variables, dimensions and size `sel` saves.
    pub fn plan_extract(&self, sel: &Selection) -> Result<Plan, String> {
        const NONE: &str = "the selection holds no cells";
        if !self.is_netcdf() {
            return Err("Extract works on netCDF files".into());
        }
        if sel.vars.is_empty() {
            return Err("no variable is ticked".into());
        }
        if sel.ranges.values().any(|&(_, n)| n == 0) {
            return Err(NONE.into());
        }
        let mut queue: Vec<String> = sel.vars.clone();
        let mut vars: Vec<PlanVar> = Vec::new();
        let mut skipped = Vec::new();
        let mut bytes = 0u64;
        let mut i = 0;
        while i < queue.len() {
            let path = queue[i].clone();
            let ticked = i < sel.vars.len();
            i += 1;
            let derived = if self.meta.var(&path).is_none() { self.derived(&path) } else { None };
            let info = match (self.meta.var(&path), &derived) {
                (Some(v), _) => v.clone(),
                (None, Some(d)) => d.info.clone(),
                (None, None) => return Err(format!("{path} is not in the file")),
            };
            let elem: u64 = match (&derived, &info.dtype) {
                (Some(_), _) => 4,
                (None, DType::Other(kind)) => {
                    skipped.push(format!("{} ({kind} type)", info.name));
                    continue;
                }
                (None, DType::Char) => 1,
                (None, DType::Str) => 8,
                (None, t) => t.size().unwrap_or(8) as u64,
            };
            let mut dims = Vec::new();
            let (mut start, mut count) = (Vec::new(), Vec::new());
            for (name, &len) in info.dims.iter().zip(&info.shape) {
                let key = self.dim_key(&path, name).ok_or_else(|| format!("dimension {name} of {path} not found"))?;
                let (s, n) = sel.ranges.get(&key).copied().unwrap_or((0, len));
                if s + n > len {
                    return Err(format!("the range of {name} is outside the file"));
                }
                // The dimension's coordinate variable.
                if let Some(c) = find_group(&self.meta.root, &key.group)
                    .and_then(|g| g.vars.iter().find(|v| v.name == *name && v.dims.len() == 1 && v.dims[0] == *name))
                    && !queue.contains(&c.path)
                {
                    queue.push(c.path.clone());
                }
                dims.push(key);
                start.push(s);
                count.push(n);
            }
            if ticked && count.contains(&0) {
                return Err(NONE.into());
            }
            for (attr, gm) in [("coordinates", false), ("bounds", false), ("grid_mapping", true)] {
                let Some(text) = info.attrs.iter().find(|a| a.name == attr).and_then(|a| a.value.as_str()) else {
                    continue;
                };
                for name in names_in(text, gm) {
                    if let Some(v) = self.var_by_name(&path, &name)
                        && !queue.contains(&v.path)
                    {
                        queue.push(v.path.clone());
                    }
                }
            }
            bytes += count.iter().map(|&n| n as u64).product::<u64>() * elem;
            vars.push(PlanVar { path, derived, dims, start, count });
        }
        // Written dimensions, in file order.
        let mut order = Vec::new();
        all_dims(&self.meta.root, &mut order);
        let mut dims = Vec::new();
        for k in order {
            let used = vars.iter().find_map(|v| v.dims.iter().position(|d| *d == k).map(|j| v.count[j]));
            if let Some(n) = used {
                let unlimited = find_group(&self.meta.root, &k.group)
                    .and_then(|g| g.dims.iter().find(|d| d.name == k.name))
                    .is_some_and(|d| d.unlimited);
                dims.push((k, n, unlimited));
            }
        }
        Ok(Plan { vars, dims, skipped, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    pub(crate) fn key(group: &str, name: &str) -> DimKey {
        DimKey { group: group.into(), name: name.into() }
    }

    #[test]
    fn plan_includes_coordinates() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        assert!(nc.is_netcdf());
        assert_eq!(nc.dim_key("/ocean/uo", "lat"), Some(key("/ocean", "lat")));
        let mut sel = Selection { vars: vec!["/ocean/uo".into()], ..Default::default() };
        sel.ranges.insert(key("/ocean", "time"), (3, 5));
        sel.ranges.insert(key("/ocean", "lat"), (2, 4));
        sel.ranges.insert(key("/ocean", "lon"), (5, 6));
        let p = nc.plan_extract(&sel).unwrap();
        let mut paths: Vec<&str> = p.vars.iter().map(|v| v.path.as_str()).collect();
        paths.sort();
        assert_eq!(paths, ["/ocean/lat", "/ocean/lon", "/ocean/time", "/ocean/uo"]);
        let uo = p.vars.iter().find(|v| v.path == "/ocean/uo").unwrap();
        assert_eq!((uo.start.clone(), uo.count.clone()), (vec![3, 2, 5], vec![5, 4, 6]));
        let lens: Vec<(String, usize)> = p.dims.iter().map(|(k, n, _)| (k.name.clone(), *n)).collect();
        assert_eq!(lens, [("time".to_string(), 5), ("lat".into(), 4), ("lon".into(), 6)]);
        assert_eq!(p.bytes, (5 * 4 * 6 * 4 + 5 * 8 + 4 * 8 + 6 * 8) as u64);
        assert!(p.skipped.is_empty());

        let nc = NcFile::open(&fixtures::write_curvilinear(dir.path())).unwrap();
        let p = nc.plan_extract(&Selection { vars: vec!["/sst".into()], ..Default::default() }).unwrap();
        let mut paths: Vec<&str> = p.vars.iter().map(|v| v.path.as_str()).collect();
        paths.sort();
        assert_eq!(paths, ["/nav_lat", "/nav_lon", "/sst"]);
    }

    #[test]
    fn plan_errors() {
        let dir = tempfile::tempdir().unwrap();
        let nc = NcFile::open(&fixtures::write_nc4(dir.path())).unwrap();
        assert_eq!(nc.plan_extract(&Selection::default()).unwrap_err(), "no variable is ticked");
        let mut sel = Selection { vars: vec!["/ocean/uo".into()], ..Default::default() };
        sel.ranges.insert(key("/ocean", "lat"), (2, 0));
        assert_eq!(nc.plan_extract(&sel).unwrap_err(), "the selection holds no cells");

        let h5 = NcFile::open(&fixtures::hdf5::write_hidden_h5(dir.path())).unwrap();
        assert!(!h5.is_netcdf());
        let var = h5.meta.all_vars()[0].path.clone();
        let err = h5.plan_extract(&Selection { vars: vec![var], ..Default::default() }).unwrap_err();
        assert_eq!(err, "Extract works on netCDF files");
    }
}
