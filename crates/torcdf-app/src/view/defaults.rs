//! Default view, axes, request and value range for a variable (spec 6.3).

use torcdf_core::cf::{Geo, detect_geo};
use torcdf_core::{Dataset, Slice, SliceRequest, VarInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewKind {
    Image,
    Line,
    Table,
}

/// Dimension indices plotted; every other dimension gets a slider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisChoice {
    pub y: usize,
    pub x: usize,
}

pub const OVERVIEW_TARGET: usize = 2048;

pub fn choose_view(_ds: &Dataset, v: &VarInfo) -> ViewKind {
    if !v.dtype.is_numeric() {
        return ViewKind::Table;
    }
    match v.shape.iter().filter(|&&n| n > 1).count() {
        0 => ViewKind::Table,
        1 => ViewKind::Line,
        _ => ViewKind::Image,
    }
}

/// Lat/lon dimensions when the variable is geographic (y = lat), otherwise
/// the last two dimensions of length > 1.
pub fn choose_axes(ds: &Dataset, v: &VarInfo) -> Option<AxisChoice> {
    match detect_geo(ds, v) {
        Geo::Regular { lat_dim, lon_dim, .. } => {
            return Some(AxisChoice { y: lat_dim, x: lon_dim });
        }
        Geo::Curvilinear { y_dim, x_dim, .. } => {
            return Some(AxisChoice { y: y_dim, x: x_dim });
        }
        Geo::None => {}
    }
    let mut big = v.shape.iter().enumerate().filter(|&(_, &n)| n > 1).map(|(i, _)| i).rev();
    let x = big.next()?;
    let y = big.next()?;
    Some(AxisChoice { y, x })
}

/// `ceil(len / target)`, at least 1.
pub fn overview_stride(len: usize, target: usize) -> usize {
    len.div_ceil(target.max(1)).max(1)
}

/// Full extent (thinned to the overview size) on the plotted axes; one index
/// from `slider_pos` on every other dimension.
pub fn build_request(v: &VarInfo, axes: &AxisChoice, slider_pos: &[usize]) -> SliceRequest {
    let n = v.shape.len();
    let (mut start, mut count, mut stride) = (vec![0; n], vec![1; n], vec![1; n]);
    for d in 0..n {
        if d == axes.y || d == axes.x {
            count[d] = v.shape[d];
            stride[d] = overview_stride(v.shape[d], OVERVIEW_TARGET);
        } else {
            start[d] = slider_pos.get(d).copied().unwrap_or(0);
        }
    }
    SliceRequest { var: v.path.clone(), start, count, stride }
}

/// 2nd to 98th percentile of the finite values; `None` if there are none.
/// A constant field is widened to a total width of 1.0.
pub fn auto_range(s: &Slice) -> Option<(f32, f32)> {
    let mut vals: Vec<f32> = s.data.iter().copied().filter(|x| x.is_finite()).collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_unstable_by(f32::total_cmp);
    let at = |p: f64| vals[((vals.len() - 1) as f64 * p).round() as usize];
    let (lo, hi) = (at(0.02), at(0.98));
    if lo == hi {
        Some((lo - 0.5, hi + 0.5))
    } else {
        Some((lo, hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torcdf_core::{NcFile, fixtures};

    fn open(f: impl Fn(&std::path::Path) -> std::path::PathBuf) -> Dataset {
        let d = tempfile::tempdir().unwrap();
        NcFile::open(&f(d.path())).unwrap().meta.clone()
    }

    fn slice_of(v: &[f32]) -> Slice {
        Slice {
            shape: vec![v.len()],
            data: v.to_vec(),
            valid_count: v.iter().filter(|x| !x.is_nan()).count(),
        }
    }

    #[test]
    fn geo_var_is_image_on_lat_lon() {
        let classic = open(fixtures::write_classic);
        let t = classic.var("/t").unwrap();
        assert_eq!(choose_view(&classic, t), ViewKind::Image);
        assert_eq!(choose_axes(&classic, t), Some(AxisChoice { y: 1, x: 2 }));
    }

    #[test]
    fn curvilinear_uses_its_dims() {
        let ds = open(fixtures::write_curvilinear);
        let sst = ds.var("/sst").unwrap();
        assert_eq!(choose_axes(&ds, sst), Some(AxisChoice { y: 0, x: 1 }));
    }

    #[test]
    fn no_conventions_defaults_to_image_with_index_axes() {
        let bare = open(fixtures::write_bare);
        let data = bare.var("/data").unwrap();
        assert_eq!(choose_view(&bare, data), ViewKind::Image);
        assert_eq!(choose_axes(&bare, data), Some(AxisChoice { y: 0, x: 1 }));
    }

    #[test]
    fn one_dim_is_line() {
        let classic = open(fixtures::write_classic);
        let lat = classic.var("/lat").unwrap();
        assert_eq!(choose_view(&classic, lat), ViewKind::Line);
        assert_eq!(choose_axes(&classic, lat), None);
    }

    #[test]
    fn string_is_table() {
        let nc4 = open(fixtures::write_nc4);
        let label = nc4.var("/ocean/label").unwrap();
        assert_eq!(choose_view(&nc4, label), ViewKind::Table);
    }

    #[test]
    fn gebco_stride() {
        assert_eq!(overview_stride(86400, OVERVIEW_TARGET), 43);
        assert_eq!(overview_stride(100, OVERVIEW_TARGET), 1);
        assert_eq!(overview_stride(0, OVERVIEW_TARGET), 1);
    }

    #[test]
    fn request_for_slider() {
        let classic = open(fixtures::write_classic);
        let t = classic.var("/t").unwrap();
        assert_eq!(
            build_request(t, &AxisChoice { y: 1, x: 2 }, &[2, 0, 0]),
            SliceRequest {
                var: "/t".into(),
                start: vec![2, 0, 0],
                count: vec![1, 4, 8],
                stride: vec![1, 1, 1]
            }
        );
    }

    #[test]
    fn request_strides_large_axes() {
        let mut v = open(fixtures::write_bare).var("/data").unwrap().clone();
        v.shape = vec![86400, 43200];
        let r = build_request(&v, &AxisChoice { y: 0, x: 1 }, &[5, 5]);
        assert_eq!(r.start, vec![0, 0]);
        assert_eq!(r.count, vec![86400, 43200]);
        assert_eq!(r.stride, vec![43, 22]);
    }

    #[test]
    fn auto_range_all_nan_is_none() {
        let s = Slice { shape: vec![2], data: vec![f32::NAN; 2], valid_count: 0 };
        assert_eq!(auto_range(&s), None);
    }

    #[test]
    fn auto_range_constant_widens() {
        assert_eq!(auto_range(&slice_of(&[5.0, 5.0])), Some((4.5, 5.5)));
    }

    #[test]
    fn auto_range_clips_outliers_and_ignores_nan() {
        let mut v: Vec<f32> = (0..=100).map(|i| i as f32).collect();
        v.push(f32::NAN);
        v[100] = 1.0e9;
        let (lo, hi) = auto_range(&slice_of(&v)).unwrap();
        assert_eq!(lo, 2.0);
        assert!(hi < 100.0 && hi > 90.0, "hi = {hi}");
    }
}
