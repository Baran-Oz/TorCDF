//! Pure helpers that turn dates, values and lat/lon boxes into index ranges.

use crate::cf::TimeUnits;

pub const SEAM: &str = "the region crosses the grid's longitude seam (0°/360°); save each side separately";

/// "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" (also "T" separator) → value in `t`'s units.
pub fn parse_date(text: &str, t: &TimeUnits) -> Option<f64> {
    let text = text.trim();
    let (date, time) = match text.find(['T', 't', ' ']) {
        Some(i) => (&text[..i], text[i + 1..].trim()),
        None => (text, ""),
    };
    let mut dp = date.split('-');
    let y: i32 = dp.next()?.trim().parse().ok()?;
    let m: u8 = dp.next()?.trim().parse().ok()?;
    let d: u8 = dp.next()?.trim().parse().ok()?;
    if dp.next().is_some() {
        return None;
    }
    let (h, mi) = if time.is_empty() {
        (0, 0)
    } else {
        let (h, mi) = time.split_once(':')?;
        (h.trim().parse::<u8>().ok()?, mi.trim().parse::<u8>().ok()?)
    };
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || h > 23 || mi > 59 {
        return None;
    }
    let v = t.encode(y, m, d, h, mi);
    // Days past the month's end (Feb 30 outside 360-day) do not round-trip.
    let back = t.decode(v);
    (back.year == y && back.month == m && back.day == d).then_some(v)
}

/// Comparison slack for float32 coordinates against typed bounds.
fn slack(lo: f64, hi: f64) -> f64 {
    1e-6 * lo.abs().max(hi.abs()).max(1.0)
}

/// (start, count) of the positions whose value lies in [lo, hi] (lo/hi in either order):
/// from the first to the last matching position (so descending or non-monotonic axes work).
pub fn value_range(values: &[f64], lo: f64, hi: f64) -> Option<(usize, usize)> {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    let e = slack(lo, hi);
    let inside = |v: &f64| *v >= lo - e && *v <= hi + e;
    let first = values.iter().position(inside)?;
    let last = values.iter().rposition(inside)?;
    Some((first, last - first + 1))
}

/// (start, count) of a 1-D longitude axis inside [west, east] degrees. The box is
/// first moved into the axis's convention (0..360 or -180..180).
/// Err(SEAM) when the box needs both ends of the axis.
pub fn lon_range(lons: &[f64], west: f64, east: f64) -> Result<(usize, usize), String> {
    const NONE: &str = "the selection holds no cells";
    if east - west >= 360.0 {
        return value_range(lons, f64::NEG_INFINITY, f64::INFINITY).ok_or_else(|| NONE.to_string());
    }
    let max = lons.iter().copied().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
    let base = if max > 180.0 { 0.0 } else { -180.0 };
    let wrap = |x: f64| (x - base).rem_euclid(360.0) + base;
    let w = wrap(west);
    let mut e = wrap(east);
    if e == base && east != west {
        // East typed on the convention's upper end (180 or 360).
        e = base + 360.0;
    }
    if w <= e {
        return value_range(lons, w, e).ok_or_else(|| NONE.to_string());
    }
    // The box wraps past the axis's end: fine only when one side is empty.
    match (value_range(lons, w, base + 360.0), value_range(lons, base, e)) {
        (Some(_), Some(_)) => Err(SEAM.to_string()),
        (Some(r), None) | (None, Some(r)) => Ok(r),
        (None, None) => Err(NONE.to_string()),
    }
}

/// The smallest (y, x) index rectangle holding every cell of a 2-D lat/lon grid inside the box:
/// [(y_start, y_count), (x_start, x_count)].
#[allow(clippy::too_many_arguments)]
pub fn curv_box(
    lat: &[f64],
    lon: &[f64],
    ny: usize,
    nx: usize,
    south: f64,
    north: f64,
    west: f64,
    east: f64,
) -> Option<[(usize, usize); 2]> {
    let (south, north) = if south <= north { (south, north) } else { (north, south) };
    let e = slack(south, north);
    let width = if east - west >= 360.0 { 360.0 } else { (east - west).rem_euclid(360.0) };
    let (mut y0, mut y1, mut x0, mut x1) = (usize::MAX, 0, usize::MAX, 0);
    for y in 0..ny {
        for x in 0..nx {
            let i = y * nx + x;
            let (Some(&la), Some(&lo)) = (lat.get(i), lon.get(i)) else { continue };
            if !(la >= south - e && la <= north + e) || !lo.is_finite() {
                continue;
            }
            let off = (lo - west).rem_euclid(360.0);
            if off > width + 1e-6 && off < 360.0 - 1e-6 {
                continue;
            }
            y0 = y0.min(y);
            y1 = y1.max(y);
            x0 = x0.min(x);
            x1 = x1.max(x);
        }
    }
    (y0 != usize::MAX).then(|| [(y0, y1 - y0 + 1), (x0, x1 - x0 + 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tu(units: &str, cal: Option<&str>) -> TimeUnits {
        TimeUnits::parse(units, cal).unwrap()
    }

    #[test]
    fn encode_inverts_decode() {
        for (units, cal) in [
            ("hours since 2000-01-01", None),
            ("days since 2000-01-01", Some("noleap")),
            ("days since 2000-01-01", Some("360_day")),
        ] {
            let t = tu(units, cal);
            assert_eq!(t.decode(t.encode(2003, 2, 28, 6, 30)).to_string(), "2003-02-28 06:30", "{units} {cal:?}");
        }
    }

    #[test]
    fn parse_dates() {
        let t = tu("hours since 2000-01-01", None);
        assert_eq!(parse_date("2000-01-02", &t), Some(24.0));
        assert_eq!(parse_date("2000-01-02 06:00", &t), Some(30.0));
        assert_eq!(parse_date("2000-01-02T06:00", &t), Some(30.0));
        assert_eq!(parse_date("2000-13-01", &t), None);
        assert_eq!(parse_date("x", &t), None);
    }

    #[test]
    fn values_any_order() {
        assert_eq!(value_range(&[0., 1., 2., 3.], 1., 2.), Some((1, 2)));
        assert_eq!(value_range(&[3., 2., 1., 0.], 1., 2.), Some((1, 2)));
        assert_eq!(value_range(&[0., 1., 2., 3.], 2., 1.), Some((1, 2)));
        assert_eq!(value_range(&[0., 1., 2., 3.], 5., 6.), None);
        assert_eq!(value_range(&[0., 5., 1., 4.], 1., 4.), Some((2, 2)));
    }

    #[test]
    fn lon_conventions() {
        let east: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        assert_eq!(lon_range(&east, -20.0, 20.0), Err(SEAM.to_string()));
        assert_eq!(lon_range(&east, 10.0, 30.0), Ok((1, 3)));
        assert_eq!(lon_range(&east, -170.0, -150.0), Ok((19, 3)));
        let centred: Vec<f64> = (0..36).map(|i| -180.0 + i as f64 * 10.0).collect();
        assert_eq!(lon_range(&centred, 200.0, 220.0), Ok((2, 3)));
        assert_eq!(lon_range(&centred, 170.0, -170.0), Err(SEAM.to_string()));
    }

    #[test]
    fn curvilinear_box() {
        let lat = [10., 10., 10., 20., 20., 20., 30., 30., 30.];
        let lon = [0., 10., 20., 0., 10., 20., 0., 10., 20.];
        assert_eq!(curv_box(&lat, &lon, 3, 3, 15.0, 30.0, 5.0, 20.0), Some([(1, 2), (1, 2)]));
        assert_eq!(curv_box(&lat, &lon, 3, 3, 50.0, 60.0, 5.0, 20.0), None);
    }
}
