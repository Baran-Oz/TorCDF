//! Running statistics (counts, extremes, weighted mean and spread) and
//! percentile helpers. Pure: no file access.

/// Percentiles shown: P5, P25, median, P75, P95.
pub const PERCENTILES: [f64; 5] = [5.0, 25.0, 50.0, 75.0, 95.0];
/// Bars of the histogram shown.
pub const SHOW_BINS: usize = 50;

/// Counts, extremes with their positions, and a weighted mean and
/// variance by West's (1979) update of Welford's method.
#[derive(Clone, Debug, Default)]
pub struct Acc {
    valid: u64,
    missing: u64,
    min: Option<(f64, u64)>,
    max: Option<(f64, u64)>,
    wsum: f64,
    mean: f64,
    m2: f64,
}

impl Acc {
    /// A valid value `v` with weight `w` (> 0) at flat position `pos`.
    pub fn push(&mut self, v: f32, w: f64, pos: u64) {
        let x = f64::from(v);
        self.valid += 1;
        if self.min.is_none_or(|(m, _)| x < m) {
            self.min = Some((x, pos));
        }
        if self.max.is_none_or(|(m, _)| x > m) {
            self.max = Some((x, pos));
        }
        let wsum = self.wsum + w;
        if wsum <= 0.0 {
            return;
        }
        let d = x - self.mean;
        self.mean += d * w / wsum;
        self.m2 += w * d * (x - self.mean);
        self.wsum = wsum;
    }

    pub fn push_missing(&mut self, n: u64) {
        self.missing += n;
    }

    pub fn valid(&self) -> u64 {
        self.valid
    }

    pub fn missing(&self) -> u64 {
        self.missing
    }

    /// Smallest value and its position; the first occurrence wins.
    pub fn min(&self) -> Option<(f64, u64)> {
        self.min
    }

    /// Largest value and its position; the first occurrence wins.
    pub fn max(&self) -> Option<(f64, u64)> {
        self.max
    }

    /// Weighted mean.
    pub fn mean(&self) -> Option<f64> {
        (self.wsum > 0.0).then_some(self.mean)
    }

    /// Weighted population standard deviation.
    pub fn std(&self) -> Option<f64> {
        (self.wsum > 0.0).then(|| (self.m2 / self.wsum).max(0.0).sqrt())
    }
}

/// numpy "linear" percentiles of `values` (sorted in place).
pub fn exact_percentiles(values: &mut [f32], ps: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    values.sort_unstable_by(f32::total_cmp);
    let last = values.len() - 1;
    ps.iter()
        .map(|p| {
            let pos = p / 100.0 * last as f64;
            let i = (pos.floor() as usize).min(last);
            let j = (i + 1).min(last);
            let frac = pos - i as f64;
            f64::from(values[i]) + (f64::from(values[j]) - f64::from(values[i])) * frac
        })
        .collect()
}

/// Percentiles from `counts` of equal bins spanning [lo, hi], linear within a bin.
pub fn hist_percentiles(counts: &[u64], lo: f64, hi: f64, ps: &[f64]) -> Vec<f64> {
    let total: u64 = counts.iter().sum();
    if total == 0 || counts.is_empty() {
        return Vec::new();
    }
    let width = (hi - lo) / counts.len() as f64;
    ps.iter()
        .map(|p| {
            // 0-based rank, as numpy's linear method.
            let rank = p / 100.0 * (total - 1) as f64;
            let mut before = 0.0;
            for (i, &c) in counts.iter().enumerate() {
                let c = c as f64;
                if c > 0.0 && rank < before + c {
                    let frac = (rank - before + 0.5) / c;
                    return (lo + (i as f64 + frac) * width).clamp(lo, hi);
                }
                before += c;
            }
            hi
        })
        .collect()
}

/// `counts` regrouped into `n` bins (bin i goes to i * n / counts.len()).
pub fn regroup(counts: &[u64], n: usize) -> Vec<u64> {
    let mut out = vec![0; n];
    if n == 0 {
        return out;
    }
    for (i, &c) in counts.iter().enumerate() {
        out[i * n / counts.len()] += c;
    }
    out
}

/// `n` counts of `values` between lo and hi (hi falls in the last bin).
pub fn bin_values(values: &[f32], lo: f64, hi: f64, n: usize) -> Vec<u64> {
    let mut out = vec![0; n];
    if n == 0 {
        return out;
    }
    for &v in values.iter().filter(|v| v.is_finite()) {
        out[bin_of(f64::from(v), lo, hi, n)] += 1;
    }
    out
}

/// The bin of `x` among `n` equal bins over [lo, hi] (hi in the last; all in
/// bin 0 when lo == hi).
pub(crate) fn bin_of(x: f64, lo: f64, hi: f64, n: usize) -> usize {
    if hi <= lo {
        return 0;
    }
    (((x - lo) / (hi - lo) * n as f64).floor().max(0.0) as usize).min(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc(vals: &[(f32, f64)]) -> Acc {
        let mut a = Acc::default();
        for (i, &(v, w)) in vals.iter().enumerate() {
            a.push(v, w, i as u64);
        }
        a
    }

    #[test]
    fn plain_mean_and_std() {
        let a = acc(&[2., 4., 4., 4., 5., 5., 7., 9.].map(|v| (v, 1.0)));
        assert_eq!(a.valid(), 8);
        assert!((a.mean().unwrap() - 5.0).abs() < 1e-12);
        assert!((a.std().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_mean() {
        let a = acc(&[(1.0, 1.0), (3.0, 3.0)]);
        assert!((a.mean().unwrap() - 2.5).abs() < 1e-12);
        assert!((a.std().unwrap() - 0.75f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn extremes_first_wins() {
        let a = acc(&[(3.0, 1.0), (1.0, 1.0), (1.0, 1.0), (3.0, 1.0)]);
        assert_eq!(a.min(), Some((1.0, 1)));
        assert_eq!(a.max(), Some((3.0, 0)));
    }

    #[test]
    fn missing_only() {
        let mut a = Acc::default();
        a.push_missing(5);
        assert_eq!((a.missing(), a.valid()), (5, 0));
        assert_eq!((a.min(), a.mean(), a.std()), (None, None, None));
    }

    #[test]
    fn exact_linear() {
        assert_eq!(exact_percentiles(&mut [4., 1., 3., 2.], &[25.0, 50.0]), [1.75, 2.5]);
        assert_eq!(exact_percentiles(&mut [7.], &PERCENTILES), [7.0; 5]);
    }

    #[test]
    fn histogram_percentiles_close() {
        let counts = vec![1u64; 1000];
        let p = hist_percentiles(&counts, 0.0, 999.0, &[50.0, 5.0]);
        assert!((p[0] - 499.5).abs() <= 1.0, "{p:?}");
        assert!((p[1] - 49.95).abs() <= 1.0, "{p:?}");
    }

    #[test]
    fn regroup_and_bins() {
        assert_eq!(regroup(&[1; 10], 5), [2; 5]);
        assert_eq!(bin_values(&[0., 0.5, 1.], 0., 1., 2), [1, 2]);
        assert_eq!(bin_values(&[3., 3.], 3., 3., 4), [2, 0, 0, 0]);
    }
}
