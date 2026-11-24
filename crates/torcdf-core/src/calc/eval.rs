//! Computing a calculated variable for one slice request.
//!
//! Every input is read through `read_slice` over exactly the request's
//! indices (projected onto its own dimensions), so a strided overview reads
//! strided inputs and a formula may use other calculated variables.

use super::{BinOp, Derived, Expr, Func};
use crate::read::check_shape;
use crate::{CoreError, NcFile, Slice, SliceRequest};

#[cfg(test)]
thread_local! {
    /// Input requests made by `read_derived` on this thread (tests only).
    pub(super) static INPUT_LOG: std::cell::RefCell<Vec<SliceRequest>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A value over the request grid, or one number for all of it.
enum Val {
    Const(f64),
    Arr(Vec<f64>),
}

impl Val {
    fn at(&self, i: usize) -> f64 {
        match self {
            Val::Const(v) => *v,
            Val::Arr(a) => a[i],
        }
    }
}

fn truth(x: f64) -> bool {
    x != 0.0
}

fn bit(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

fn bin(op: BinOp, a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div if b == 0.0 => f64::NAN,
        BinOp::Div => a / b,
        BinOp::Pow => a.powf(b),
        BinOp::Lt => bit(a < b),
        BinOp::Le => bit(a <= b),
        BinOp::Gt => bit(a > b),
        BinOp::Ge => bit(a >= b),
        BinOp::Eq => bit(a == b),
        BinOp::Ne => bit(a != b),
        BinOp::And => bit(truth(a) && truth(b)),
        BinOp::Or => bit(truth(a) || truth(b)),
    }
}

fn call(f: Func, a: &[f64]) -> f64 {
    if f == Func::Where {
        return if a[0].is_nan() {
            f64::NAN
        } else if truth(a[0]) {
            a[1]
        } else {
            a[2]
        };
    }
    if a.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let x = a[0];
    match f {
        Func::Sqrt if x < 0.0 => f64::NAN,
        Func::Sqrt => x.sqrt(),
        Func::Abs => x.abs(),
        Func::Exp => x.exp(),
        Func::Log | Func::Log10 if x <= 0.0 => f64::NAN,
        Func::Log => x.ln(),
        Func::Log10 => x.log10(),
        Func::Sin => x.sin(),
        Func::Cos => x.cos(),
        Func::Tan => x.tan(),
        Func::Asin => x.asin(),
        Func::Acos => x.acos(),
        Func::Atan => x.atan(),
        Func::Atan2 => x.atan2(a[1]),
        Func::Deg => x.to_degrees(),
        Func::Rad => x.to_radians(),
        Func::Min => x.min(a[1]),
        Func::Max => x.max(a[1]),
        Func::Hypot => x.hypot(a[1]),
        Func::Floor => x.floor(),
        Func::Ceil => x.ceil(),
        Func::Round => x.round(),
        Func::Where => unreachable!("handled above"),
    }
}

/// For each of `input`'s dimensions, its position among `out` (matched by
/// name, in order).
fn dim_map(input: &[String], out: &[String]) -> Vec<usize> {
    let mut j = 0;
    input
        .iter()
        .map(|d| {
            while j < out.len() && out[j] != *d {
                j += 1;
            }
            j += 1;
            j - 1
        })
        .collect()
}

struct Ctx<'a> {
    file: &'a NcFile,
    req: &'a SliceRequest,
    dims: &'a [String],
    /// Output shape (`ceil(count / stride)` per dimension) and size.
    shape: Vec<usize>,
    total: usize,
}

impl Ctx<'_> {
    fn eval(&self, e: &Expr) -> Result<Val, CoreError> {
        Ok(match e {
            Expr::Num(v) => Val::Const(*v),
            Expr::Var(path) => self.input(path)?,
            Expr::Neg(a) => self.map1(self.eval(a)?, |x| -x),
            Expr::Not(a) => self.map1(self.eval(a)?, |x| if x.is_nan() { f64::NAN } else { bit(!truth(x)) }),
            Expr::Bin(op, a, b) => {
                let (a, b) = (self.eval(a)?, self.eval(b)?);
                match (&a, &b) {
                    (Val::Const(x), Val::Const(y)) => Val::Const(bin(*op, *x, *y)),
                    _ => Val::Arr((0..self.total).map(|i| bin(*op, a.at(i), b.at(i))).collect()),
                }
            }
            Expr::Call(f, args) => {
                let vals = args.iter().map(|a| self.eval(a)).collect::<Result<Vec<_>, _>>()?;
                if vals.iter().all(|v| matches!(v, Val::Const(_))) {
                    let xs: Vec<f64> = vals.iter().map(|v| v.at(0)).collect();
                    Val::Const(call(*f, &xs))
                } else {
                    let mut xs = vec![0.0; vals.len()];
                    Val::Arr(
                        (0..self.total)
                            .map(|i| {
                                for (x, v) in xs.iter_mut().zip(&vals) {
                                    *x = v.at(i);
                                }
                                call(*f, &xs)
                            })
                            .collect(),
                    )
                }
            }
        })
    }

    fn map1(&self, v: Val, f: impl Fn(f64) -> f64) -> Val {
        match v {
            Val::Const(x) => Val::Const(f(x)),
            Val::Arr(a) => Val::Arr(a.into_iter().map(f).collect()),
        }
    }

    /// Input `path` over the request's indices, spread over the output grid.
    fn input(&self, path: &str) -> Result<Val, CoreError> {
        let dims = match self.file.meta.var(path) {
            Some(v) => v.dims.clone(),
            None => self
                .file
                .derived(path)
                .map(|d| d.info.dims.clone())
                .ok_or_else(|| CoreError::Read { var: path.into(), msg: "no such variable".into() })?,
        };
        let map = dim_map(&dims, self.dims);
        let sub = SliceRequest {
            var: path.into(),
            start: map.iter().map(|&k| self.req.start[k]).collect(),
            count: map.iter().map(|&k| self.req.count[k]).collect(),
            stride: map.iter().map(|&k| self.req.stride[k]).collect(),
        };
        #[cfg(test)]
        INPUT_LOG.with(|l| l.borrow_mut().push(sub.clone()));
        let s = self.file.read_slice(&sub)?;
        if map.len() == self.shape.len() {
            return Ok(Val::Arr(s.data.iter().map(|&x| x as f64).collect()));
        }
        // Row-major strides of the input, placed at its dimensions' output
        // positions (0 elsewhere: broadcast).
        let mut in_stride = vec![0usize; self.shape.len()];
        let mut acc = 1;
        for (i, &k) in map.iter().enumerate().rev() {
            in_stride[k] = acc;
            acc *= s.shape[i];
        }
        let mut idx = vec![0usize; self.shape.len()];
        let mut out = Vec::with_capacity(self.total);
        for _ in 0..self.total {
            let at: usize = idx.iter().zip(&in_stride).map(|(i, s)| i * s).sum();
            out.push(s.data[at] as f64);
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(Val::Arr(out))
    }
}

impl NcFile {
    /// `d`'s values over `r`'s indices.
    pub(crate) fn read_derived(&self, d: &Derived, r: &SliceRequest) -> Result<Slice, CoreError> {
        let (shape, total) = check_shape(r, &d.info.shape)?;
        let ctx = Ctx { file: self, req: r, dims: &d.info.dims, shape: shape.clone(), total };
        let v = ctx.eval(&d.expr)?;
        // Never infinity: anything not finite is missing.
        let clean = |x: f64| if x.is_finite() { x as f32 } else { f32::NAN };
        let data: Vec<f32> = (0..total).map(|i| clean(v.at(i))).collect();
        let valid_count = data.iter().filter(|x| !x.is_nan()).count();
        Ok(Slice { shape, data, valid_count, vector: None })
    }
}

#[cfg(test)]
mod tests {
    use crate::fixtures;
    use crate::{NcFile, SeriesRequest, SliceRequest, SliceService, layer_path};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn nc4() -> (tempfile::TempDir, Arc<NcFile>) {
        let d = tempfile::tempdir().unwrap();
        let f = fixtures::paired(NcFile::open(&fixtures::write_nc4(d.path())).unwrap());
        (d, Arc::new(f))
    }

    fn add(f: &NcFile, name: &str, formula: &str) -> String {
        f.add_derived(f.check_formula(name, formula, None, None).unwrap()).path.clone()
    }

    fn req(var: &str, start: [usize; 3], count: [usize; 3], stride: [usize; 3]) -> SliceRequest {
        SliceRequest { var: var.into(), start: start.into(), count: count.into(), stride: stride.into() }
    }

    fn plane(var: &str, t: usize) -> SliceRequest {
        req(var, [t, 0, 0], [1, 10, 20], [1, 1, 1])
    }

    fn same(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-5 * y.abs().max(1.0), "at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn sum_matches_inputs() {
        let (_d, f) = nc4();
        let p = add(&f, "s", "uo + vo*2");
        let got = f.read_slice(&plane(&p, 3)).unwrap();
        let u = f.read_slice(&plane("/ocean/uo", 3)).unwrap();
        let v = f.read_slice(&plane("/ocean/vo", 3)).unwrap();
        let want: Vec<f32> = u.data.iter().zip(&v.data).map(|(a, b)| a + b * 2.0).collect();
        assert_eq!(got.shape, [1, 10, 20]);
        same(&got.data, &want);
        assert_eq!(got.valid_count, got.data.iter().filter(|x| !x.is_nan()).count());
    }

    #[test]
    fn strided_offset_equals_thinned_full() {
        let (_d, f) = nc4();
        let p = add(&f, "m", "uo*vo");
        let full = f.read_slice(&plane(&p, 0)).unwrap();
        let got = f.read_slice(&req(&p, [0, 1, 2], [1, 8, 15], [1, 3, 4])).unwrap();
        assert_eq!(got.shape, [1, 3, 4]);
        let want: Vec<f32> =
            [1, 4, 7].iter().flat_map(|&y| [2, 6, 10, 14].map(|x| full.data[y * 20 + x])).collect();
        same(&got.data, &want);
    }

    #[test]
    fn strided_inputs_stay_strided() {
        let (_d, f) = nc4();
        let p = add(&f, "m", "uo*vo + lat");
        super::INPUT_LOG.with(|l| l.borrow_mut().clear());
        f.read_slice(&req(&p, [0, 1, 2], [1, 8, 15], [1, 3, 4])).unwrap();
        let log = super::INPUT_LOG.with(|l| l.borrow().clone());
        assert_eq!(log.len(), 3, "{log:?}");
        for r in &log[..2] {
            assert_eq!((r.start.clone(), r.count.clone(), r.stride.clone()), (vec![0, 1, 2], vec![1, 8, 15], vec![1, 3, 4]));
        }
        // The 1-D latitude is read over its own dimension only.
        assert_eq!((log[2].start.clone(), log[2].count.clone(), log[2].stride.clone()), (vec![1], vec![8], vec![3]));
    }

    #[test]
    fn broadcast_lower_dimension() {
        let (_d, f) = nc4();
        let p = add(&f, "ul", "uo * 0 + lat");
        let got = f.read_slice(&plane(&p, 2)).unwrap();
        let lat = f.read_slice(&SliceRequest { var: "/ocean/lat".into(), start: vec![0], count: vec![10], stride: vec![1] }).unwrap();
        let u = f.read_slice(&plane("/ocean/uo", 2)).unwrap();
        let want: Vec<f32> = (0..200).map(|i| if u.data[i].is_nan() { f32::NAN } else { lat.data[i / 20] }).collect();
        same(&got.data, &want);
    }

    #[test]
    fn missing_and_division() {
        let (_d, f) = nc4();
        let p = add(&f, "q", "uo / (uo - uo)");
        let got = f.read_slice(&plane(&p, 0)).unwrap();
        assert!(got.data.iter().all(|x| x.is_nan()));
        assert_eq!(got.valid_count, 0);
        let p = add(&f, "r", "sqrt(-abs(uo) - 1) + log(uo - uo)");
        assert!(f.read_slice(&plane(&p, 0)).unwrap().data.iter().all(|x| x.is_nan()));
        let p = add(&f, "w", "where(uo > 0, 1, -1) * min(uo, 0) * max(uo, 0)");
        let got = f.read_slice(&plane(&p, 0)).unwrap();
        let u = f.read_slice(&plane("/ocean/uo", 0)).unwrap();
        // Missing in, missing out (unlike f32::min, which skips NaN).
        let want: Vec<f32> = u.data.iter().map(|&a| if a.is_nan() { f32::NAN } else { (if a > 0.0 { 1.0 } else { -1.0 }) * a.min(0.0) * a.max(0.0) }).collect();
        same(&got.data, &want);
    }

    #[test]
    fn packed_input_unpacked() {
        let d = tempfile::tempdir().unwrap();
        let f = NcFile::open(&fixtures::write_classic(d.path())).unwrap();
        let p = add(&f, "k", "t*1");
        let r = SliceRequest { var: p.clone(), start: vec![0, 0, 0], count: vec![3, 4, 8], stride: vec![1, 1, 1] };
        let got = f.read_slice(&r).unwrap();
        let want = f.read_slice(&SliceRequest { var: "/t".into(), ..r }).unwrap();
        same(&got.data, &want.data);
        assert!(want.data.iter().any(|x| x.is_nan()), "the fixture has fill values");
    }

    #[test]
    fn derived_of_derived() {
        let (_d, f) = nc4();
        add(&f, "speed", "sqrt(uo^2 + vo^2)");
        let p = add(&f, "twice", "speed*2");
        let s = f.read_slice(&plane("/ocean/speed", 5)).unwrap();
        let want: Vec<f32> = s.data.iter().map(|x| x * 2.0).collect();
        same(&f.read_slice(&plane(&p, 5)).unwrap().data, &want);
    }

    #[test]
    fn series_of_derived() {
        let (_d, f) = nc4();
        let p = add(&f, "s", "uo - vo");
        let svc = SliceService::new(1 << 22, 2, Arc::new(|| {}));
        let mut h = svc.request_series(&f, SeriesRequest { var: p.clone(), point: vec![0, 3, 4], along: 0, centre: 5 });
        let t0 = Instant::now();
        while h.finished().is_none() {
            assert!(t0.elapsed() < Duration::from_secs(20));
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(matches!(h.finished(), Some(Ok(()))));
        let mut got = vec![f32::NAN; 24];
        for b in h.take_blocks() {
            got[b.start..b.start + b.values.len()].copy_from_slice(&b.values);
        }
        let want: Vec<f32> = (0..24).map(|t| f.read_slice(&plane(&p, t)).unwrap().data[3 * 20 + 4]).collect();
        same(&got, &want);
    }

    #[test]
    fn layer_with_derived_x() {
        let (_d, f) = nc4();
        let p = add(&f, "speed", "sqrt(uo^2 + vo^2)");
        let s = f.read_slice(&plane(&layer_path(&p, "/ocean/uo+vo"), 1)).unwrap();
        same(&s.data, &f.read_slice(&plane(&p, 1)).unwrap().data);
        assert!(s.vector.is_some());
    }

    #[test]
    fn forget_file_drops_cache() {
        let (_d, f) = nc4();
        let p = add(&f, "s", "uo*2");
        let svc = SliceService::new(1 << 22, 2, Arc::new(|| {}));
        let r = plane(&p, 0);
        let mut h = svc.request(&f, r.clone());
        let t0 = Instant::now();
        while h.try_take().is_none() {
            assert!(t0.elapsed() < Duration::from_secs(20));
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(svc.cached(&f, &r).is_some());
        svc.forget_file(&f);
        assert!(svc.cached(&f, &r).is_none());
    }
}
