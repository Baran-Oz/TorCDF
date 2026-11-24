//! CF time decoding with our own day arithmetic (no date crate).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Calendar {
    Standard,
    ProlepticGregorian,
    Julian,
    NoLeap,
    AllLeap,
    Day360,
}

impl Calendar {
    /// Parses a CF calendar name (case-insensitive); `None` input means the default.
    fn from_name(name: Option<&str>) -> Option<Calendar> {
        let Some(name) = name else { return Some(Calendar::Standard) };
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "" | "standard" | "gregorian" => Calendar::Standard,
            "proleptic_gregorian" => Calendar::ProlepticGregorian,
            "julian" => Calendar::Julian,
            "noleap" | "365_day" => Calendar::NoLeap,
            "all_leap" | "366_day" => Calendar::AllLeap,
            "360_day" => Calendar::Day360,
            _ => return None,
        })
    }

    /// Day number of a calendar date. Day 0 is arbitrary but consistent per calendar.
    fn to_days(self, y: i64, m: i64, d: i64) -> i64 {
        match self {
            Calendar::ProlepticGregorian => gregorian_to_jdn(y, m, d),
            Calendar::Julian => julian_to_jdn(y, m, d),
            Calendar::Standard => {
                if (y, m, d) < (1582, 10, 15) {
                    julian_to_jdn(y, m, d)
                } else {
                    gregorian_to_jdn(y, m, d)
                }
            }
            Calendar::NoLeap => y * 365 + CUM_NOLEAP[(m - 1) as usize] + d - 1,
            Calendar::AllLeap => y * 366 + CUM_ALLLEAP[(m - 1) as usize] + d - 1,
            Calendar::Day360 => y * 360 + (m - 1) * 30 + d - 1,
        }
    }

    fn days_to_date(self, n: i64) -> (i64, i64, i64) {
        match self {
            Calendar::ProlepticGregorian => jdn_to_gregorian(n),
            Calendar::Julian => jdn_to_julian(n),
            Calendar::Standard => {
                if n < JDN_GREGORIAN_START {
                    jdn_to_julian(n)
                } else {
                    jdn_to_gregorian(n)
                }
            }
            Calendar::NoLeap => from_table(n, 365, &CUM_NOLEAP),
            Calendar::AllLeap => from_table(n, 366, &CUM_ALLLEAP),
            Calendar::Day360 => {
                let y = n.div_euclid(360);
                let r = n.rem_euclid(360);
                (y, r / 30 + 1, r % 30 + 1)
            }
        }
    }

    fn max_day(self, m: i64) -> i64 {
        match self {
            Calendar::Day360 => 30,
            Calendar::NoLeap => CUM_NOLEAP[m as usize] - CUM_NOLEAP[m as usize - 1],
            _ => CUM_ALLLEAP[m as usize] - CUM_ALLLEAP[m as usize - 1],
        }
    }
}

const CUM_NOLEAP: [i64; 13] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
const CUM_ALLLEAP: [i64; 13] = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366];
/// Julian day number of 1582-10-15, the first Gregorian day of the mixed calendar.
const JDN_GREGORIAN_START: i64 = 2_299_161;

fn from_table(n: i64, year_len: i64, cum: &[i64; 13]) -> (i64, i64, i64) {
    let y = n.div_euclid(year_len);
    let r = n.rem_euclid(year_len);
    let m = (1..=12).find(|&m| r < cum[m]).unwrap_or(12);
    (y, m as i64, r - cum[m - 1] + 1)
}

fn gregorian_to_jdn(y: i64, m: i64, d: i64) -> i64 {
    let a = (14 - m) / 12;
    let yy = y + 4800 - a;
    let mm = m + 12 * a - 3;
    d + (153 * mm + 2) / 5 + 365 * yy + yy.div_euclid(4) - yy.div_euclid(100) + yy.div_euclid(400)
        - 32045
}

fn jdn_to_gregorian(j: i64) -> (i64, i64, i64) {
    let f = j + 1401 + ((4 * j + 274_277).div_euclid(146_097) * 3).div_euclid(4) - 38;
    richards_tail(f)
}

/// Common tail of the Richards (Julian day -> calendar date) algorithm.
fn richards_tail(f: i64) -> (i64, i64, i64) {
    let e = 4 * f + 3;
    let h = e.rem_euclid(1461) / 4 * 5 + 2;
    let d = h.rem_euclid(153) / 5 + 1;
    let m = (h.div_euclid(153) + 2).rem_euclid(12) + 1;
    (e.div_euclid(1461) - 4716 + (14 - m).div_euclid(12), m, d)
}

fn julian_to_jdn(y: i64, m: i64, d: i64) -> i64 {
    let a = (14 - m) / 12;
    let yy = y + 4800 - a;
    let mm = m + 12 * a - 3;
    d + (153 * mm + 2) / 5 + 365 * yy + yy.div_euclid(4) - 32083
}

fn jdn_to_julian(j: i64) -> (i64, i64, i64) {
    richards_tail(j + 1401)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: f64,
}

impl fmt::Display for CfDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02} {:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute
        )?;
        if self.second != 0.0 {
            if self.second.fract() == 0.0 {
                write!(f, ":{:02}", self.second as u32)?;
            } else {
                write!(f, ":{:06.3}", self.second)?;
            }
        }
        Ok(())
    }
}

impl CfDateTime {
    /// `YYYY-MM-DDTHH:MM:SS`, with `.mmm` only when the seconds are
    /// fractional (for files; `Display` is the label form).
    pub fn iso(&self) -> String {
        let head = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:",
            self.year, self.month, self.day, self.hour, self.minute
        );
        if self.second.fract() == 0.0 {
            format!("{head}{:02}", self.second as u32)
        } else {
            format!("{head}{:06.3}", self.second)
        }
    }
}

/// A parsed CF time-units string: unit length, epoch and calendar.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeUnits {
    unit_secs: f64,
    /// Calendar day number of the epoch.
    epoch_day: i64,
    /// Seconds into the epoch day (already adjusted for any UTC offset).
    epoch_sod: f64,
    calendar: Calendar,
}

const SECS_PER_DAY: f64 = 86_400.0;

/// A spacing of time-axis ticks: `secs` long (approximate for months), on
/// the first of the month every `months` months when that is not 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickStep {
    pub secs: f64,
    pub months: u32,
}

/// Most ticks [`TimeUnits::tick_values`] returns.
pub const MAX_TICKS: usize = 2000;

const MONTH_SECS: f64 = YEAR_SECS / 12.0;
const fn secs(s: f64) -> TickStep {
    TickStep { secs: s, months: 0 }
}
const fn months(n: u32) -> TickStep {
    TickStep { secs: MONTH_SECS * n as f64, months: n }
}

/// Tick spacings for time axes, finest first.
pub const TICK_STEPS: &[TickStep] = &[
    secs(60.0),
    secs(300.0),
    secs(900.0),
    secs(1800.0),
    secs(3600.0),
    secs(3.0 * 3600.0),
    secs(6.0 * 3600.0),
    secs(12.0 * 3600.0),
    secs(SECS_PER_DAY),
    secs(2.0 * SECS_PER_DAY),
    secs(7.0 * SECS_PER_DAY),
    secs(14.0 * SECS_PER_DAY),
    months(1),
    months(3),
    months(6),
    months(12),
    months(24),
    months(60),
    months(120),
    months(240),
    months(600),
    months(1200),
];
/// udunits: tropical year.
const YEAR_SECS: f64 = 365.242198781 * SECS_PER_DAY;

impl TimeUnits {
    /// Parses e.g. `"days since 2000-01-01"`; `None` when the units or the
    /// calendar are not understood.
    pub fn parse(units: &str, calendar: Option<&str>) -> Option<TimeUnits> {
        let calendar = Calendar::from_name(calendar)?;
        let lower = units.trim().to_ascii_lowercase();
        let (unit, epoch) = lower.split_once(" since ")?;
        let unit_secs = match unit.trim() {
            "ms" | "msec" | "msecs" | "millisecond" | "milliseconds" => 0.001,
            "s" | "sec" | "secs" | "second" | "seconds" => 1.0,
            "min" | "mins" | "minute" | "minutes" => 60.0,
            "h" | "hr" | "hrs" | "hour" | "hours" => 3600.0,
            "d" | "day" | "days" => SECS_PER_DAY,
            "month" | "months" => YEAR_SECS / 12.0,
            "year" | "years" | "yr" | "yrs" => YEAR_SECS,
            _ => return None,
        };
        let (y, m, d, sod) = parse_epoch(epoch.trim())?;
        if !(1..=12).contains(&m) || d < 1 || d > calendar.max_day(m) {
            return None;
        }
        Some(TimeUnits { unit_secs, epoch_day: calendar.to_days(y, m, d), epoch_sod: sod, calendar })
    }

    /// The value (file units) of this date and time; inverse of `decode`.
    pub fn encode(&self, year: i32, month: u8, day: u8, hour: u8, minute: u8) -> f64 {
        let day = self.calendar.to_days(i64::from(year), i64::from(month), i64::from(day));
        let secs = (day - self.epoch_day) as f64 * SECS_PER_DAY + f64::from(hour) * 3600.0
            + f64::from(minute) * 60.0
            - self.epoch_sod;
        secs / self.unit_secs
    }

    /// Length of one unit in seconds (hours: 3600).
    pub fn unit_secs(&self) -> f64 {
        self.unit_secs
    }

    /// Values in `[lo, hi]` (file units) where ticks of `step` fall: whole
    /// minutes/hours/days counted from midnight, or the first of every
    /// `step.months`-th month (January for years). At most [`MAX_TICKS`].
    pub fn tick_values(&self, lo: f64, hi: f64, step: &TickStep) -> Vec<f64> {
        let mut out = Vec::new();
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return out;
        }
        // Seconds since midnight of the epoch day.
        let secs = |v: f64| v * self.unit_secs + self.epoch_sod;
        let value = |s: f64| (s - self.epoch_sod) / self.unit_secs;
        if step.months == 0 {
            let first = (secs(lo) / step.secs).ceil();
            for k in 0..MAX_TICKS {
                let v = value((first + k as f64) * step.secs);
                if v > hi {
                    break;
                }
                out.push(v);
            }
        } else {
            let d = self.decode(lo);
            let n = i64::from(step.months);
            // Months since year 0, rounded down to a multiple of the step.
            let mut m = (i64::from(d.year) * 12 + i64::from(d.month) - 1).div_euclid(n) * n;
            while out.len() < MAX_TICKS {
                let day = self.calendar.to_days(m.div_euclid(12), m.rem_euclid(12) + 1, 1);
                let v = value((day - self.epoch_day) as f64 * SECS_PER_DAY);
                if v > hi {
                    break;
                }
                if v >= lo {
                    out.push(v);
                }
                m += n;
            }
        }
        out
    }

    pub fn decode(&self, v: f64) -> CfDateTime {
        // Keep arithmetic comfortably inside i64 even for garbage values.
        let v = if v.is_finite() { v } else { 0.0 };
        let total = (v * self.unit_secs + self.epoch_sod).clamp(-1e16, 1e16);
        let mut days = (total / SECS_PER_DAY).floor();
        // Round to milliseconds to hide float noise (0.1 day etc.).
        let mut sod = ((total - days * SECS_PER_DAY) * 1000.0).round() / 1000.0;
        if sod >= SECS_PER_DAY {
            sod -= SECS_PER_DAY;
            days += 1.0;
        }
        let (y, m, d) = self.calendar.days_to_date(self.epoch_day + days as i64);
        let whole = sod.floor();
        let hour = (whole / 3600.0) as u8;
        let minute = ((whole - hour as f64 * 3600.0) / 60.0) as u8;
        let second = sod - hour as f64 * 3600.0 - minute as f64 * 60.0;
        CfDateTime {
            year: y as i32,
            month: m as u8,
            day: d as u8,
            hour,
            minute,
            // Remove subtraction noise: values are millisecond-rounded.
            second: (second * 1000.0).round() / 1000.0,
        }
    }
}

/// Parses `YYYY-MM-DD[( |T)HH[:MM[:SS[.f]]]][Z|UTC|(+|-)HH[:MM]]` into
/// (year, month, day, seconds-of-day in UTC).
fn parse_epoch(s: &str) -> Option<(i64, i64, i64, f64)> {
    let s = s.trim();
    let sep = s.char_indices().find(|&(_, c)| c == 't' || c.is_whitespace());
    let (date, rest) = match sep {
        // Skip the separator by its own UTF-8 length (it may be multi-byte).
        Some((i, c)) => (&s[..i], s[i + c.len_utf8()..].trim()),
        None => (s, ""),
    };
    let mut dp = date.split('-');
    let y: i64 = dp.next()?.parse().ok()?;
    let m: i64 = dp.next()?.parse().ok()?;
    let d: i64 = dp.next().unwrap_or("1").parse().ok()?;
    if dp.next().is_some() {
        return None;
    }

    // Split the time-of-day from a trailing zone designator.
    let mut time = rest;
    let mut offset_secs = 0.0;
    if let Some(t) = time.strip_suffix("utc") {
        time = t.trim_end();
    } else if let Some(t) = time.strip_suffix('z') {
        time = t.trim_end();
    } else if let Some(i) = time.rfind(['+', '-']) {
        let (t, z) = time.split_at(i);
        let sign = if z.starts_with('-') { -1.0 } else { 1.0 };
        // Accepted: hh, hhmm, hh:mm.
        let digits = z[1..].trim();
        // ASCII-only first, so the byte slicing below cannot split a char.
        if !digits.bytes().all(|b| b.is_ascii_digit() || b == b':') {
            return None;
        }
        let (zh, zm): (f64, f64) = match digits.split_once(':') {
            Some((h, m)) if h.len() <= 2 && m.len() == 2 => (h.parse().ok()?, m.parse().ok()?),
            Some(_) => return None,
            None if digits.len() == 4 => (digits[..2].parse().ok()?, digits[2..].parse().ok()?),
            None if (1..=2).contains(&digits.len()) => (digits.parse().ok()?, 0.0),
            None => return None,
        };
        if zh > 23.0 || zm > 59.0 {
            return None;
        }
        offset_secs = sign * (zh * 3600.0 + zm * 60.0);
        time = t.trim_end();
    }

    let mut sod = 0.0;
    if !time.is_empty() {
        let mut tp = time.split(':');
        let h: f64 = tp.next()?.parse().ok()?;
        let mi: f64 = tp.next().map_or(Some(0.0), |x| x.parse().ok())?;
        let se: f64 = tp.next().map_or(Some(0.0), |x| x.parse().ok())?;
        if tp.next().is_some() || !(0.0..24.0).contains(&h) || !(0.0..60.0).contains(&mi) {
            return None;
        }
        sod = h * 3600.0 + mi * 60.0 + se;
    }
    Some((y, m, d, sod - offset_secs))
}

#[cfg(test)]
mod iso_tests {
    use super::*;

    #[test]
    fn iso_format() {
        let mut t = CfDateTime { year: 2000, month: 1, day: 1, hour: 0, minute: 0, second: 0.0 };
        assert_eq!(t.iso(), "2000-01-01T00:00:00");
        t.second = 1.5;
        assert_eq!(t.iso(), "2000-01-01T00:00:01.500");
    }
}

#[cfg(test)]
mod tick_tests {
    use super::*;

    fn dates(t: &TimeUnits, vals: &[f64]) -> Vec<String> {
        vals.iter().map(|&v| t.decode(v).to_string()).collect()
    }

    #[test]
    fn year_ticks_fall_on_new_year() {
        let t = TimeUnits::parse("seconds since 1970-01-01 00:00:00", Some("gregorian")).unwrap();
        // 1999-07-15 .. 2003-03-01
        let (lo, hi) = (932_000_000.0, 1_046_476_800.0);
        let year = TICK_STEPS.iter().find(|s| s.months == 12).unwrap();
        let v = t.tick_values(lo, hi, year);
        assert_eq!(dates(&t, &v), ["2000-01-01 00:00", "2001-01-01 00:00", "2002-01-01 00:00", "2003-01-01 00:00"]);
    }

    #[test]
    fn month_ticks_on_the_first() {
        let t = TimeUnits::parse("days since 2000-01-01", Some("noleap")).unwrap();
        let three = TICK_STEPS.iter().find(|s| s.months == 3).unwrap();
        let v = t.tick_values(20.0, 200.0, three);
        assert_eq!(dates(&t, &v), ["2000-04-01 00:00", "2000-07-01 00:00"]);
    }

    #[test]
    fn day_and_hour_ticks_on_whole_units() {
        let t = TimeUnits::parse("hours since 2025-11-30 06:00:00", None).unwrap();
        let day = TICK_STEPS.iter().find(|s| s.secs == 86_400.0).unwrap();
        // 2025-11-30 06:00 .. 2025-12-02 06:00
        assert_eq!(dates(&t, &t.tick_values(0.0, 48.0, day)), ["2025-12-01 00:00", "2025-12-02 00:00"]);
        let six = TICK_STEPS.iter().find(|s| s.secs == 21_600.0).unwrap();
        assert_eq!(dates(&t, &t.tick_values(1.0, 13.0, six)), ["2025-11-30 12:00", "2025-11-30 18:00"]);
    }

    #[test]
    fn ticks_are_bounded() {
        let t = TimeUnits::parse("seconds since 1970-01-01", None).unwrap();
        let minute = &TICK_STEPS[0];
        assert!(t.tick_values(0.0, 1e12, minute).len() <= MAX_TICKS);
        assert!(t.tick_values(f64::NAN, 1.0, minute).is_empty());
        assert!(t.tick_values(5.0, 1.0, minute).is_empty());
    }
}
