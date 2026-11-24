//! The Statistics window's form: the shared selection, the run and its
//! results, formatted as a table for the window, Copy and CSV.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

use torcdf_core::NcFile;
use torcdf_core::cf::{Geo, detect_geo};
use torcdf_core::stats::{Extreme, StatsJob, StatsOutcome, StatsRequest};

use crate::app::Job;
use crate::ui::fmt_value;
use crate::view::selection::SelectionForm;

/// Row labels, in order.
pub const ROWS: [&str; 13] =
    ["Valid", "Missing", "Min", "Min at", "Max", "Max at", "Mean", "Std dev", "P5", "P25", "Median", "P75", "P95"];

/// Results as text: a title, one column per variable, the rows of [`ROWS`].
#[derive(Clone, Debug, PartialEq)]
pub struct StatsTable {
    /// "<file name>: <selection summary>".
    pub title: String,
    pub columns: Vec<String>,
    /// One cell per column, per row of [`ROWS`].
    pub cells: Vec<Vec<String>>,
}

impl StatsTable {
    /// The table with `sep` between cells ("\t" for Copy, "," for CSV;
    /// CSV cells holding a comma or quote are quoted).
    pub fn text(&self, sep: &str) -> String {
        let cell = |c: &str| {
            if sep == "," && (c.contains(',') || c.contains('"')) {
                format!("\"{}\"", c.replace('"', "\"\""))
            } else {
                c.to_string()
            }
        };
        let mut out = format!("# {}\n", self.title);
        let head: Vec<String> = std::iter::once("statistic".to_string()).chain(self.columns.iter().map(|c| cell(c))).collect();
        out += &head.join(sep);
        out.push('\n');
        for (label, cells) in ROWS.iter().zip(&self.cells) {
            let row: Vec<String> = std::iter::once(label.to_string()).chain(cells.iter().map(|c| cell(c))).collect();
            out += &row.join(sep);
            out.push('\n');
        }
        out
    }
}

/// "36.02 N 14.40 E".
fn lonlat_text((lon, lat): (f64, f64)) -> String {
    let lon = (lon + 180.0).rem_euclid(360.0) - 180.0;
    let ns = if lat < 0.0 { 'S' } else { 'N' };
    let ew = if lon < 0.0 { 'W' } else { 'E' };
    format!("{:.2} {ns} {:.2} {ew}", lat.abs(), lon.abs())
}

/// Where an extreme is: each dimension's value (dates for time) except
/// lat/lon, which are given as degrees at the end.
pub fn where_text(nc: &NcFile, sel: &SelectionForm, var: &str, e: &Extreme) -> String {
    let Some(info) = nc.var_info_any(var) else { return String::new() };
    let geo: Vec<usize> = match detect_geo(&nc.meta_with_derived(), &info) {
        Geo::Regular { lat_dim, lon_dim, .. } => vec![lat_dim, lon_dim],
        Geo::Curvilinear { y_dim, x_dim, .. } => vec![y_dim, x_dim],
        Geo::None => Vec::new(),
    };
    let mut parts: Vec<String> = Vec::new();
    for (d, (name, &i)) in info.dims.iter().zip(&e.index).enumerate() {
        if e.lonlat.is_some() && geo.contains(&d) {
            continue;
        }
        let text = nc.dim_key(var, name).and_then(|k| sel.text_at(&k, i));
        parts.push(text.unwrap_or_else(|| format!("{name} {i}")));
    }
    let mut out = parts.join(", ");
    if let Some(ll) = e.lonlat {
        if !out.is_empty() {
            out += ", ";
        }
        out += &lonlat_text(ll);
    }
    out
}

/// The results of `outcome` (for `sel`'s ticked variables) as a table.
pub fn build_table(nc: &NcFile, sel: &SelectionForm, req: &StatsRequest, outcome: &StatsOutcome) -> StatsTable {
    let mut columns = Vec::new();
    let mut cells = vec![Vec::new(); ROWS.len()];
    for (var, r) in req.vars.iter().zip(outcome) {
        let info = nc.var_info_any(var);
        let name = info.as_ref().map_or_else(|| var.clone(), |i| i.name.clone());
        let units = info.as_ref().and_then(|i| i.attrs.iter().find(|a| a.name == "units")).and_then(|a| a.value.as_str());
        let mut head = match units {
            Some(u) => format!("{name} [{u}]"),
            None => name,
        };
        match r {
            Err(e) => {
                cells[0].push(e.clone());
                for c in cells.iter_mut().skip(1) {
                    c.push(String::new());
                }
            }
            Ok(s) => {
                if req.region.is_some() && !s.region_applied {
                    head += " (no lat/lon: region not applied)";
                }
                let all = (s.valid + s.missing).max(1) as f64;
                let weighted = if s.weighted { " (area-weighted)" } else { "" };
                let approx = if s.approximate { "≈ " } else { "" };
                let mut col = vec![
                    s.valid.to_string(),
                    format!("{} ({:.1} %)", s.missing, s.missing as f64 / all * 100.0),
                    fmt_value(s.min.value),
                    where_text(nc, sel, var, &s.min),
                    fmt_value(s.max.value),
                    where_text(nc, sel, var, &s.max),
                    format!("{}{weighted}", fmt_value(s.mean)),
                    format!("{}{weighted}", fmt_value(s.std)),
                ];
                col.extend(s.percentiles.iter().map(|p| format!("{approx}{}", fmt_value(*p))));
                col.resize(ROWS.len(), String::new());
                for (c, v) in cells.iter_mut().zip(col) {
                    c.push(v);
                }
            }
        }
        columns.push(head);
    }
    StatsTable { title: format!("{}: {}", sel.file_name, sel.summary()), columns, cells }
}

/// The Statistics window: what to summarise, the run and its results.
pub struct StatsForm {
    pub sel: SelectionForm,
    /// "Weight by cell area (cos latitude)".
    pub weighted: bool,
    pub job: Option<StatsJob>,
    /// The request of the run in progress.
    pub running: Option<StatsRequest>,
    /// The last results, with the request they answer.
    pub results: Option<(StatsRequest, StatsTable, StatsOutcome)>,
    /// Last error or save line.
    pub status: Option<String>,
    pub csv_dialog: Option<Job<Option<PathBuf>>>,
    /// The variable whose histogram is shown (index into the results).
    pub hist_var: usize,
}

impl StatsForm {
    pub fn new(file: u64, file_name: String, ticked: BTreeSet<String>) -> StatsForm {
        StatsForm {
            sel: SelectionForm::new(file, file_name, ticked),
            weighted: true,
            job: None,
            running: None,
            results: None,
            status: None,
            csv_dialog: None,
            hist_var: 0,
        }
    }

    /// Coordinates are being read, a run is going or the CSV dialog is open.
    pub fn busy(&self) -> bool {
        self.sel.loading_coords() || self.job.is_some() || self.csv_dialog.is_some()
    }

    /// The results are for another selection than the form now holds.
    pub fn out_of_date(&self) -> bool {
        let Some((req, ..)) = &self.results else { return false };
        self.sel.stats_request(self.weighted).ok().as_ref() != Some(req)
    }

    /// The results as text (see [`StatsTable::text`]).
    pub fn table(&self, sep: &str) -> Option<String> {
        self.results.as_ref().map(|(_, t, _)| t.text(sep))
    }
}

impl Deref for StatsForm {
    type Target = SelectionForm;
    fn deref(&self) -> &SelectionForm {
        &self.sel
    }
}

impl DerefMut for StatsForm {
    fn deref_mut(&mut self) -> &mut SelectionForm {
        &mut self.sel
    }
}
