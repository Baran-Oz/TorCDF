//! The Extract window's form: the shared selection plus its save.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;

use torcdf_core::NcFile;
use torcdf_core::extract::{ExtractJob, Plan};

use crate::app::Job;
use crate::view::selection::SelectionForm;

/// The Extract window: what to save, and the save.
pub struct ExtractForm {
    pub sel: SelectionForm,
    pub check: Result<Plan, String>,
    pub job: Option<ExtractJob>,
    /// Last outcome line ("Saved to …" / "Not saved: …").
    pub status: Option<String>,
    /// The Save dialog, while open.
    pub dialog: Option<Job<Option<PathBuf>>>,
}

impl ExtractForm {
    pub fn new(file: u64, file_name: String, ticked: BTreeSet<String>) -> ExtractForm {
        ExtractForm {
            sel: SelectionForm::new(file, file_name, ticked),
            check: Err("reading coordinates…".into()),
            job: None,
            status: None,
            dialog: None,
        }
    }

    /// Coordinates are being read, the dialog is open or a save runs.
    pub fn busy(&self) -> bool {
        self.sel.loading_coords() || self.dialog.is_some() || self.job.is_some()
    }

    /// Checks the form against `nc` again (rows snap; see `SelectionForm::refresh`).
    pub fn recheck(&mut self, nc: &NcFile) {
        self.check = self.sel.refresh().and_then(|s| nc.plan_extract(&s));
    }
}

impl Deref for ExtractForm {
    type Target = SelectionForm;
    fn deref(&self) -> &SelectionForm {
        &self.sel
    }
}

impl DerefMut for ExtractForm {
    fn deref_mut(&mut self) -> &mut SelectionForm {
        &mut self.sel
    }
}
