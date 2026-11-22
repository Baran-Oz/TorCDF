use std::time::Instant;
use torcdf_core::fixtures;
use torcdf_core::{AttrValue, CoreError, DType, FileFormat, NcFile, Storage};

#[test]
fn opens_classic() {
    let d = tempfile::tempdir().unwrap();
    let f = NcFile::open(&fixtures::write_classic(d.path())).unwrap();
    assert_eq!(f.meta.format, FileFormat::Classic);
    let t = f.meta.var("/t").unwrap();
    assert_eq!(t.shape, vec![3, 4, 8]);
    assert_eq!(t.dtype, DType::I16);
    assert_eq!(t.storage, Storage::Contiguous);
    assert_eq!(t.dims, vec!["time", "lat", "lon"]);
    assert!(f.meta.root.dims.iter().find(|d| d.name == "time").unwrap().unlimited);
    let sf = t.attrs.iter().find(|a| a.name == "scale_factor").unwrap();
    assert_eq!(sf.value.as_f64s(), Some(vec![0.1]));
    assert_eq!(f.meta.all_vars().len(), 4);
}

#[test]
fn opens_groups_and_storage() {
    let d = tempfile::tempdir().unwrap();
    let f = NcFile::open(&fixtures::write_nc4(d.path())).unwrap();
    assert_eq!(f.meta.format, FileFormat::Netcdf4);
    let uo = f.meta.var("/ocean/uo").unwrap();
    assert_eq!(uo.path, "/ocean/uo");
    assert_eq!(uo.storage, Storage::Chunked { chunk: vec![1, 10, 20], compressed: true });
    assert_eq!(f.meta.var("/ocean/label").unwrap().dtype, DType::Str);
    assert_eq!(f.meta.group_of("/ocean/uo").path, "/ocean");
    assert_eq!(f.meta.group_of("/ocean/uo").name, "ocean");
    assert_eq!(f.meta.root.groups.len(), 1);
    assert!(f.meta.var("/uo").is_none());
    assert_eq!(
        f.meta.root.attrs.iter().find(|a| a.name == "title").unwrap().value,
        AttrValue::Text("fixture".into())
    );
}

#[test]
fn opens_unicode_path() {
    let d = tempfile::tempdir().unwrap();
    let sub = d.path().join("Zoë data files");
    std::fs::create_dir(&sub).unwrap();
    let src = fixtures::write_bare(d.path());
    let dst = sub.join("bäre.nc");
    std::fs::copy(&src, &dst).unwrap();
    let f = NcFile::open(&dst).unwrap();
    assert!(f.meta.var("/data").is_some());
}

#[test]
fn truncated_file_is_error() {
    let d = tempfile::tempdir().unwrap();
    let src = fixtures::write_nc4(d.path());
    let bytes = std::fs::read(&src).unwrap();
    let bad = d.path().join("bad.nc");
    std::fs::write(&bad, &bytes[..bytes.len() / 3]).unwrap();
    assert!(matches!(NcFile::open(&bad), Err(CoreError::Open { .. })));
    let txt = d.path().join("notes.nc");
    std::fs::write(&txt, "hello").unwrap();
    assert!(matches!(NcFile::open(&txt), Err(CoreError::Open { .. })));
}

#[test]
fn ncfile_is_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<NcFile>();
}

#[test]
fn open_reads_no_data() {
    let Ok(p) = std::env::var("TORCDF_GEBCO") else {
        eprintln!("TORCDF_GEBCO not set; skipping");
        return;
    };
    let start = Instant::now();
    let f = NcFile::open(std::path::Path::new(&p)).unwrap();
    let el = start.elapsed();
    assert!(el.as_secs_f64() < 1.0, "open took {el:?}");
    assert!(!f.meta.all_vars().is_empty());
}
