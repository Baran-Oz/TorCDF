use torcdf_core::fixtures;

#[test]
fn static_netcdf_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    for p in [
        fixtures::write_classic(dir.path()),
        fixtures::write_nc4(dir.path()),
        fixtures::write_curvilinear(dir.path()),
        fixtures::write_bare(dir.path()),
    ] {
        let f = netcdf::open(&p).unwrap();
        assert!(f.variables().count() > 0 || f.groups().unwrap().count() > 0);
    }
}
