//! Windows executable resources, MSVC only:
//! - a manifest that switches the process ANSI code page to UTF-8
//!   (Windows 10 1903+), so netCDF-C (which opens paths through the ANSI
//!   API) accepts non-ASCII file paths;
//! - the app icon, shown by Explorer and the taskbar.
fn main() {
    println!("cargo:rerun-if-changed=utf8.manifest");
    println!("cargo:rerun-if-changed=assets/torcdf.ico");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        let dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
        println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg-bins=/MANIFESTINPUT:{dir}/utf8.manifest");
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/torcdf.ico");
        if let Err(e) = res.compile() {
            panic!("could not embed the app icon: {e}");
        }
    }
}
