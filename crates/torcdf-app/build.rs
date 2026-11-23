//! Embeds a manifest that switches the process ANSI code page to UTF-8
//! (Windows 10 1903+), so netCDF-C (which opens paths through the ANSI API)
//! accepts non-ASCII file paths. Binaries only, MSVC only.
fn main() {
    println!("cargo:rerun-if-changed=utf8.manifest");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        let dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
        println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg-bins=/MANIFESTINPUT:{dir}/utf8.manifest");
    }
}
