# Building TorCDF (Windows)

## Prerequisites

Verified toolchain (the versions the build and tests were last run with):

- Rust stable, target `x86_64-pc-windows-msvc`: `rustc 1.99.0 (b940084d7 2026-09-28)`.
  The workspace uses edition 2024 and let chains (Rust 1.88+), so use a current stable.
- Visual Studio Build Tools with the MSVC C++ toolset and a Windows SDK:
  MSVC 14.50.35717 (Build Tools 18) and Windows SDK 10.0.26100.0.
- CMake 3.21 or newer (HDF5 and netCDF-C are built with it): verified with `cmake version 4.3.0-rc2`.
- Perl is not required: no build script in this workspace or in the `netcdf-src` / `hdf5-metno-src`
  crates invokes it. (The verification machine does have Perl on PATH, so this was not tested
  without it.)

Nothing needs to be installed at runtime. netCDF-C and HDF5 are compiled from source by the
`netcdf-src` crate and statically linked, and the C runtime is linked statically (`+crt-static`),
so `torcdf.exe` is a single self-contained file.

## Build and test

```
set CARGO_TARGET_DIR=C:\t\tc
cargo build -p torcdf-app --release
cargo test --workspace
```

The executable is `<target dir>\release\torcdf.exe` (here `C:\t\tc\release\torcdf.exe`).
The first build compiles HDF5 and netCDF-C and takes several minutes.

### Short paths (MAX_PATH)

MSBuild fails on paths over 260 characters, and the HDF5 CMake try-compile creates deep paths inside
the target directory. A deep checkout (for example under `.claude\worktrees\...`) therefore breaks
the build. Use a short target directory (`CARGO_TARGET_DIR=C:\t\tc`) or clone to a short path.

### Do not set RUSTFLAGS

`.cargo/config.toml` sets `-C target-feature=+crt-static` through `target.<triple>.rustflags`. The
`RUSTFLAGS` environment variable overrides that setting entirely, which drops `+crt-static` and
breaks linking against the statically built C libraries. If extra flags are needed, put them in
`.cargo/config.toml` or use `CARGO_ENCODED_RUSTFLAGS`, which must then include the `+crt-static` flag too.

### What `.cargo/msvc-static-crt.cmake` does

It is passed to CMake as `CMAKE_TOOLCHAIN_FILE` (see `[env]` in `.cargo/config.toml`, together with
`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`). It makes HDF5 and netCDF-C build against the static CRT
(`/MT`) to match Rust's `+crt-static`, and defines `_STATIC_INLINE_UCRT_FUNCTIONS=1` for netCDF-C so
the newer UCRT inline `fstat` wrapper does not leak out as an external symbol (it otherwise shadowed
`_fstat64`, breaking HDF5 file creation, and collided with `libucrt.lib`). Verified with the MSVC and
Windows SDK versions listed above; other versions are untested.

## Optional environment variables

- `TORCDF_GEBCO=<path to a GEBCO .nc file>`: enables the `open_reads_no_data` test in
  `crates/torcdf-core/tests/open.rs`, which asserts that opening the file takes under 1 s. Without
  it the test prints a skip message and passes.
- `TORCDF_LOG=<file>`: the app appends timestamped timing lines (select, first image, time axis,
  view errors) to this file, written from a separate logger thread.
- `TORCDF_SELECT=<variable path, e.g. /sst>`: only with `TORCDF_LOG`; selects that variable
  automatically when a file opens, so first-plot times can be measured without clicking.
