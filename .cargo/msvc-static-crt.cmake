# Build C dependencies (HDF5, netCDF-C) against the static CRT (/MT) to match
# Rust's +crt-static. Used via CMAKE_TOOLCHAIN_FILE in .cargo/config.toml.
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

# netCDF-C's config.h maps fstat -> _fstat64. The newer UCRT headers otherwise
# emit the inline fstat() wrapper (renamed to _fstat64 by that macro) as an
# external symbol, which shadows the real _fstat64 for HDF5 (wrong struct
# layout, breaking H5Fcreate) and collides with libucrt.lib (LNK2005).
# Making the UCRT inline wrappers static keeps them private to netCDF-C.
if(CMAKE_SOURCE_DIR MATCHES "netcdf-src")
  add_compile_definitions(_STATIC_INLINE_UCRT_FUNCTIONS=1)
endif()
