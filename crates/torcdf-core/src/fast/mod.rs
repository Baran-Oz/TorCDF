//! Fast chunk reader: locates chunks through HDF5, then reads and decodes
//! them outside the netCDF lock.
//!
//! Only the HDF5 groundwork exists so far; the reader that uses it comes
//! next, so outside tests nothing here is called yet.
#![cfg_attr(not(test), allow(dead_code))]

pub(crate) mod h5;

use std::fs::File;
use std::io;

/// Fills `buf` from `file` starting at byte `offset`, without a shared file
/// cursor, so several threads can read one handle at once.
pub(crate) fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let (mut buf, mut offset) = (buf, offset);
        while !buf.is_empty() {
            match file.seek_read(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}
