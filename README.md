# TorCDF

A fast viewer for netCDF and HDF5 files on Windows. Open a file of any size, see what is in it, and turn a
variable into an interactive map or plot in one or two clicks. Nothing to install besides the app: no Java, no
Python, no runtime. The netCDF-C and HDF5 libraries are built into the single `torcdf.exe`.

## Features

- **Browse** groups, dimensions, variables and attributes. Files open instantly, because only the structure is read.
- **Maps and plots:** colour maps on a world map with coastlines and borders, line plots, images and tables. Pan, zoom,
  and step through time and levels.
- **Arrows** for vector pairs such as u/v wind or currents, including rotated and staggered grids.
- **Pop-up time series:** click a point on the map to see its values over time.
- **Animation:** play through time, with adjustable speed and looping.
- **Calculation:** make your own variables from formulas (Ctrl+K).
- **Extract:** cut variables by time, region and other ranges, and save them as netCDF-4, keeping attributes,
  coordinates and compression (Ctrl+Shift+E).
- **Statistics:** count, minimum and maximum (with where and when), mean, standard deviation, percentiles and a
  histogram over any selection (Ctrl+I).
- **Formats:** netCDF-3, netCDF-4 and plain HDF5.
- **Export:** PNG images of the plot.
- **Looks:** light and dark.

## Install

Windows 10 or 11, 64-bit. The installer installs TorCDF for you only (no admin rights needed), or for all users if
you choose that on its first page.

**winget** (coming soon):

```
winget install Baran-Oz.TorCDF
```

**curl** (Command Prompt): download and run the installer:

```
curl -L -o "%TEMP%\TorCDF-setup.exe" https://github.com/Baran-Oz/TorCDF/releases/latest/download/TorCDF-setup.exe && "%TEMP%\TorCDF-setup.exe"
```

**PowerShell**: download, check the checksum and install silently for the current user:

```
irm https://github.com/Baran-Oz/TorCDF/releases/latest/download/install.ps1 | iex
```

**By hand:** download `TorCDF-<version>-setup.exe` from the
[Releases page](https://github.com/Baran-Oz/TorCDF/releases/latest). `torcdf.exe` on the same page runs without
installing.

Notes:

- The installer is not code-signed yet. Windows SmartScreen may say "Windows protected your PC": click
  **More info → Run anyway**.
- The installer offers to open `.nc` files with TorCDF (untick it to leave `.nc` files alone). If you picked another
  app for `.nc` files with **Open with → Always**, Windows keeps that choice; use **Open with → Choose another app**
  to switch to TorCDF. TorCDF also appears under "Open with" for
  `.nc4`, `.h5` and `.hdf5` files.
- Uninstall from **Settings → Apps**. Your settings in `%APPDATA%\TorCDF` are kept.

## Keyboard shortcuts

**Help → Keyboard shortcuts** (F1) in the app lists them all.

## Build from source

See [BUILDING.md](BUILDING.md).

## Licence

Copyright (C) 2026 Baran Oz.

TorCDF is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It
is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See [LICENSE](LICENSE).

The third-party software TorCDF includes is listed, with its licences, in
[THIRD_PARTY_LICENSES.txt](crates/torcdf-app/assets/THIRD_PARTY_LICENSES.txt). The coastlines and borders are from
Natural Earth (public domain).
