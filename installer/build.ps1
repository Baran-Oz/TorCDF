# Builds the TorCDF installer: installer\Output\TorCDF-<version>-setup.exe
#   installer\build.ps1              release build + licence check + installer
#   installer\build.ps1 -SkipBuild   reuse the existing release torcdf.exe
# Local installers are for testing only; installers for other people come from the release workflow.
# Needs Inno Setup 6 and cargo-about. Never sets RUSTFLAGS (see BUILDING.md).
param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = 'C:\t\tc' }
$exe = Join-Path $env:CARGO_TARGET_DIR 'release\torcdf.exe'

$version = (& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'check-version.ps1') | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $version) { throw 'version not found in [workspace.package] of Cargo.toml' }

Push-Location $root
try {
    if (-not $SkipBuild) {
        & cargo build -p torcdf-app --release
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($LASTEXITCODE)" }
    }
    if (-not (Test-Path $exe)) { throw "$exe not found" }

    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'licences.ps1') -Check
    if ($LASTEXITCODE -ne 0) { throw 'THIRD_PARTY_LICENSES.txt is out of date' }

    $stage = Join-Path $PSScriptRoot 'Output\stage'
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force $stage | Out-Null
    Copy-Item $exe $stage
    Copy-Item (Join-Path $root 'LICENSE') (Join-Path $stage 'LICENSE.txt')
    Copy-Item (Join-Path $root 'crates\torcdf-app\assets\THIRD_PARTY_LICENSES.txt') $stage

    $iscc = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) { $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source }
    if (-not $iscc) { throw 'ISCC.exe (Inno Setup 6) not found' }

    & $iscc /Q "/DAppVersion=$version" "/DStage=$stage" (Join-Path $PSScriptRoot 'torcdf.iss')
    if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }
    $setup = Join-Path $PSScriptRoot "Output\TorCDF-$version-setup.exe"
    Write-Host "Built $setup"
} finally { Pop-Location }
