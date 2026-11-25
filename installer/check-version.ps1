# Prints the workspace version from Cargo.toml ([workspace.package] version).
# With -Tag vX.Y.Z: exits 0 if the tag is "v" + that version, else prints both and exits 1.
param([string]$Tag)
$root = Split-Path -Parent $PSScriptRoot
$version = $null; $inPkg = $false
foreach ($line in Get-Content (Join-Path $root 'Cargo.toml')) {
    if ($line -match '^\[(.+)\]') { $inPkg = ($Matches[1] -eq 'workspace.package') }
    elseif ($inPkg -and $line -match '^version\s*=\s*"([^"]+)"') { $version = $Matches[1]; break }
}
if (-not $version) { Write-Host 'version not found in [workspace.package] of Cargo.toml'; exit 1 }
if (-not $Tag) { Write-Output $version; exit 0 }
if ($Tag -ne "v$version") { Write-Host "Tag $Tag does not match the Cargo.toml version $version (expected v$version)."; exit 1 }
Write-Host "Tag $Tag matches the Cargo.toml version."
exit 0
