# Installs a TorCDF setup silently for the current user into a scratch folder, checks the result,
# uninstalls it and checks that nothing is left. Never ticks "Open .nc files" (/TASKS="").
#   installer\smoke.ps1 -Setup installer\Output\TorCDF-0.1.0-setup.exe
param([Parameter(Mandatory = $true)][string]$Setup, [string]$Dir = 'C:\t\torcdf-smoke')
$ErrorActionPreference = 'Stop'
$appId = '{8B08E1FD-149D-46AF-9076-060F5CF0DFC3}_is1'
$uninstKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$appId"
$failures = @()
function Test-That([bool]$ok, [string]$what) {
    if ($ok) { Write-Host "ok   $what" } else { Write-Host "FAIL $what"; $script:failures += $what }
}
function Get-NcDefault {
    $k = Get-Item 'HKCU:\Software\Classes\.nc' -ErrorAction SilentlyContinue
    if ($k) { $k.GetValue('') } else { $null }
}

if (-not (Test-Path $Setup)) { Write-Host "FAIL setup not found: $Setup"; exit 1 }
$version = if ((Split-Path -Leaf $Setup) -match 'TorCDF-([0-9.]+)-setup\.exe') { $Matches[1] } else { $null }
$ncBefore = Get-NcDefault

$p = Start-Process -FilePath $Setup -Wait -PassThru -ArgumentList @(
    '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/CURRENTUSER', '/TASKS=""', "/DIR=`"$Dir`"")
Test-That ($p.ExitCode -eq 0) "setup exit code 0 (got $($p.ExitCode))"
foreach ($f in 'torcdf.exe', 'LICENSE.txt', 'THIRD_PARTY_LICENSES.txt', 'unins000.exe') {
    Test-That (Test-Path (Join-Path $Dir $f)) "installed $f"
}
$key = Get-ItemProperty $uninstKey -ErrorAction SilentlyContinue
Test-That ($null -ne $key) 'uninstall entry exists'
if ($version -and $key) { Test-That ($key.DisplayVersion -eq $version) "DisplayVersion $version (got $($key.DisplayVersion))" }
Test-That ((Get-NcDefault) -eq $ncBefore) '.nc default unchanged without the task'
Test-That (Test-Path 'HKCU:\Software\Classes\TorCDF.netCDF\shell\open\command') 'TorCDF.netCDF file type registered'

$unins = Join-Path $Dir 'unins000.exe'
if (Test-Path $unins) {
    $u = Start-Process -FilePath $unins -Wait -PassThru -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'
    Test-That ($u.ExitCode -eq 0) "uninstall exit code 0 (got $($u.ExitCode))"
    # The uninstaller removes itself from a helper process; give it a moment.
    for ($i = 0; $i -lt 20 -and (Test-Path $Dir); $i++) { Start-Sleep -Milliseconds 500 }
}
Test-That (-not (Test-Path $Dir)) 'install folder removed'
Test-That (-not (Test-Path $uninstKey)) 'uninstall entry removed'
Test-That (-not (Test-Path 'HKCU:\Software\Classes\TorCDF.netCDF')) 'file type removed'
Test-That ((Get-NcDefault) -eq $ncBefore) '.nc default unchanged after uninstall'

if ($failures.Count) { Write-Host "$($failures.Count) check(s) failed"; exit 1 }
Write-Host 'smoke test passed'
exit 0
