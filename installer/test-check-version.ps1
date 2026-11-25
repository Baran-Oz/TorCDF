# Tests installer\check-version.ps1 against the version in Cargo.toml.
$script = Join-Path $PSScriptRoot 'check-version.ps1'
if (-not (Test-Path $script)) { Write-Host 'FAIL check-version.ps1 not found'; exit 1 }
$v = $null; $inPkg = $false
foreach ($line in Get-Content (Join-Path (Split-Path -Parent $PSScriptRoot) 'Cargo.toml')) {
    if ($line -match '^\[(.+)\]') { $inPkg = ($Matches[1] -eq 'workspace.package') }
    elseif ($inPkg -and $line -match '^version\s*=\s*"([^"]+)"') { $v = $Matches[1]; break }
}
$failures = 0
$printed = (& powershell -NoProfile -ExecutionPolicy Bypass -File $script | Out-String).Trim()
if ($printed -eq $v) { Write-Host "ok   prints $v" } else { Write-Host "FAIL prints '$printed', want '$v'"; $failures++ }
foreach ($case in @(@("v$v", 0), @("v$v.9", 1), @("$v", 1), @('vX', 1))) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $script -Tag $case[0] | Out-Null
    if ($LASTEXITCODE -eq $case[1]) { Write-Host "ok   tag $($case[0]) -> exit $($case[1])" }
    else { Write-Host "FAIL tag $($case[0]): exit $LASTEXITCODE, want $($case[1])"; $failures++ }
}
if ($failures) { exit 1 }
Write-Host 'check-version tests passed'
exit 0
