# Tests installer\install.ps1's download and checksum checks with local file:// URLs (-DryRun: no setup runs).
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'install.ps1'
$failures = 0

function Invoke-Case([string]$name, [scriptblock]$prepare, [int]$wantExit, [string]$wantText) {
    $dir = Join-Path ([IO.Path]::GetTempPath()) ("torcdf-install-test-" + [guid]::NewGuid())
    New-Item -ItemType Directory $dir | Out-Null
    try {
        & $prepare $dir
        $url = ([Uri]$dir).AbsoluteUri
        $out = & powershell -NoProfile -ExecutionPolicy Bypass -File $script -BaseUrl $url -DryRun 2>&1 | Out-String
        $code = $LASTEXITCODE
        if ($code -eq $wantExit -and $out -match [regex]::Escape($wantText)) {
            Write-Host "ok   $name"
        } else {
            Write-Host "FAIL $name (exit $code, want $wantExit; output: $($out.Trim()))"
            $script:failures++
        }
    } finally { Remove-Item -Recurse -Force $dir }
}

function New-FakeSetup([string]$dir) {
    $setup = Join-Path $dir 'TorCDF-setup.exe'
    [IO.File]::WriteAllBytes($setup, [byte[]](1..200))
    (Get-FileHash -Algorithm SHA256 $setup).Hash.ToLower()
}

if (-not (Test-Path $script)) { Write-Host "FAIL install.ps1 not found"; exit 1 }

Invoke-Case 'correct checksum is accepted' {
    param($d) $h = New-FakeSetup $d
    Set-Content (Join-Path $d 'SHA256SUMS.txt') "$h  TorCDF-0.1.0-setup.exe`n$h  TorCDF-setup.exe" -Encoding ascii
} 0 'checksum OK'

Invoke-Case 'wrong checksum is refused' {
    param($d) New-FakeSetup $d | Out-Null
    Set-Content (Join-Path $d 'SHA256SUMS.txt') ("0" * 64 + "  TorCDF-setup.exe") -Encoding ascii
} 1 'checksum mismatch'

Invoke-Case 'missing SHA256SUMS.txt is refused' {
    param($d) New-FakeSetup $d | Out-Null
} 1 'download failed'

if ($failures) { Write-Host "$failures case(s) failed"; exit 1 }
Write-Host 'install.ps1 tests passed'
exit 0
