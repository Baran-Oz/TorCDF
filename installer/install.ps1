# Installs the latest TorCDF for the current user:
#   irm https://github.com/Baran-Oz/TorCDF/releases/latest/download/install.ps1 | iex
# Downloads TorCDF-setup.exe and SHA256SUMS.txt from the release, stops if the checksum does not match,
# then runs the installer silently (no admin rights needed).
# Options when run as a file: -BaseUrl <url> (another release or a local folder), -DryRun (check only).
# Under "irm | iex" the same options can be set with $env:TORCDF_BASEURL and $env:TORCDF_DRYRUN = '1'.
param(
    [string]$BaseUrl = $(if ($env:TORCDF_BASEURL) { $env:TORCDF_BASEURL } else { 'https://github.com/Baran-Oz/TorCDF/releases/latest/download' }),
    [switch]$DryRun = $($env:TORCDF_DRYRUN -eq '1')
)

function Install-TorCDF([string]$BaseUrl, [bool]$DryRun) {
    $appId = '{8B08E1FD-149D-46AF-9076-060F5CF0DFC3}_is1'
    $uninstall = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + $appId
    if (Test-Path "HKLM:\$uninstall") {
        Write-Host 'TorCDF is already installed for all users; update it from that installer or winget.'
        return 1
    }

    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $dir = Join-Path ([IO.Path]::GetTempPath()) ('torcdf-install-' + [guid]::NewGuid())
    New-Item -ItemType Directory $dir | Out-Null
    try {
        $base = $BaseUrl.TrimEnd('/')
        $setup = Join-Path $dir 'TorCDF-setup.exe'
        $sums = Join-Path $dir 'SHA256SUMS.txt'
        $web = New-Object Net.WebClient
        try {
            Write-Host "Downloading TorCDF from $base ..."
            $web.DownloadFile("$base/TorCDF-setup.exe", $setup)
            $web.DownloadFile("$base/SHA256SUMS.txt", $sums)
        } catch {
            Write-Host "TorCDF download failed: $($_.Exception.Message)"
            return 1
        } finally { $web.Dispose() }

        $want = $null
        foreach ($line in Get-Content $sums) {
            if ($line -match '^\s*([0-9a-fA-F]{64})\s+\*?TorCDF-setup\.exe\s*$') { $want = $Matches[1].ToLower() }
        }
        $got = (Get-FileHash -Algorithm SHA256 $setup).Hash.ToLower()
        if (-not $want -or $want -ne $got) {
            Write-Host 'TorCDF checksum mismatch: the download is damaged or not the published file. Nothing was installed.'
            return 1
        }
        Write-Host 'checksum OK'
        if ($DryRun) { return 0 }

        Write-Host 'Installing TorCDF for the current user ...'
        $p = Start-Process -FilePath $setup -Wait -PassThru -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/CURRENTUSER'
        if ($p.ExitCode -ne 0) {
            Write-Host "The TorCDF installer failed (exit code $($p.ExitCode))."
            return 1
        }
        $where = (Get-ItemProperty "HKCU:\$uninstall" -ErrorAction SilentlyContinue).InstallLocation
        Write-Host "TorCDF is installed in $where. Start it from the Start menu, or double-click a .nc file."
        return 0
    } finally {
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
    }
}

$code = Install-TorCDF $BaseUrl ([bool]$DryRun)
# Exit only when run as a file: under "irm | iex", exit would close the user's PowerShell window.
if ($PSCommandPath) { exit $code }
