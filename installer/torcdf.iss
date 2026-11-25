; TorCDF installer (Inno Setup 6). Build with installer\build.ps1, which stages the files and passes
;   /DAppVersion=<x.y.z> /DStage=<folder with torcdf.exe, LICENSE.txt, THIRD_PARTY_LICENSES.txt>
; The AppId below must never change: upgrades find the old install through it.

#ifndef AppVersion
  #error AppVersion is not defined (use installer\build.ps1)
#endif
#ifndef Stage
  #error Stage is not defined (use installer\build.ps1)
#endif

[Setup]
AppId={{8B08E1FD-149D-46AF-9076-060F5CF0DFC3}
AppName=TorCDF
AppVersion={#AppVersion}
AppVerName=TorCDF {#AppVersion}
AppPublisher=Baran Oz
AppPublisherURL=https://github.com/Baran-Oz/TorCDF
AppSupportURL=https://github.com/Baran-Oz/TorCDF/issues
AppUpdatesURL=https://github.com/Baran-Oz/TorCDF/releases
AppCopyright=Copyright (C) 2026 Baran Oz
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\TorCDF
DefaultGroupName=TorCDF
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
LicenseFile={#Stage}\LICENSE.txt
SetupIconFile=..\crates\torcdf-app\assets\torcdf.ico
UninstallDisplayIcon={app}\torcdf.exe
UninstallDisplayName=TorCDF
ChangesAssociations=yes
CloseApplications=yes
RestartApplications=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
OutputDir=Output
OutputBaseFilename=TorCDF-{#AppVersion}-setup

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "ncassoc"; Description: "Open .nc files with TorCDF"; GroupDescription: "File types:"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#Stage}\torcdf.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Stage}\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Stage}\THIRD_PARTY_LICENSES.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\TorCDF"; Filename: "{app}\torcdf.exe"
Name: "{autodesktop}\TorCDF"; Filename: "{app}\torcdf.exe"; Tasks: desktopicon

[Registry]
; The file type TorCDF offers. HKA = per-user or all-users, following the install mode.
Root: HKA; Subkey: "Software\Classes\TorCDF.netCDF"; ValueType: string; ValueName: ""; ValueData: "netCDF file"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\TorCDF.netCDF\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\torcdf.exe,0"
Root: HKA; Subkey: "Software\Classes\TorCDF.netCDF\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\torcdf.exe"" ""%1"""
; .nc becomes TorCDF's only when the task is ticked; Windows keeps a choice the user already made.
Root: HKA; Subkey: "Software\Classes\.nc"; ValueType: string; ValueName: ""; ValueData: "TorCDF.netCDF"; Flags: uninsdeletevalue uninsdeletekeyifempty; Tasks: ncassoc
Root: HKA; Subkey: "Software\Classes\.nc\OpenWithProgids"; ValueType: string; ValueName: "TorCDF.netCDF"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty
; Other types only get TorCDF under "Open with".
Root: HKA; Subkey: "Software\Classes\.nc4\OpenWithProgids"; ValueType: string; ValueName: "TorCDF.netCDF"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKA; Subkey: "Software\Classes\.h5\OpenWithProgids"; ValueType: string; ValueName: "TorCDF.netCDF"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKA; Subkey: "Software\Classes\.hdf5\OpenWithProgids"; ValueType: string; ValueName: "TorCDF.netCDF"; ValueData: ""; Flags: uninsdeletevalue uninsdeletekeyifempty
; "torcdf" from the Run box.
Root: HKA; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\torcdf.exe"; ValueType: string; ValueName: ""; ValueData: "{app}\torcdf.exe"; Flags: uninsdeletekey

[Run]
Filename: "{app}\torcdf.exe"; Description: "{cm:LaunchProgram,TorCDF}"; Flags: nowait postinstall skipifsilent
