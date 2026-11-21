#ifndef PayloadDir
  #error Specify /DPayloadDir=<path to the prepared LabelingCapture distribution>
#endif
#ifndef AppVersion
  #define AppVersion "2026.11.21.1"
#endif

[Setup]
AppId={{DC8891FA-B6C8-4327-9B11-BE803F80E3A3}
AppName=LabelingCapture
AppVersion={#AppVersion}
AppVerName=LabelingCapture
DefaultDirName={localappdata}\Programs\LabelingCapture
DefaultGroupName=LabelingCapture
UsePreviousGroup=no
DisableDirPage=no
DisableProgramGroupPage=yes
DisableStartupPrompt=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19041
WizardStyle=modern
OutputBaseFilename=LabelingCapture-Setup
SetupIconFile={#PayloadDir}\Source\picture\图标_128x128.ico
UninstallDisplayIcon={app}\LabelingCapture.exe
Compression=lzma2/fast
SolidCompression=yes
RestartApplications=no

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Default.isl,ChineseSimplified.isl"

[Tasks]
Name: "startmenu"; Description: "在开始菜单中创建快捷方式"
Name: "desktop"; Description: "在桌面上创建快捷方式"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\LabelingCapture.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PayloadDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PayloadDir}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PayloadDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadDir}\Source\*"; DestDir: "{app}\Source"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "LabelingCapture.iss"; DestDir: "{app}\Source\installer"; Flags: ignoreversion
Source: "ChineseSimplified.isl"; DestDir: "{app}\Source\installer"; Flags: ignoreversion

[Icons]
Name: "{group}\LabelingCapture"; Filename: "{app}\LabelingCapture.exe"; WorkingDir: "{app}"; Tasks: startmenu
Name: "{userdesktop}\LabelingCapture"; Filename: "{app}\LabelingCapture.exe"; WorkingDir: "{app}"; Tasks: desktop

[Run]
Filename: "{app}\LabelingCapture.exe"; WorkingDir: "{app}"; Description: "启动LabelingCapture"; Flags: nowait postinstall skipifsilent unchecked
