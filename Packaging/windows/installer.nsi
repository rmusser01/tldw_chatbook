; NSIS Installer Script for tldw_chatbook
; Requires NSIS 3.0+ with MUI2

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

; Constants
!define PRODUCT_NAME "tldw chatbook"
!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "0.1.6.2"
!endif
!ifndef PRODUCT_PUBLISHER
  !define PRODUCT_PUBLISHER "TLDW Project"
!endif
!define PRODUCT_WEB_SITE "https://github.com/rmusser01/tldw_chatbook"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\tldw-cli.exe"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"

; MUI Settings
!define MUI_ABORTWARNING
!define MUI_ICON "assets\icon.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "assets\banner.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "assets\banner.bmp"

; Welcome page
!insertmacro MUI_PAGE_WELCOME
; License page
!define MUI_LICENSEPAGE_CHECKBOX
!insertmacro MUI_PAGE_LICENSE "assets\license.txt"
; Components page
!insertmacro MUI_PAGE_COMPONENTS
; Directory page
!insertmacro MUI_PAGE_DIRECTORY
; Instfiles page
!insertmacro MUI_PAGE_INSTFILES
; Finish page
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Launch ${PRODUCT_NAME} in Terminal"
!define MUI_FINISHPAGE_RUN_FUNCTION "LaunchTUI"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README.txt"
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_INSTFILES

; Language files
!insertmacro MUI_LANGUAGE "English"

; Installer attributes
Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "tldw-chatbook-${PRODUCT_VERSION}-setup.exe"
InstallDir "$PROGRAMFILES\${PRODUCT_NAME}"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""
ShowInstDetails show
ShowUnInstDetails show
RequestExecutionLevel admin

; Version information
VIProductVersion "${PRODUCT_VERSION}.0"
VIAddVersionKey "ProductName" "${PRODUCT_NAME}"
VIAddVersionKey "CompanyName" "${PRODUCT_PUBLISHER}"
VIAddVersionKey "LegalCopyright" "© 2024 Robert Musser. Licensed under AGPL-3.0-or-later"
VIAddVersionKey "FileDescription" "${PRODUCT_NAME} Installer"
VIAddVersionKey "FileVersion" "${PRODUCT_VERSION}"

Section "Core Application (required)" SEC01
  SectionIn RO
  
  ; Set output path
  SetOutPath "$INSTDIR"
  SetOverwrite try
  
  ; Copy main executable and dependencies
  File /r "dist\tldw-cli.exe.dist\*.*"
  
  ; Copy batch wrappers
  File "dist\*.bat"
  
  ; Copy documentation
  File "dist\README.txt"
  File "dist\LICENSE.txt"
  
  ; Create shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\tldw-cli.bat" "" "$INSTDIR\tldw-cli.exe" 0
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME} (Terminal).lnk" "wt.exe" '-d "$INSTDIR" cmd /k tldw-cli.bat' "$INSTDIR\tldw-cli.exe" 0
  CreateShortCut "$DESKTOP\${PRODUCT_NAME}.lnk" "$INSTDIR\tldw-cli.bat" "" "$INSTDIR\tldw-cli.exe" 0
SectionEnd

Section "Web Server Support" SEC02
  ; Check if tldw-serve.exe was built
  ${If} ${FileExists} "dist\tldw-serve.exe.dist\*.*"
    SetOutPath "$INSTDIR"
    File /r "dist\tldw-serve.exe.dist\*.*"
    
    ; Create web server shortcuts
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME} (Web Browser).lnk" "$INSTDIR\tldw-serve.bat" "" "$INSTDIR\tldw-serve.exe" 0
    
    ; Add firewall rule
    DetailPrint "Adding Windows Firewall rule..."
    nsExec::ExecToLog 'netsh advfirewall firewall add rule name="${PRODUCT_NAME} Web Server" dir=in action=allow program="$INSTDIR\tldw-serve.exe" enable=yes'
  ${EndIf}
SectionEnd

Section "Windows Terminal Integration" SEC03
  ; Check if Windows Terminal is installed
  ReadRegStr $0 HKCU "Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe" "PackageID"
  ${If} $0 == ""
    MessageBox MB_YESNO "Windows Terminal is not installed. Would you like to download it?" IDYES download_wt
    Goto skip_wt
    download_wt:
      ExecShell "open" "https://aka.ms/terminal"
    skip_wt:
  ${EndIf}
SectionEnd

Section -AdditionalIcons
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk" "$INSTDIR\uninst.exe"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\README.lnk" "$INSTDIR\README.txt"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Website.lnk" "$INSTDIR\${PRODUCT_NAME}.url"
SectionEnd

Section -Post
  WriteUninstaller "$INSTDIR\uninst.exe"
  
  ; Write registry keys
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\tldw-cli.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "$(^Name)"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninst.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\tldw-cli.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
  
  ; Calculate and write install size
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "EstimatedSize" "$0"
  
  ; Write URL file
  WriteIniStr "$INSTDIR\${PRODUCT_NAME}.url" "InternetShortcut" "URL" "${PRODUCT_WEB_SITE}"
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC01} "The core application files (required)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC02} "Web server support for browser-based access"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC03} "Integration with Windows Terminal for better TUI experience"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; Uninstaller
Section Uninstall
  ; Remove firewall rule
  nsExec::ExecToLog 'netsh advfirewall firewall delete rule name="${PRODUCT_NAME} Web Server"'
  
  ; Remove files and directories
  Delete "$INSTDIR\${PRODUCT_NAME}.url"
  Delete "$INSTDIR\uninst.exe"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\*.bat"
  
  ; Remove program directories
  RMDir /r "$INSTDIR\tldw-cli.exe.dist"
  RMDir /r "$INSTDIR\tldw-serve.exe.dist"
  RMDir "$INSTDIR"
  
  ; Remove shortcuts
  Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.*"
  RMDir "$SMPROGRAMS\${PRODUCT_NAME}"
  Delete "$DESKTOP\${PRODUCT_NAME}.lnk"
  
  ; Remove registry keys
  DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
  
  ; Clean up app data (optional - ask user)
  MessageBox MB_YESNO "Remove application data and settings?" IDNO skip_appdata
    RMDir /r "$APPDATA\tldw_cli"
    RMDir /r "$LOCALAPPDATA\tldw_cli"
  skip_appdata:
  
  SetAutoClose true
SectionEnd

; Functions
Function LaunchTUI
  Exec '"$INSTDIR\tldw-cli.bat"'
FunctionEnd

Function .onInit
  ; Check for previous installation
  ReadRegStr $R0 ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString"
  StrCmp $R0 "" done
  
  MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION \
  "${PRODUCT_NAME} is already installed. $\n$\nClick `OK` to remove the previous version or `Cancel` to cancel this upgrade." \
  IDOK uninst
  Abort
  
  uninst:
    ClearErrors
    ExecWait '$R0 _?=$INSTDIR'
    
  done:
FunctionEnd