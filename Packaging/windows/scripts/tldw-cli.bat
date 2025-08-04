@echo off
REM Launcher for tldw chatbook TUI mode

TITLE tldw chatbook

REM Check if Windows Terminal is available
where wt.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Launch in Windows Terminal for better experience
    echo Launching in Windows Terminal...
    start "" wt.exe -d "%~dp0" cmd /k "%~dp0tldw-cli.exe"
    exit
)

REM Check if running in Windows Terminal already
if defined WT_SESSION (
    REM Already in Windows Terminal, just run the app
    "%~dp0tldw-cli.exe" %*
    pause
    exit
)

REM Fallback to cmd.exe with improved settings
echo Configuring console for optimal display...

REM Set console to UTF-8
chcp 65001 >nul

REM Set console size for TUI
mode con: cols=120 lines=30

REM Set console colors for better visibility
color 0F

REM Run the application
"%~dp0tldw-cli.exe" %*

REM Keep window open if app crashes
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code: %ERRORLEVEL%
    pause
)