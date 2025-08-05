@echo off
REM Launcher for tldw chatbook TUI mode

TITLE tldw chatbook

REM Check if Windows Terminal is available
where wt.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM If not already in WT, launch this script in a new WT window and exit.
    if not defined WT_SESSION (
        echo Launching in Windows Terminal...
        start "tldw chatbook" wt.exe -d "%~dp0" "%~f0" %*
        exit /b 0
    )
)

REM If we are here, we are either in WT, or WT is not available.
REM Configure console settings if not in Windows Terminal
if not defined WT_SESSION (
    echo Configuring console for optimal display...
    
    REM Set console to UTF-8
    chcp 65001 >nul
    
    REM Set console size for TUI
    mode con: cols=120 lines=30
    
    REM Set console colors for better visibility
    color 0F
)

REM Run the application directly.
"%~dp0tldw-cli.exe" %*

REM If not in WT (i.e., running in legacy cmd), pause on error.
if not defined WT_SESSION (
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo Application exited with error code: %ERRORLEVEL%
        pause
    )
)
