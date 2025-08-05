@echo off
REM Launcher for tldw chatbook Web Server mode

TITLE tldw chatbook - Web Server

echo Starting tldw chatbook web server...
echo.
echo The application will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Check if a custom port was provided
set PORT=8000
if not "%1"=="" set PORT=%1

REM Start the web server
"%~dp0tldw-serve.exe" --port %PORT%

REM If server exits, wait for user input
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server exited with error code: %ERRORLEVEL%
    pause
) else (
    REM Server stopped normally
    echo.
    echo Server stopped.
    timeout /t 3 >nul
)
