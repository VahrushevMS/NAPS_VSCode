@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Environment not found. Run START_WINDOWS.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py %*
set "NAPS_EXIT=%ERRORLEVEL%"
pause
exit /b %NAPS_EXIT%
