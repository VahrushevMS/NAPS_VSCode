@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 bootstrap.py %*
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python not found. Install Python 3.12 or 3.13 x64 from https://www.python.org/downloads/
        echo Enable "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    python bootstrap.py %*
)
set "NAPS_EXIT=%ERRORLEVEL%"
echo.
if "%NAPS_EXIT%"=="0" (echo Done. Results are in the outputs folder.) else (echo Run failed. Read the error above and README.md.)
pause
exit /b %NAPS_EXIT%
