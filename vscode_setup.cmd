@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 bootstrap.py --install-only %*
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python not found. Install Python 3.12 or 3.13 x64 and enable Add Python to PATH.
        exit /b 1
    )
    python bootstrap.py --install-only %*
)
exit /b %ERRORLEVEL%
