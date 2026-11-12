@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (py -3 bootstrap.py --install-only %*) else (python bootstrap.py --install-only %*)
set "NAPS_EXIT=%ERRORLEVEL%"
pause
exit /b %NAPS_EXIT%
