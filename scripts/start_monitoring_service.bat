@echo off
REM KIHC 24/7 headless monitoring service - starts minimized in the background.
REM
REM   scripts\start_monitoring_service.bat                       (default: El-Heddaf TV)
REM   scripts\start_monitoring_service.bat --channel el_heddaf_tv --channel echorouk_tv
REM   scripts\start_monitoring_service.bat --db D:\kihc\kihc.db --channel "My TV=https://host/index.m3u8"
REM
REM Stop gracefully:  scripts\stop_monitoring_service.bat   (or close the minimized window)
REM Logs:             kihc\data\logs\kihc_service.log
REM Run at boot:      Task Scheduler -> "At startup" -> this script (see README note in the report)
setlocal
cd /d "%~dp0.."

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [KIHC] Virtual environment not found at "%PY%".
    echo [KIHC] Create it first:  python -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)

set "ARGS=%*"
REM Phase 28.2: dedicated IPTV feeds only (no web / YouTube fallback); default fleet = data\service_channels.json
set "KIHC_IPTV_ONLY=1"
set "KIHC_IPTV_RELAY=1"
if "%~1"=="" set "ARGS="

start "KIHC 24/7 Monitoring Service" /MIN "%PY%" -m kihc.cli run-service %ARGS%
echo [KIHC] Monitoring service launched (minimized): run-service %ARGS%
echo [KIHC] Status:  "%PY%" -m kihc.cli service-status
echo [KIHC] Stop:    scripts\stop_monitoring_service.bat
endlocal
exit /b 0
