@echo off
REM Spotly 24/7 headless monitoring service - starts minimized in the background.
REM
REM   scripts\start_monitoring_service.bat                       (default: El-Heddaf TV)
REM   scripts\start_monitoring_service.bat --channel el_heddaf_tv --channel echorouk_tv
REM   scripts\start_monitoring_service.bat --db D:\spotly\spotly.db --channel "My TV=https://host/index.m3u8"
REM
REM Stop gracefully:  scripts\stop_monitoring_service.bat   (or close the minimized window)
REM Logs:             spotly\data\logs\spotly_service.log
REM Run at boot:      Task Scheduler -> "At startup" -> this script (see README note in the report)
setlocal
cd /d "%~dp0.."

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [Spotly] Virtual environment not found at "%PY%".
    echo [Spotly] Create it first:  python -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)

set "ARGS=%*"
REM Phase 28.2: dedicated IPTV feeds only (no web / YouTube fallback); default fleet = data\service_channels.json
set "SPOTLY_IPTV_ONLY=1"
set "SPOTLY_IPTV_RELAY=1"
if "%~1"=="" set "ARGS="

start "Spotly 24/7 Monitoring Service" /MIN "%PY%" -m spotly.cli run-service %ARGS%
echo [Spotly] Monitoring service launched (minimized): run-service %ARGS%
echo [Spotly] Status:  "%PY%" -m spotly.cli service-status
echo [Spotly] Stop:    scripts\stop_monitoring_service.bat
endlocal
exit /b 0
