@echo off
REM Gracefully stop the Spotly 24/7 monitoring service (pending detections are confirmed first).
REM   scripts\stop_monitoring_service.bat [--db path\to\spotly.db]
setlocal
cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [Spotly] Virtual environment not found at "%PY%".
    exit /b 1
)
"%PY%" -m spotly.cli stop-service %*
endlocal
