@echo off
REM Gracefully stop the KIHC 24/7 monitoring service (pending detections are confirmed first).
REM   scripts\stop_monitoring_service.bat [--db path\to\kihc.db]
setlocal
cd /d "%~dp0.."
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [KIHC] Virtual environment not found at "%PY%".
    exit /b 1
)
"%PY%" -m kihc.cli stop-service %*
endlocal
