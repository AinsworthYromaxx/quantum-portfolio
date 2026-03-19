@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ===================================================
echo   V4 Ultimate Quantum Portfolio Dashboard
echo   http://localhost:8502
echo  ===================================================
echo.

REM Kill anything on port 8502
echo Checking port 8502...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8502 "') do (
    if not "%%a"=="0" (
        echo   Killing PID %%a...
        taskkill /f /t /pid %%a >nul 2>&1
    )
)
timeout /t 3 /nobreak >nul

set STREAMLIT=C:\Users\maxim\OneDrive\Bureau\Python\.venv\Scripts\streamlit.exe
if not exist "%STREAMLIT%" set STREAMLIT=C:\Users\maxim\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe
if not exist "%STREAMLIT%" set STREAMLIT=streamlit

echo   Starting... Press Ctrl+C to stop.
echo.

REM Open browser after a short delay (in background)
start /b cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8502"

%STREAMLIT% run dashboard.py --server.port 8502 --server.headless true --server.maxUploadSize 200 --browser.gatherUsageStats false

echo.
echo  Stopped.
pause