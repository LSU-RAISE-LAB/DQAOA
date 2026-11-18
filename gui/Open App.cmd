@echo off
setlocal
cd /d "%~dp0"
title DQAOA - App
if not exist "%~dp0.venv\dqaoa-ready.txt" (
    echo Please double-click Install Requirements.cmd first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0launcher\open_app.py"
if errorlevel 1 pause
