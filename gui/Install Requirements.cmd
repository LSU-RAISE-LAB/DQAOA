@echo off
setlocal
cd /d "%~dp0"
title DQAOA - Install Requirements
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\install.ps1"
echo.
pause
