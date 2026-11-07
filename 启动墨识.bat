@echo off
chcp 65001 >nul
set /p PY=<"%~dp0runtime\python.txt"
if not exist "%~dp0desktop.py" echo 安装不完整 & pause & exit /b 1
start "" "%PY%" "%~dp0desktop.py"
