@echo off
chcp 65001 >nul
set "PY=D:\Anaconda\envs\pytorch\python.exe"
if not exist "%PY%" (
  echo 未找到 %PY%
  pause
  exit /b 1
)
if not exist "%~dp0moshii\scripts\repack_samples.py" (
  echo 请在项目根目录运行此脚本。
  pause
  exit /b 1
)
"%PY%" "%~dp0moshii\scripts\repack_samples.py" %*
echo.
pause
