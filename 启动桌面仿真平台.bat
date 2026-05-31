@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if exist "%USERPROFILE%\.conda\envs\spad-detector\python.exe" (
  set "SPAD_PYTHON_EXE=%USERPROFILE%\.conda\envs\spad-detector\python.exe"
)

npm run desktop

pause
