@echo off
setlocal

echo ========================================
echo  Starting SPAD Detector Platform
echo ========================================
echo.

cd /d "%~dp0"

echo Starting backend API on http://127.0.0.1:8000 ...
start "SPAD Backend" cmd /k "npm run backend"

echo Starting frontend UI on http://127.0.0.1:3000 ...
start "SPAD Frontend" cmd /k "npm run dev"

timeout /t 5 /nobreak > nul
start http://127.0.0.1:3000/

echo.
echo ========================================
echo  Services requested:
echo  Backend:  http://127.0.0.1:8000/api/health
echo  Frontend: http://127.0.0.1:3000/
echo ========================================
pause
