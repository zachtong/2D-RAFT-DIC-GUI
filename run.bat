@echo off
title RAFTcorr
call conda activate raftcorr

:: Kill any stale RAFTcorr server before starting (prevents ghost process issue)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)

python run_prod.py %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start RAFTcorr.
    echo Run install.bat first if you haven't.
    pause
)
