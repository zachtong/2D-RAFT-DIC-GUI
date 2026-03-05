@echo off
python run_prod.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start RAFTcorr.
    echo Run install.bat first if you haven't.
    pause
)
