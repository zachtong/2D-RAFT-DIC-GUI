@echo off
echo ============================================
echo   RAFTcorr Installer
echo ============================================
echo.
echo This will install RAFTcorr and all dependencies.
echo Requires: Python 3.8+, NVIDIA GPU with CUDA
echo.
echo Installing...
pip install -e .
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed.
    echo Make sure Python and CUDA are installed.
    pause
    exit /b 1
)
echo.
echo ============================================
echo   Installation complete!
echo.
echo   To launch RAFTcorr:
echo     Double-click run.bat
echo     Or run: python run_prod.py
echo ============================================
pause
