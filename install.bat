@echo off
echo ============================================
echo   RAFTcorr Installer
echo ============================================
echo.
echo Requires: Anaconda/Miniconda, NVIDIA GPU with CUDA
echo.

:: Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] conda not found.
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.anaconda.com/miniconda/
    pause
    exit /b 1
)

:: Create conda environment (skip if already exists)
call conda info --envs | findstr /c:"raftcorr" >nul 2>&1
if %errorlevel% equ 0 (
    echo Environment "raftcorr" already exists, updating...
) else (
    echo Creating conda environment "raftcorr" with Python 3.10...
    call conda create -n raftcorr python=3.10 -y
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create conda environment.
        pause
        exit /b 1
    )
)

:: Activate and install
echo.
echo Activating environment and installing dependencies...
call conda activate raftcorr
pip install -e .
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed.
    echo Make sure NVIDIA GPU and CUDA driver are installed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Installation complete!
echo.
echo   To launch RAFTcorr:
echo     Double-click run.bat
echo     Or run:
echo       conda activate raftcorr
echo       python run_prod.py
echo ============================================
pause
