@echo off
setlocal enabledelayedexpansion
echo ============================================
echo   RAFTcorr Installer
echo ============================================
echo.
echo Requires: Anaconda/Miniconda, NVIDIA GPU with CUDA
echo.

:: Check if conda is available
where conda >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] conda not found.
    echo Please install Anaconda or Miniconda first:
    echo   https://docs.anaconda.com/miniconda/
    pause
    exit /b 1
)

:: Check if environment already exists
call conda info --envs | findstr /c:"raftcorr" >nul 2>&1
if !errorlevel! equ 0 goto :env_ready

:: Create conda environment
echo Creating conda environment "raftcorr" with Python 3.10...
call conda create -n raftcorr python=3.10 -y
echo [OK] Environment created.

:env_ready
echo.
call conda activate raftcorr

:: ---- Detect CUDA version from nvidia-smi ----
echo Detecting CUDA version...
set CUDA_MAJOR=0
for /f "tokens=*" %%L in ('nvidia-smi 2^>nul ^| findstr "CUDA Version"') do set "NV_LINE=%%L"
if defined NV_LINE (
    set "CUDA_PART=!NV_LINE:*CUDA Version: =!"
    for /f "tokens=1 delims=. " %%a in ("!CUDA_PART!") do set CUDA_MAJOR=%%a
)

if !CUDA_MAJOR! equ 0 (
    echo [ERROR] NVIDIA GPU / CUDA not detected.
    echo RAFTcorr requires an NVIDIA GPU with CUDA driver.
    echo Install NVIDIA driver and verify with: nvidia-smi
    pause
    exit /b 1
)

echo [OK] Detected CUDA !CUDA_MAJOR!.x

:: ---- Install PyTorch with CUDA ----
if !CUDA_MAJOR! geq 12 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu124
) else (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
)

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url !TORCH_INDEX!
if !errorlevel! neq 0 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)

:: ---- Install RAFTcorr + remaining dependencies ----
echo Installing RAFTcorr...
pip install -e .
if !errorlevel! neq 0 (
    echo [ERROR] RAFTcorr installation failed.
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
