@echo off
REM Activation script for MASt3R-SLAM Windows Build
REM This script activates the virtual environment and sets up CUDA paths

echo ========================================
echo MASt3R-SLAM Windows Build Environment
echo ========================================
echo.

REM Check if venv exists
if not exist "%~dp0venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create it first with: python -m venv venv
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call "%~dp0venv\Scripts\activate.bat"

REM Set CUDA paths (adjust if your CUDA is installed elsewhere)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%
set CUDA_HOME=%CUDA_PATH%

REM Set architecture list for CUDA compilation
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0

echo.
echo Environment activated!
echo - Python: %VIRTUAL_ENV%
echo - CUDA: %CUDA_PATH%
echo - TORCH_CUDA_ARCH_LIST: %TORCH_CUDA_ARCH_LIST%
echo.
echo You can now run MASt3R-SLAM commands.
echo To deactivate, type: deactivate
echo.
