@echo off
REM Quick setup script for MASt3R-SLAM on Windows 11 with RTX 5090
REM This script automates the initial environment setup

echo ========================================
echo MASt3R-SLAM Windows Build Setup
echo ========================================
echo.

REM Check Python version
python --version | findstr "3.11" >nul
if errorlevel 1 (
    echo ERROR: Python 3.11 is required!
    echo Current version:
    python --version
    pause
    exit /b 1
)

REM Check CUDA installation
nvcc --version | findstr "12.8" >nul
if errorlevel 1 (
    echo WARNING: CUDA 12.8 not detected!
    echo Current CUDA version:
    nvcc --version
    echo.
    echo This build requires CUDA 12.8 for sm_120 support.
    echo Continue anyway? (Press Ctrl+C to cancel)
    pause
)

echo Step 1/5: Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

echo Step 2/5: Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo Step 3/5: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 4/5: Installing PyTorch 2.8.0 with CUDA 12.8...
echo This may take several minutes...
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch!
    pause
    exit /b 1
)
echo.

echo Step 5/5: Verifying PyTorch installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('sm_120 support:', 'sm_120' in torch.cuda.get_arch_list())"
if errorlevel 1 (
    echo ERROR: PyTorch verification failed!
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download checkpoints (see WINDOWS_BUILD_GUIDE.md section 4)
echo 2. Apply PyTorch 2.8.0 patches (see WINDOWS_BUILD_GUIDE.md section 5)
echo 3. Build CUDA extensions (see WINDOWS_BUILD_GUIDE.md sections 6-7)
echo.
echo To activate this environment in the future, run:
echo   activate_venv.bat
echo.
pause
