@echo off
REM MASt3R-SLAM GUI Launcher for Windows
REM This batch file activates the virtual environment and launches the GUI

echo ================================================================================
echo MASt3R-SLAM GUI Launcher
echo ================================================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please create virtual environment first:
    echo    python -m venv venv
    echo    venv\Scripts\activate.bat
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Launch GUI
echo [2/2] Launching GUI...
python slam_launcher.py
if errorlevel 1 (
    echo.
    echo [ERROR] GUI launch failed
    pause
    exit /b 1
)

echo.
echo GUI closed
pause
