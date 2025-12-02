@echo off
cd /d C:\Users\5090\MASt3R-SLAM-WINBUILD

REM Clear Python bytecode cache to ensure latest code is used
echo Clearing Python cache...
if exist mast3r_slam\__pycache__ rmdir /s /q mast3r_slam\__pycache__
if exist thirdparty\mast3r\dust3r\__pycache__ rmdir /s /q thirdparty\mast3r\dust3r\__pycache__

REM Prevent Python from creating new .pyc files
set PYTHONDONTWRITEBYTECODE=1

call venv\Scripts\activate.bat

echo Running SLAM on TUM freiburg1_xyz dataset...
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_xyz --config config\base.yaml --no-viz --save-as results\test_run

pause
