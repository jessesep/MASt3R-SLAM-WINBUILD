"""Simple SLAM runner for testing"""
import sys
import os

print("=" * 80)
print("MASt3R-SLAM Test Run")
print("=" * 80)
print()

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
print("[1/4] Importing modules...")
try:
    import torch
    import numpy as np
    from pathlib import Path
    print(f"  [OK] PyTorch {torch.__version__}")
    print(f"  [OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# Check dataset
print("\n[2/4] Checking dataset...")
dataset_path = Path("datasets/tum/rgbd_dataset_freiburg1_xyz")
if not dataset_path.exists():
    print(f"  [FAIL] Dataset not found: {dataset_path}")
    sys.exit(1)

rgb_dir = dataset_path / "rgb"
depth_dir = dataset_path / "depth"
rgb_files = sorted(list(rgb_dir.glob("*.png")))
depth_files = sorted(list(depth_dir.glob("*.png")))

print(f"  [OK] Dataset found: {dataset_path}")
print(f"  [OK] RGB images: {len(rgb_files)}")
print(f"  [OK] Depth images: {len(depth_files)}")

# Import MASt3R-SLAM
print("\n[3/4] Importing MASt3R-SLAM...")
try:
    import mast3r_slam
    from mast3r_slam.config import load_config
    print(f"  [OK] mast3r_slam imported")
except Exception as e:
    print(f"  [FAIL] Failed to import mast3r_slam: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load config
print("\n[4/4] Loading configuration...")
try:
    config_path = "config/base.yaml"
    config = load_config(config_path)
    print(f"  [OK] Config loaded from {config_path}")
    print(f"  [OK] Image downsampling: {config.get('img_downsample', 1)}")
    print(f"  [OK] Use CUDA: {config.get('use_cuda', True)}")
except Exception as e:
    print(f"  [FAIL] Failed to load config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("SLAM INITIALIZATION SUCCESSFUL")
print("=" * 80)
print()
print("Note: Full SLAM run requires running main.py from Windows Command Prompt")
print("or PowerShell to avoid MINGW64 environment issues.")
print()
print("Next steps:")
print("1. Open Windows Command Prompt or PowerShell")
print("2. cd C:\\Users\\5090\\MASt3R-SLAM-WINBUILD")
print("3. .\\venv\\Scripts\\activate.bat")
print("4. python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz --config config/base.yaml --no-viz")
