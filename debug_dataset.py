"""Debug script to see what's happening with dataset loading"""
import sys
import pathlib
import numpy as np

# Show Python version and path
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"Working dir: {pathlib.Path.cwd()}")
print()

# Try to load the dataset the same way main.py does
dataset_path = "datasets/tum/rgbd_dataset_freiburg1_xyz"
print(f"Dataset path (input): {dataset_path}")

dataset_path_obj = pathlib.Path(dataset_path)
print(f"Dataset path (pathlib): {dataset_path_obj}")
print(f"Dataset path exists: {dataset_path_obj.exists()}")
print()

rgb_list = dataset_path_obj / "rgb.txt"
print(f"RGB list path: {rgb_list}")
print(f"RGB list exists: {rgb_list.exists()}")
print()

# Try loading
try:
    print("Attempting np.loadtxt...")
    tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, comments="#")
    print(f"✓ Success! Shape: {tstamp_rgb.shape}")
    print(f"First row: {tstamp_rgb[0]}")

    # Build file list
    rgb_files = [dataset_path_obj / f for f in tstamp_rgb[:, 1]]
    print(f"✓ Built file list: {len(rgb_files)} files")
    print(f"First file: {rgb_files[0]}")
    print(f"First file exists: {rgb_files[0].exists()}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Now let's try importing and using the actual dataloader...")
print("="*60 + "\n")

from mast3r_slam.config import load_config
from mast3r_slam.dataloader import load_dataset

# Load config
load_config("config/base.yaml")

# Try to load dataset
try:
    print("Creating dataset with load_dataset()...")
    dataset = load_dataset(dataset_path)
    print(f"✓ Dataset created!")
    print(f"  Type: {type(dataset)}")
    print(f"  Length: {len(dataset)}")
    print(f"  rgb_files length: {len(dataset.rgb_files)}")
    print(f"  First rgb_file: {dataset.rgb_files[0]}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
