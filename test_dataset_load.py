"""Test dataset loading after cache clear"""
import sys
sys.path.insert(0, ".")

from mast3r_slam.config import load_config
from mast3r_slam.dataloader import load_dataset

# Load config first
load_config("config/base.yaml")

# Try to load dataset
print("Loading TUM dataset...")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")

print(f"[OK] Dataset loaded successfully!")
print(f"  Number of frames: {len(dataset)}")
print(f"  RGB files sample: {dataset.rgb_files[0]}")
print(f"  Timestamps sample: {dataset.timestamps[0]}")

# Try to get image shape (this is where it was failing)
print("\nGetting image shape...")
img_shape, raw_shape = dataset.get_img_shape()
print(f"[OK] Image shape retrieved!")
print(f"  Processed shape: {img_shape}")
print(f"  Raw shape: {raw_shape}")

print("\n[SUCCESS] All dataset operations work!")
