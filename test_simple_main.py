"""Simplified main.py without multiprocessing spawn"""
import argparse
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset

print("Testing dataset loading like main.py (without spawn)...\n")

# Parse args like main.py
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_xyz")
parser.add_argument("--config", default="config/base.yaml")
args = parser.parse_args()

print(f"1. Dataset path: {args.dataset}")
print(f"2. Config path: {args.config}\n")

# Load config
print("3. Loading config...")
load_config(args.config)
print(f"   Config loaded: {config}\n")

# Load dataset - THIS is line 170 in main.py
print("4. Loading dataset...")
try:
    dataset = load_dataset(args.dataset)
    print(f"   ✓ Dataset loaded!")
    print(f"   Type: {type(dataset)}")
    print(f"   Length: {len(dataset)}")
    print(f"   rgb_files length: {len(dataset.rgb_files)}")
    if len(dataset.rgb_files) > 0:
        print(f"   First file: {dataset.rgb_files[0]}")
except Exception as e:
    print(f"   ✗ Error during load_dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Subsample - THIS is line 171 in main.py
print("\n5. Subsampling...")
subsample = config["dataset"]["subsample"]
print(f"   Subsample factor: {subsample}")
print(f"   Before: {len(dataset.rgb_files)} files")
dataset.subsample(subsample)
print(f"   After: {len(dataset.rgb_files)} files")

# Get image shape - THIS is line 172 in main.py where it FAILS
print("\n6. Getting image shape...")
try:
    print(f"   Calling dataset.get_img_shape()...")
    print(f"   rgb_files length before call: {len(dataset.rgb_files)}")
    h, w = dataset.get_img_shape()[0]
    print(f"   ✓ Success! Shape: {h}x{w}")
    print("\n✅ ALL TESTS PASSED!")
except Exception as e:
    print(f"   ✗ Error at get_img_shape: {e}")
    print(f"   rgb_files length at error: {len(dataset.rgb_files)}")
    import traceback
    traceback.print_exc()
