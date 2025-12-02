"""Quick test to diagnose np.loadtxt issue"""
import numpy as np
import pathlib

rgb_list = pathlib.Path("datasets/tum/rgbd_dataset_freiburg1_xyz/rgb.txt")

print(f"File exists: {rgb_list.exists()}")
print(f"File path: {rgb_list}\n")

# Test with comments parameter
try:
    tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, comments="#")
    print(f"[OK] loadtxt succeeded with comments='#'")
    print(f"  Shape: {tstamp_rgb.shape}")
    print(f"  Type: {type(tstamp_rgb)}")
    print(f"  First 3 rows:\n{tstamp_rgb[:3]}")

    # Try accessing column 1
    if len(tstamp_rgb.shape) == 2:
        filenames = tstamp_rgb[:, 1]
        print(f"\n[OK] Column 1 access works")
        print(f"  First 3 filenames: {filenames[:3]}")
    else:
        print(f"\n[ERROR] tstamp_rgb is not 2D! Shape: {tstamp_rgb.shape}")
        print(f"  Content: {tstamp_rgb}")

except Exception as e:
    print(f"[ERROR] loadtxt failed: {e}")

# Test alternative: skiprows
print("\n" + "="*50)
print("Testing with skiprows=3...")
try:
    tstamp_rgb2 = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=3)
    print(f"[OK] loadtxt succeeded with skiprows=3")
    print(f"  Shape: {tstamp_rgb2.shape}")
    print(f"  First 3 rows:\n{tstamp_rgb2[:3]}")
except Exception as e:
    print(f"[ERROR] loadtxt failed: {e}")
