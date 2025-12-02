"""Quick verification test for MASt3R-SLAM Windows Build"""
import sys
import os

print("=" * 80)
print("MASt3R-SLAM Quick Verification Test")
print("=" * 80)
print()

# Test results
results = {}

# Test 1: Basic imports
print("[1/7] Testing basic imports...")
try:
    import numpy as np
    import torch
    import cv2
    print(f"  [OK] NumPy {np.__version__}")
    print(f"  [OK] PyTorch {torch.__version__}")
    print(f"  [OK] OpenCV {cv2.__version__}")
    results['basic_imports'] = True
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    results['basic_imports'] = False

# Test 2: CUDA availability
print("\n[2/7] Testing CUDA...")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        compute_cap = torch.cuda.get_device_properties(0)
        print(f"  [OK] CUDA {torch.version.cuda}")
        print(f"  [OK] GPU: {gpu_name}")
        print(f"  [OK] Memory: {gpu_memory:.2f} GB")
        print(f"  [OK] Compute: {compute_cap.major}.{compute_cap.minor}")
        results['cuda'] = True
    else:
        print("  [FAIL] CUDA not available")
        results['cuda'] = False
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    results['cuda'] = False

# Test 3: LieTorch
print("\n[3/7] Testing LieTorch...")
try:
    import lietorch
    print(f"  [OK] lietorch imported successfully")
    results['lietorch'] = True
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    results['lietorch'] = False

# Test 4: curope
print("\n[4/7] Testing curope...")
try:
    import curope
    print(f"  [OK] curope imported successfully")
    results['curope'] = True
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    results['curope'] = False

# Test 5: mast3r_slam_backends
print("\n[5/7] Testing mast3r_slam_backends...")
try:
    import mast3r_slam_backends
    print(f"  [OK] mast3r_slam_backends imported successfully")
    results['backends'] = True
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    results['backends'] = False

# Test 6: MASt3R packages
print("\n[6/7] Testing MASt3R packages...")
try:
    import mast3r
    import mast3r_slam
    print(f"  [OK] mast3r imported successfully")
    print(f"  [OK] mast3r_slam imported successfully")
    results['mast3r'] = True
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    results['mast3r'] = False

# Test 7: Checkpoints
print("\n[7/7] Checking model checkpoints...")
try:
    import os
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') or f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(checkpoint_dir, f)) for f in os.listdir(checkpoint_dir))
        print(f"  [OK] Found {len(files)} model files")
        print(f"  [OK] Total size: {total_size / 1024**3:.2f} GB")
        results['checkpoints'] = len(files) > 0
    else:
        print(f"  [FAIL] Checkpoints directory not found")
        results['checkpoints'] = False
except Exception as e:
    print(f"  [FAIL] Failed: {e}")
    results['checkpoints'] = False

# Summary
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
passed = sum(1 for v in results.values() if v)
total = len(results)
print(f"Tests passed: {passed}/{total}")
print()

for test_name, result in results.items():
    status = "[PASS]" if result else "[FAIL]"
    print(f"  {status:8s} {test_name}")

print()
if passed == total:
    print("[PASS] All tests passed! Your Windows build is working properly.")
    sys.exit(0)
else:
    print("[FAIL] Some tests failed. See details above.")
    print()
    print("Note: curope and mast3r_slam_backends DLL errors are likely due")
    print("to running from Git Bash/MINGW environment.")
    print("Try running this test from Windows Command Prompt or PowerShell.")
    sys.exit(1)
