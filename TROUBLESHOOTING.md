# MASt3R-SLAM Windows Build - Troubleshooting Guide

**Date:** December 2, 2025
**Build:** Windows 11, RTX 5090, CUDA 12.8, PyTorch 2.8.0

---

## Issues Fixed

### 1. Windows Path Separator Issue ✅ FIXED
**Problem:** Dataset loading failed - `load_dataset()` returned wrong dataset type (RGBFiles instead of TUMDataset)

**Root Cause:** Path detection in `load_dataset()` split paths by `/` only, but Windows uses `\`
- Path: `datasets\tum\rgbd_dataset_freiburg1_xyz`
- After split by `/`: `['datasets\\tum\\rgbd_dataset_freiburg1_xyz']` (single element)
- "tum" not found → wrong dataset type → empty rgb_files → IndexError

**Fix:** `mast3r_slam/dataloader.py` line 328-330
```python
# Convert to POSIX format before splitting
dataset_path_str = str(pathlib.Path(dataset_path).as_posix())
split_dataset_type = dataset_path_str.split("/")
```

**Status:** ✅ FIXED - Dataset now loads correctly (798 frames)

---

### 2. Non-Writable NumPy Array Warning ✅ FIXED
**Problem:** Warning about non-writable NumPy array when creating tensors
```
UserWarning: The given NumPy array is not writable
```

**Root Cause:** `np.asarray()` returns read-only view of PIL image on Windows

**Fix:** `mast3r_slam/mast3r_utils.py` line 269
```python
# Changed from np.asarray() to np.array() to ensure writable copy
unnormalized_img=np.array(img),
```

**Status:** ✅ FIXED - No more warnings, writable arrays

---

## Current Issue - SLAM Tracking Failure

### Symptoms
- SLAM starts successfully, loads model and dataset
- Frame 0 initializes correctly
- Frame 1 tracking fails immediately (match_frac = 0.0000)
- Enters endless relocalization loop
- All relocalization attempts fail
- No point cloud is built

### Debug Output
```Skipped frame 1 - match_frac=0.0000 < 0.05
  valid_match_k: 0/196608
  valid_Cf: 196608, valid_Ck: 196608, valid_Q: 0
  ERROR: Zero valid matches from mast3r_match_asymmetric!

RELOCALIZING against kf 1 and [0]
  ERROR: X21 is all zeros BEFORE normalization!
    X21 shape: torch.Size([2, 384, 512, 3]), min: 0.000, max: 0.000
  DEBUG: iter_proj returned ZERO valid projections!
```

### Root Cause Analysis

**What Works:**
1. ✅ **Mono Inference** - Tested standalone, produces valid 3D points
   ```
   X min: -2.164, max: 3.828, mean: 0.438
   Non-zero elements: 589824 / 589824
   ```

2. ✅ **Asymmetric Matching** - Tested standalone, finds matches
   ```
   Xji: non-zero: 589824/589824
   valid_match_j: 175299/196608 valid matches (89%)
   ```

**What Fails:**
- ❌ **SLAM Tracking Loop** - Same functions fail when called from main.py
- ❌ **Relocalization** - Produces all-zero 3D points (X21 = 0)
- ❌ **CUDA Matching Kernel** - `iter_proj()` returns zero valid projections

### Hypothesis

The issue is **NOT** in the core MASt3R functions - they work fine in isolation. The problem appears when:
1. Running in the full SLAM loop with multiprocessing (spawn mode)
2. Shared memory / multiprocessing Manager interactions
3. CUDA kernel state or context in spawned processes

**Key Observation:** The zeros appear specifically in:
- `mast3r_asymmetric_inference()` when called during tracking/relocalization
- But NOT when called standalone in test scripts

This suggests:
- **Multiprocessing context issue** - spawn mode on Windows
- **CUDA context issue** - Kernel state not properly initialized in spawned processes
- **Shared memory corruption** - SharedKeyframes interaction

---

## Comparison: Ubuntu vs Windows

### Ubuntu Build (Working)
- Uses `fork` multiprocessing (default on Linux)
- Fork copies entire process memory space
- CUDA context inherited correctly
- MASt3R inference works in all code paths

### Windows Build (Failing)
- Uses `spawn` multiprocessing (required on Windows)
- Spawn creates fresh Python interpreter
- Must re-import all modules
- CUDA context must be re-initialized
- MASt3R fails in spawned process contexts

---

## Testing Evidence

### Test 1: Dataset Loading ✅
```bash
python test_dataset_load.py
```
**Result:** SUCCESS - 798 frames loaded, TUMDataset type

### Test 2: MASt3R Mono Inference ✅
```bash
python test_mast3r_output.py
```
**Result:** SUCCESS - Valid 3D points generated
```
X min: -2.164, max: 3.828
Non-zero: 589824/589824
```

### Test 3: Asymmetric Matching ✅
```bash
python test_mast3r_matching.py
```
**Result:** SUCCESS - 175k matches found
```
valid_match_j: 175299/196608 (89%)
```

### Test 4: Full SLAM Run ❌
```bash
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_xyz --config config\base.yaml --no-viz
```
**Result:** FAIL - Tracking fails, zero matches, relocalization loop

---

## Next Steps to Fix

### Option 1: Single-Threaded Mode
Try running without multiprocessing:
```yaml
# In config/base.yaml
single_thread: true
```

This bypasses spawn mode issues but loses parallel backend processing.

### Option 2: Investigate Spawn Mode
- Check if CUDA context properly initialized in spawned processes
- Verify shared memory tensors accessible
- Debug multiprocessing Manager with CUDA tensors

### Option 3: Port Fork-like Behavior
- Serialize model state
- Pass serialized state to spawned processes
- Re-initialize properly in child processes

### Option 4: CUDA Kernel Investigation
- Rebuild backends with debug symbols
- Add logging to `iter_proj` CUDA kernel
- Check if sm_120 kernels have issues

---

## Files Modified

### Core Fixes
1. `mast3r_slam/dataloader.py`
   - Line 328-330: Windows path handling fix
   - Line 72, 100, 124, 131: `comments="#"` for dataset loading

2. `mast3r_slam/mast3r_utils.py`
   - Line 269: `np.array()` instead of `np.asarray()`
   - Lines 206-213: Debug output for decoder

### Debug Additions
3. `mast3r_slam/tracker.py`
   - Lines 69-76: Match failure debug output

4. `mast3r_slam/matching.py`
   - Lines 41-43: X21 zeros detection
   - Lines 69-73: iter_proj failure detection

### Test Scripts
5. `test_dataset_load.py` - Dataset loading verification
6. `test_mast3r_output.py` - Mono inference test
7. `test_mast3r_matching.py` - Asymmetric matching test
8. `test_tracking_scenario.py` - Full tracking scenario test
9. `test_simple_main.py` - Simplified main.py test
10. `debug_dataset.py` - Dataset loading diagnostics

---

## Configuration

### Working Configuration
```yaml
use_calib: false
single_thread: false
dataset:
  subsample: 1
  img_downsample: 1
tracking:
  min_match_frac: 0.05  # 5% matches required
  Q_conf: 1.5  # Quality threshold
```

### Test Recommendations
1. Try `single_thread: true` to bypass multiprocessing
2. Try `img_downsample: 2` to reduce resolution
3. Try lowering `Q_conf: 1.0` (currently 1.5)
4. Try lowering `min_match_frac: 0.01` (currently 0.05)

---

## Known Working Components

| Component | Status | Notes |
|-----------|--------|-------|
| CUDA Extensions | ✅ Working | All 3 compiled for sm_120 |
| Model Loading | ✅ Working | Checkpoints load correctly |
| Dataset Loading | ✅ Working | Fixed Windows path issue |
| Mono Inference | ✅ Working | Produces valid 3D points |
| Asymmetric Matching | ✅ Working | Finds 89% matches standalone |
| Image Loading | ✅ Working | Fixed writable array issue |
| Tracking Loop | ❌ Failing | Zero matches in SLAM context |
| Relocalization | ❌ Failing | Produces all-zero points |
| CUDA Kernels | ❌ Failing | iter_proj returns zero valid |

---

## System Information

```
OS: Windows 11
GPU: NVIDIA GeForce RTX 5090 (32 GB)
Architecture: Blackwell (sm_120, compute capability 12.0)
CUDA: 12.8
Driver: 567.72
Python: 3.11.9
PyTorch: 2.8.0+cu128
Virtual Environment: venv
```

---

## Useful Commands

### Clear Python Cache
```cmd
rmdir /s /q mast3r_slam\__pycache__
set PYTHONDONTWRITEBYTECODE=1
```

### Run Tests
```cmd
python test_dataset_load.py
python test_mast3r_output.py
python test_mast3r_matching.py
```

### Run SLAM with Debug
```cmd
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_xyz --config config\base.yaml --no-viz
```

---

## Summary

✅ **Fixed Issues:**
1. Windows path separator in dataset loading
2. Non-writable NumPy array warning

❌ **Remaining Issue:**
- SLAM tracking fails with zero matches
- Core MASt3R functions work standalone
- Fails specifically in SLAM loop context
- Likely Windows spawn mode / CUDA context issue

**Status:** Core functionality proven working, but integration with SLAM loop failing due to Windows-specific multiprocessing/CUDA interaction.

---

*Last Updated: December 2, 2025*
*Contributors: Claude Code*
