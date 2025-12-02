# Windows Crash Analysis - December 2, 2025

## Summary

**CRASH LOCATION IDENTIFIED**: `tracker.opt_pose_ray_dist_sim3()`

The segfault occurs during pose optimization in the tracking loop, specifically when calling the CUDA-accelerated pose optimization function.

## Debugging Path

1. ✅ Confirmed environment works - test_tracking_simple.py passes (89% match rate)
2. ✅ Confirmed main.py crashes with segfault (exit code 139)
3. ✅ Created minimal test without FactorGraph - still crashes
4. ✅ Narrowed down to `tracker.track()` function
5. ✅ Confirmed all pre-processing works:
   - mast3r_match_asymmetric ✓
   - frame.update_pointmap ✓
   - get_points_poses ✓
   - valid mask computation ✓
   - match_frac calculation ✓ (0.8507)
6. ❌ **CRASH**: Happens when calling `opt_pose_ray_dist_sim3()`

## Exact Crash Point

File: `mast3r_slam/tracker.py`
Function: `track()`
Line: ~90-95

```python
try:
    if not use_calib:
        T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(  # <-- CRASH HERE
            Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
        )
```

## What Works

- Model loading
- Dataset loading
- MASt3R inference (mono)
- Feature matching (asymmetric and symmetric)
- All Python-level tensor operations
- CUDA basic operations

## What Doesn't Work

- `opt_pose_ray_dist_sim3()` - CUDA-based pose optimization
- Likely also: `opt_pose_calib_sim3()` - CUDA-based calib pose optimization

## Root Cause

The `opt_pose_ray_dist_sim3()` function uses custom CUDA kernels for optimization that have Windows-specific issues. This is likely related to:
- Windows CUDA context management
- Memory alignment issues in custom CUDA kernels
- lietorch operations in the optimization loop

## Possible Solutions

### Option 1: Skip Pose Optimization (Quick Test)
Modify `tracker.track()` to skip the optimization and use initial pose estimate.
- **Pro**: Would let us test if rest of SLAM works
- **Con**: Tracking quality would be poor

### Option 2: Replace with CPU-based Optimization
Implement a pure PyTorch (CPU/CUDA without custom kernels) version of pose optimization.
- **Pro**: Might work on Windows
- **Con**: Slower, significant development effort

### Option 3: Fix CUDA Kernels
Debug and fix the actual CUDA kernel code in the backend.
- **Pro**: Proper fix
- **Con**: Requires C++/CUDA expertise, very difficult

### Option 4: Use WSL2/Linux
Run on Linux where it works.
- **Pro**: Known to work
- **Con**: User wants Windows native

## Next Steps

User wants Windows native solution. Recommend trying Option 1 first to see if we can at least get partial functionality, then investigate Option 2.
