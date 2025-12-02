# Bug Fixes Summary
**Date**: 2025-12-02
**Issues Addressed**: Backend jumping & --no-backend hanging

## Issues Found

### Issue 1: Backend Causes Massive Jumps (FIXED)
**Symptom**: After ~8 keyframes, entire pointcloud moves off-screen
**Root Cause**: False loop closure detection with permissive matching thresholds
**When**: Database retrieval finds similar keyframes and backend accepts bad matches

### Issue 2: --no-backend Mode Hangs (FIXED)
**Symptom**: After 8-9 keyframes, pointcloud stops updating, no new keyframes created
**Root Cause**: System enters RELOC mode when tracking fails, but no backend exists to process relocalization
**Result**: System stuck in RELOC mode forever

### Issue 3: No Threading Mode
**Symptom**: No points recorded
**Root Cause**: Windows multiprocessing + CUDA tensor sharing incompatibility
**Status**: Expected behavior, use --use-threading instead

## Fixes Applied

### Fix 1: Conservative Backend Config
**File**: `config/conservative_backend.yaml`
**Changes**:
- `retrieval.k: 2` (was 3) - Fewer loop closure candidates
- `retrieval.min_thresh: 1e-2` (was 5e-3) - Higher similarity required
- `local_opt.min_match_frac: 0.3` (was 0.1) - Require 30% point matches

**How to use**:
```bash
python main.py --dataset <path> --config config/conservative_backend.yaml --use-threading
```

### Fix 2: --no-backend Hang Prevention
**File**: `main.py:361-369`
**Change**: Only enter RELOC mode if backend exists
**Effect**: Failed tracking frames are skipped instead of causing system hang

## Usage Recommendations

### Option A: Smooth Tracking (No Global Optimization)
**Best for**: Quick demos, live camera, situations where smoothness matters more than accuracy

```bash
python main.py --dataset <path> --config config/base.yaml --use-threading --no-backend
```

**Pros**:
- Smooth visualization, no jumps
- Higher FPS
- Simple tracking only

**Cons**:
- May drift over long sequences
- No loop closure correction
- Less accurate long-term

### Option B: Conservative Backend (Recommended)
**Best for**: Accurate reconstruction, datasets with loop closures, offline processing

```bash
python main.py --dataset <path> --config config/conservative_backend.yaml --use-threading
```

**Pros**:
- Global optimization for accuracy
- Loop closure detection
- Better long-term consistency
- Reduced false loop closures vs base config

**Cons**:
- Small visible pose adjustments
- Lower FPS
- May still occasionally jump if very bad match

### Option C: Base Backend (Original)
**Best for**: Maximum loop closure aggressiveness

```bash
python main.py --dataset <path> --config config/base.yaml --use-threading
```

**Warning**: This config is prone to false loop closures causing massive jumps. Use conservative config instead.

## Diagnostic Features

The diagnostic logging added to `main.py:137-184` tracks backend optimization:

```
[DIAGNOSTIC] Backend optimization changed 2 keyframe poses:
  Max change: 0.013m
  Changed keyframes: [4, 5]
  ** WARNING: Large jump detected! (>0.5m) **
```

Watch for WARNING messages indicating problematic optimizations.

## Testing Results

### --no-backend Mode
- No longer hangs after tracking failure
- Continues processing frames (may skip frames that fail tracking)
- Works past 8-9 keyframe limit

### Conservative Backend Mode
- Needs user testing to verify false loop closure prevention
- Should reduce off-screen jumping vs base config
- Trade-off: May miss some valid loop closures

## Next Steps

1. **Test Option A (--no-backend)**: Verify smooth operation without hanging
2. **Test Option B (conservative_backend.yaml)**: Check if jumping is reduced
3. **If still jumping**: We can make config even more conservative or disable database retrieval entirely

## Files Changed

- `main.py`: Added RELOC mode check, diagnostic logging
- `config/conservative_backend.yaml`: New stricter config
- `DIAGNOSTIC_REPORT.md`: Technical analysis
- `diagnose_jumping.py`: Diagnostic documentation

## Rollback Instructions

If issues persist, revert to known working state:

```bash
git checkout v0.1.0
git checkout v0.1.0 -- main.py
```

Then run with --no-backend mode only (may hang, but was your last known working state).
