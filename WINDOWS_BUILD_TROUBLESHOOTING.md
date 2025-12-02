# Windows Build Troubleshooting Guide

This document details all errors encountered during the Windows 11 build of MASt3R-SLAM with RTX 5090 (sm_120) support and their solutions.

## Build Environment

- **OS:** Windows 11
- **GPU:** NVIDIA RTX 5090 (Blackwell, sm_120)
- **CUDA:** 12.8
- **PyTorch:** 2.8.0+cu128
- **Python:** 3.11
- **Compiler:** Visual Studio 2022 Build Tools

---

## Error 1: Git Push Authentication Failure

**Error:**
```
fatal: unable to persist credentials with the 'wincredman' credential store
```

**Cause:** Windows Credential Manager configuration issue with git authentication

**Solution:**
Use GitHub Personal Access Token directly in the remote URL:
```bash
git remote set-url origin https://username:TOKEN@github.com/username/repo.git
```

---

## Error 2: lietorch pyproject.toml Duplicate Keys

**Error:**
```
tomllib.TOMLDecodeError: Cannot overwrite a value (at line 12, column 16)
```

**Cause:** The pyproject.toml file had duplicate "version" and "description" fields

**Files Affected:**
- `temp_lietorch/pyproject.toml`

**Solution:**
Rewrote pyproject.toml with single entries for each field:
```toml
[build-system]
requires = ["setuptools", "torch>=2.6.0"]
build-backend = "setuptools.build_meta"

[project]
name = "lietorch"
version = "0.2"
description = "Lie Groups for PyTorch"
authors = [{ name="teedrz", email="zachteed@gmail.com" }]
license = { text = "BSD-3-Clause" }
readme = "README.md"
requires-python = ">=3.11"
dependencies = ["torch>=2.6.0", "numpy<2"]
```

---

## Error 3: Missing wheel Package

**Error:**
```
error: invalid command 'bdist_wheel'
```

**Cause:** wheel and ninja packages not installed in venv

**Solution:**
```bash
pip install wheel ninja
```

---

## Error 4: Eigen Submodule Not Initialized

**Error:**
```
fatal error C1083: Cannot open include file: 'Eigen/Sparse': No such file or directory
```

**Cause:** Git submodules (Eigen, pyimgui) were not initialized after cloning

**Files Affected:**
- `thirdparty/eigen/` (missing)
- `thirdparty/in3d/thirdparty/pyimgui/` (missing)

**Solution:**
Initialize all git submodules:
```bash
git submodule update --init --recursive
```

This downloads:
- Eigen linear algebra library
- pyimgui for in3d visualization

---

## Error 5: Incorrect Include Paths on Windows

**Error:**
```
fatal error C1083: Cannot open include file: 'Eigen/Sparse': No such file or directory
```
(Even after submodule initialization)

**Cause:** setup.py used forward slashes in paths which weren't being correctly resolved on Windows by NVCC

**Files Affected:**
- `setup.py` lines 11-14

**Solution:**
Modified setup.py to use Path() for proper Windows path handling:
```python
from pathlib import Path

# Convert to absolute paths with proper native separators for Windows
include_dirs = [
    str(Path(ROOT) / "mast3r_slam" / "backend" / "include"),
    str(Path(ROOT) / "thirdparty" / "eigen"),
]
```

---

## Error 6: Eigen/CUDA Compatibility Warnings

**Error:**
```
Warning #20014-D: calling a __host__ function from a __host__ __device__ function is not allowed
  detected during instantiation of Eigen::SparseMatrix operations
```

**Cause:** NVCC 12.8 is strict about Eigen's SparseMatrix operations when compiling CUDA code. Eigen's diagonal().array() operations at `gn_kernels.cu:137` are marked as __host__ only but being used in __host__ __device__ context.

**Files Affected:**
- `mast3r_slam/backend/src/gn_kernels.cu` line 137
- `setup.py` NVCC compiler flags

**Root Cause:**
The code at line 137:
```cpp
L.diagonal().array() += ep + lm * L.diagonal().array();
```
Uses Eigen SparseMatrix operations that aren't fully CUDA-compatible. This is CPU-side code but NVCC's strict checking flags it as a potential device code issue.

**Solution:**
Added diagnostic suppression flags to setup.py NVCC arguments:
```python
extra_compile_args["nvcc"] = [
    "-O3",
    "-Xcudafe", "--diag_suppress=20014",  # Suppress __host__/__device__ warnings for Eigen
    "-Xcudafe", "--diag_suppress=177",     # Suppress unreferenced label warnings
    "-gencode=arch=compute_60,code=sm_60",
    # ... other gencode flags
]
```

These flags tell NVCC to suppress the Eigen-related warnings that are safe to ignore in this context.

---

## Error 7: Additional NVCC Warnings

**Warning:**
```
warning #177-D: label "https" was declared but never referenced
```

**Cause:** Comment or code artifact creating an unreferenced label

**Solution:**
Suppressed with `-Xcudafe --diag_suppress=177` (included in Error 6 fix above)

---

## Key Success Factors

1. **Always use `TORCH_CUDA_ARCH_LIST`** when building CUDA extensions:
   ```bash
   set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
   ```

2. **Use `--no-deps` flag** to prevent dependency upgrades during CUDA extension installation:
   ```bash
   pip install --no-build-isolation --no-deps -e .
   ```

3. **Initialize submodules** before building:
   ```bash
   git submodule update --init --recursive
   ```

4. **Use Path() for Windows paths** in setup.py to ensure proper path resolution

5. **Add appropriate NVCC diagnostic suppressions** for Eigen/CUDA compatibility

---

## Common Build Commands

### Full rebuild sequence:
```bash
# Activate environment
activate_venv.bat

# Build lietorch
cd temp_lietorch
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
pip install --no-build-isolation -e .
cd ..

# Build curope
cd thirdparty\mast3r\dust3r\croco\models\curope
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
pip install --no-build-isolation --no-deps -e .
cd ..\..\..\..\..\..

# Build in3d
pip install -e thirdparty\in3d

# Build mast3r_slam
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
pip install --no-build-isolation -e .
```

### Clean rebuild:
```bash
# Remove build artifacts
rmdir /s /q build
rmdir /s /q *.egg-info

# Rebuild
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
pip install --no-build-isolation -e .
```

---

## Architecture-Specific Notes

### RTX 5090 (sm_120) Support

The RTX 5090 requires:
- CUDA 12.8+ (native sm_120 support)
- PyTorch 2.8.0+cu128 (includes sm_120 in arch list)
- Explicit `TORCH_CUDA_ARCH_LIST` including `12.0`

Verify sm_120 support:
```python
python -c "import torch; print('sm_120 support:', 'sm_120' in torch.cuda.get_arch_list())"
```

Expected output:
```
sm_120 support: True
```

---

## File Modifications Summary

### Files Modified for Windows Compatibility:

1. **setup.py** (2 changes)
   - Lines 11-15: Fixed include paths using Path()
   - Lines 45-48: Added NVCC diagnostic suppressions

2. **temp_lietorch/pyproject.toml** (1 change)
   - Removed duplicate version and description fields

### Files NOT Modified (Already Patched):

These files were already patched for PyTorch 2.8.0 compatibility:
- `thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu`
- `mast3r_slam/backend/src/matching_kernels.cu`
- `mast3r_slam/backend/src/gn_kernels.cu`
- `thirdparty/mast3r/mast3r/model.py`

---

## Verification After Build

Test all imports:
```bash
python -c "import torch; import lietorch; import curope; import mast3r; import mast3r_slam; import in3d; print('All imports successful!')"
```

Verify CUDA device:
```bash
python -c "import torch; print('Device:', torch.cuda.get_device_name(0)); print('sm_120:', 'sm_120' in torch.cuda.get_arch_list())"
```

---

## References

- Original issue tracker: [GitHub Issues](https://github.com/jessesep/MASt3R-SLAM-WINBUILD/issues)
- PyTorch CUDA Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
- Eigen Documentation: https://eigen.tuxfamily.org/
- NVCC Documentation: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
