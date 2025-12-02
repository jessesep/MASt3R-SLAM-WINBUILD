# MASt3R-SLAM Windows 11 Build Guide for RTX 5090

This guide provides step-by-step instructions for building MASt3R-SLAM natively on Windows 11 with NVIDIA RTX 5090 (Blackwell architecture, sm_120).

## System Requirements

- **OS:** Windows 11
- **GPU:** NVIDIA RTX 5090 (Blackwell architecture, sm_120, compute capability 10.0)
- **CUDA Toolkit:** 12.8+ (for native sm_120 support)
- **Python:** 3.11
- **Compiler:** Visual Studio 2022 with C++ Build Tools

## Pre-Installation Checklist

- [ ] NVIDIA GPU Driver 580.95.05 or newer installed
- [ ] CUDA Toolkit 12.8 installed
- [ ] Visual Studio 2022 with "Desktop development with C++" workload
- [ ] Git for Windows installed
- [ ] Python 3.11 installed (for venv) OR Conda/Miniconda (for conda)

## Installation Steps

### 1. Verify CUDA Installation

```cmd
nvcc --version
```

Expected output: `release 12.8, V12.8.61` or newer

### 2. Create Python Environment

**Option A: Using venv (Recommended for this build)**

```cmd
cd %USERPROFILE%
git clone https://github.com/jessesep/MASt3R-SLAM-WINBUILD.git --recursive
cd MASt3R-SLAM-WINBUILD

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip
```

**Option B: Using Conda**

```cmd
conda create -n mast3r-slam-windows python=3.11 -y
conda activate mast3r-slam-windows
```

**Note:** The rest of this guide assumes you're using venv. If using conda, simply replace `venv\Scripts\activate` with `conda activate mast3r-slam-windows`.

### 3. Install PyTorch 2.8.0 with CUDA 12.8

```cmd
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify installation and sm_120 support:
```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Arch list:', torch.cuda.get_arch_list())"
```

**Expected output:**
```
PyTorch: 2.8.0+cu128
CUDA available: True
Arch list: ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

✅ **Critical:** Confirm `sm_120` is in the arch list - this means native Blackwell support!

### 4. Download Checkpoints

```cmd
:: Make sure you're in the project directory
cd %USERPROFILE%\MASt3R-SLAM-WINBUILD

:: Download model checkpoints (~2.8GB)
mkdir checkpoints
curl -L https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -o checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
curl -L https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -o checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
curl -L https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -o checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
```

### 5. Apply PyTorch 2.8.0 Compatibility Patches

The repository includes patches for PyTorch 2.6+ API changes. These must be applied before building.

**Patch 1: thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu (line ~101)**

Find:
```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type().scalarType(), "rope_2d_cuda", ([&] {
```

Replace with:
```cpp
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda", ([&] {
```

**Patch 2: mast3r_slam/backend/src/matching_kernels.cu (line ~29)**

Find:
```cpp
at::ScalarType scalar_type = D11.type().scalarType();
```

Replace with:
```cpp
at::ScalarType scalar_type = D11.scalar_type();
```

**Patch 3: mast3r_slam/backend/src/gn_kernels.cu (3 occurrences: lines ~802, ~1219, ~1629)**

Find all occurrences of:
```cpp
auto norm = torch::linalg::linalg_norm(dx, 2, 0, false, c10::nullopt);
```
or
```cpp
delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
```

Replace each with:
```cpp
auto norm = dx.flatten().norm();
```
or
```cpp
delta_norm = dx.flatten().norm();
```

**Patch 4: thirdparty/mast3r/mast3r/model.py (line ~24)**

Find:
```python
ckpt = torch.load(model_path, map_location='cpu')
```

Replace with:
```python
ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
```

### 6. Build lietorch with sm_120 Support

lietorch is a dependency that requires CUDA compilation.

```cmd
:: Clone PyTorch 2.6+ compatible fork
cd %TEMP%
git clone https://github.com/hectorpiteau/lietorch.git
cd lietorch
git submodule update --init --recursive

:: Build with explicit architecture list for Blackwell
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
pip install --no-build-isolation -e .
```

**Architecture list explanation:**
- `7.5` = Turing (RTX 20xx series)
- `8.0` = Ampere (A100)
- `8.6` = Ampere (RTX 30xx series)
- `9.0` = Hopper (H100)
- `12.0` = Blackwell (RTX 50xx, RTX PRO 6000) ← **Our target**

Verify installation:
```cmd
python -c "import lietorch; import torch; print('lietorch imported successfully'); x = lietorch.SE3.Identity(1, device='cuda'); print('lietorch CUDA operations work')"
```

### 7. Build CUDA Extensions

**CRITICAL:** Always use `TORCH_CUDA_ARCH_LIST` and `--no-deps` to ensure sm_120 kernels are compiled and dependencies don't get upgraded.

```cmd
cd %USERPROFILE%\MASt3R-SLAM-WINBUILD

:: Build curope CUDA extension
cd thirdparty\mast3r\dust3r\croco\models\curope
if exist build rmdir /s /q build
if exist *.egg-info rmdir /s /q *.egg-info
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
pip install --no-build-isolation --no-deps -e .

:: Install mast3r package metadata
cd %USERPROFILE%\MASt3R-SLAM-WINBUILD
pip install --no-build-isolation -e thirdparty\mast3r --no-deps

:: Fix numpy version (lietorch requires numpy<2)
pip install numpy==1.26.4

:: Build in3d (no CUDA compilation)
pip install -e thirdparty\in3d

:: Build main mast3r_slam package
set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0
pip install --no-build-isolation -e .
```

### 8. Verify Installation

Test all imports:
```cmd
python -c "import torch; import lietorch; import curope; import mast3r; import mast3r_slam; import in3d; print('All imports successful!')"
```

Verify CUDA device:
```cmd
python -c "import torch; print('Device:', torch.cuda.get_device_name(0)); print('sm_120 support:', 'sm_120' in torch.cuda.get_arch_list())"
```

### 9. Run Test

Download a test dataset:
```cmd
:: Download TUM dataset example
bash scripts\download_tum.sh

:: Run on test dataset
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_room\ --config config\calib.yaml
```

For live demo with RealSense camera:
```cmd
python main.py --dataset realsense --config config\base.yaml
```

For video file:
```cmd
python main.py --dataset path\to\video.mp4 --config config\base.yaml
```

## Windows-Specific Notes

### Multiprocessing
Windows uses a different multiprocessing mechanism than Linux. If you encounter shared memory issues, the system will automatically disable multiprocessing (similar to WSL mode).

### TouchDesigner Integration
This build includes hooks for TouchDesigner integration. See `TOUCHDESIGNER_INTEGRATION.md` for details (to be added).

### Path Length Limitations
Windows has a 260-character path limit by default. If you encounter issues, enable long paths:
```cmd
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

## Troubleshooting

### CUDA Error: No Kernel Image Available
**Problem:** CUDA runtime error about missing kernel image for device.

**Solution:** Rebuild CUDA extensions with explicit `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0`.

### Import Error: DLL Load Failed
**Problem:** Python can't find CUDA DLLs.

**Solution:** Ensure CUDA bin directory is in PATH:
```cmd
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%
```

### Build Errors with Visual Studio
**Problem:** Compiler errors during CUDA extension build.

**Solution:** Ensure "Desktop development with C++" workload is installed in Visual Studio 2022.

### PyTorch Version Mismatch
**Problem:** Build process upgrades PyTorch to 2.9.1.

**Solution:** Always use `--no-deps` flag when installing CUDA extensions.

## Performance Notes

- RTX 5090 (sm_120) provides native Blackwell acceleration
- Real-time processing at ~10-15 FPS expected (depends on image resolution)
- Significantly faster than previous generation GPUs due to native sm_120 support

## Key Success Factors

1. ✅ **ALWAYS use `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0`** for ALL CUDA extension builds
2. ✅ **ALWAYS use `--no-deps`** when building CUDA extensions to prevent dependency upgrades
3. ✅ **Apply all 4 PyTorch 2.8.0 compatibility patches** before building
4. ✅ **Use Python 3.11** (tested and verified)
5. ✅ **Verify sm_120 support** in PyTorch arch list before building

## References

- Original MASt3R-SLAM: https://github.com/rmurai0610/MASt3R-SLAM
- Blackwell Setup Guide (Linux): `M-SLAM_BLACKWELL_SETUP.md`
- Paper: https://arxiv.org/abs/2412.12392
- Project Page: https://edexheim.github.io/mast3r-slam/

## License

See `LICENSE.md` for license information.

## Acknowledgements

- Original MASt3R-SLAM authors: Riku Murai, Eric Dexheimer, Andrew J. Davison
- MASt3R, DROID-SLAM, and ModernGL projects
- NVIDIA for Blackwell architecture support
