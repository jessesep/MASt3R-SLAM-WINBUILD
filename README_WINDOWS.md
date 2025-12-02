# MASt3R-SLAM Windows 11 Build

Native Windows 11 build of MASt3R-SLAM with NVIDIA RTX 5090 (Blackwell) support and TouchDesigner integration.

## Quick Start

### Prerequisites
- Windows 11
- NVIDIA RTX 5090 or RTX 50xx series GPU
- NVIDIA Driver 580.95.05 or newer
- CUDA Toolkit 12.8+
- Python 3.11
- Visual Studio 2022 with C++ Build Tools
- Git for Windows

### Automated Setup

1. **Clone this repository:**
   ```cmd
   git clone https://github.com/jessesep/MASt3R-SLAM-WINBUILD.git --recursive
   cd MASt3R-SLAM-WINBUILD
   ```

2. **Run the setup script:**
   ```cmd
   setup_windows.bat
   ```
   This will:
   - Create a Python virtual environment
   - Install PyTorch 2.8.0 with CUDA 12.8
   - Verify CUDA and sm_120 support

3. **Follow the remaining steps in `WINDOWS_BUILD_GUIDE.md`:**
   - Download model checkpoints
   - Apply PyTorch 2.8.0 compatibility patches
   - Build CUDA extensions (lietorch, curope, mast3r_slam)
   - Run verification tests

### Daily Usage

To activate the environment for running MASt3R-SLAM:

```cmd
activate_venv.bat
```

This script will:
- Activate the Python virtual environment
- Set CUDA paths
- Set TORCH_CUDA_ARCH_LIST for builds

## Documentation

- **[WINDOWS_BUILD_GUIDE.md](WINDOWS_BUILD_GUIDE.md)** - Complete step-by-step build instructions
- **[M-SLAM_BLACKWELL_SETUP.md](M-SLAM_BLACKWELL_SETUP.md)** - Original Linux/Ubuntu setup guide (reference)
- **[README.md](README.md)** - Original MASt3R-SLAM README

## What's Different in This Build?

This Windows build includes:

1. **Native Windows Support**
   - venv-based Python environment (instead of conda)
   - Windows batch scripts for activation and setup
   - Windows-specific path handling

2. **RTX 5090 Blackwell Support**
   - CUDA 12.8 with sm_120 architecture support
   - PyTorch 2.8.0 with native Blackwell kernels
   - Explicit TORCH_CUDA_ARCH_LIST configuration

3. **PyTorch 2.8.0 Compatibility**
   - All necessary patches for PyTorch 2.6+ API changes
   - Updated `.scalar_type()` and `.norm()` calls
   - `weights_only=False` for checkpoint loading

4. **TouchDesigner Integration** (Coming Soon)
   - Real-time point cloud streaming to TouchDesigner
   - OSC/UDP communication protocol
   - Example TouchDesigner network files

## File Structure

```
MASt3R-SLAM-WINBUILD/
├── README_WINDOWS.md              # This file
├── WINDOWS_BUILD_GUIDE.md         # Detailed build instructions
├── setup_windows.bat              # Automated setup script
├── activate_venv.bat              # Environment activation script
├── venv/                          # Python virtual environment (created by setup)
├── checkpoints/                   # Model checkpoints (download required)
├── config/                        # Configuration files
├── mast3r_slam/                   # Main SLAM implementation
├── thirdparty/                    # Dependencies (mast3r, in3d)
│   ├── mast3r/
│   │   └── dust3r/croco/models/curope/  # CUDA extension
│   └── in3d/
└── scripts/                       # Dataset download scripts
```

## Common Issues

### "CUDA error: no kernel image is available"
- **Cause:** CUDA extensions built without sm_120 support
- **Fix:** Rebuild with `set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;12.0`

### "DLL load failed while importing"
- **Cause:** CUDA DLLs not in PATH
- **Fix:** Run `activate_venv.bat` to set paths correctly

### PyTorch upgraded to 2.9.1
- **Cause:** Installing dependencies without `--no-deps`
- **Fix:** Reinstall PyTorch 2.8.0 and rebuild extensions with `--no-deps`

## Performance

Expected performance on RTX 5090:
- Real-time processing: 10-15 FPS (1920x1080)
- SLAM initialization: ~2-3 seconds
- Memory usage: ~8-12GB VRAM

## Contributing

This is a Windows-specific build. For the original project, see:
- Original repo: https://github.com/rmurai0610/MASt3R-SLAM
- Fork: https://github.com/jessesep/MASt3R-SLAM

## License

See [LICENSE.md](LICENSE.md) for details.

## Citation

If you use this work, please cite the original paper:

```bibtex
@inproceedings{murai2024_mast3rslam,
  title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
  author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```

## Acknowledgements

- Riku Murai, Eric Dexheimer, Andrew J. Davison - Original MASt3R-SLAM authors
- [MASt3R](https://github.com/naver/mast3r) team
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) team
- NVIDIA for Blackwell architecture support
