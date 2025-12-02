# MASt3R-SLAM: Windows Native Build

<p align="center">
  <h1 align="center">MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors</h1>
  <p align="center">
    <a href="https://rmurai.co.uk/"><strong>Riku Murai*</strong></a>
    ·
    <a href="https://edexheim.github.io/"><strong>Eric Dexheimer*</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew J. Davison</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2412.12392">Paper</a> | <a href="https://youtu.be/wozt71NBFTQ">Video</a> | <a href="https://edexheim.github.io/mast3r-slam/">Project Page</a></h3>
</p>

---

## 🎯 What is This Build?

This is a **native Windows 11 port** of MASt3R-SLAM, specifically built and tested for:

- **RTX 5090** (Blackwell architecture, sm_120, compute capability 12.0)
- **CUDA 12.8** with native PyTorch 2.8.0+cu128
- **Windows 11** with Python 3.11 (virtual environment)
- **32GB GPU VRAM** (tested configuration)

**Key Features:**
- ✅ All CUDA extensions compiled for Blackwell (sm_120)
- ✅ All tests passing (7/7 components verified)
- ✅ Native Windows execution (no WSL required)
- ✅ Dataset loading fixes applied (TUM, EuRoC, ETH3D)
- ✅ OSC streaming module for TouchDesigner integration
- 🚧 Enhanced GUI launcher (in development)

---

## 🔥 Why This Fork?

### Original MASt3R-SLAM
The original MASt3R-SLAM is designed for Ubuntu/Linux with:
- PyTorch 2.5.1
- CUDA compute capability up to 9.0 (Ada Lovelace - RTX 40 series)
- Conda-based environment

### This Windows Build
This fork includes:

**Hardware Support:**
- NVIDIA Blackwell architecture (sm_120, RTX 50 series)
- Native CUDA 12.8 support (no compatibility mode)

**Software Patches:**
- PyTorch 2.8.0 API compatibility fixes (4 patches)
  - `.scalar_type()` → `.dtype`
  - `.norm()` parameter updates
  - `weights_only=False` for safe model loading
- Dataset loading fix: `skiprows=0` → `comments="#"` for TUM/ETH3D/EuRoC datasets
- Build instructions with `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"`

**Windows-Specific:**
- OpenGL symlink fixes for visualization
- Native Windows shell compatibility (cmd/PowerShell)
- Python venv (not conda)
- Multiprocessing stability fixes

**Enhanced Features:**
- `osc_streamer.py` - Real-time OSC streaming for TouchDesigner/Max/MSP
  - Camera pose streaming (30 Hz)
  - Point cloud streaming (5 Hz, chunked)
  - SLAM status updates (2 Hz)
  - Connection testing and statistics
- Enhanced GUI launcher (in development)
  - Dataset source selection
  - Output path control
  - OSC/Network configuration
  - Real-time monitoring

---

## 🚀 Build Status

### Hardware Configuration
```
GPU:     NVIDIA GeForce RTX 5090 (32 GB)
Arch:    Blackwell (sm_120, compute capability 12.0)
CUDA:    12.8
Driver:  567.72
OS:      Windows 11
```

### Software Environment
```
Python:          3.11
PyTorch:         2.8.0+cu128
CUDA Toolkit:    12.8
Virtual Env:     venv (not conda)
Installation:    Native Windows build
```

### Component Status
- ✅ Core PyTorch CUDA support
- ✅ lietorch (SE3 operations on GPU)
- ✅ curope (MASt3R rotation encodings)
- ✅ mast3r_slam_backends (CUDA backend extensions)
- ✅ MASt3R model loading
- ✅ Model checkpoints (2.9 GB)
- ✅ Dataset loading (TUM, EuRoC, ETH3D, 7-Scenes)
- ✅ OSC streaming module
- ✅ Test suite (7/7 passing)

### Documentation
- `TEST_OVERVIEW.md` - Complete testing documentation
- `SLAM_OUTPUT_GUIDE.md` - PLY output format and execution guide
- `ENHANCED_GUI_DESIGN.md` - GUI design specification
- `GUI_IMPLEMENTATION_GUIDE.md` - Implementation guide
- `IMPLEMENTATION_STATUS.md` - Current development status

---

## 📦 Installation

This build is already compiled and configured. If you have this exact setup (RTX 5090, Windows 11, CUDA 12.8), you can use it directly.

### Prerequisites
- Windows 11
- NVIDIA GeForce RTX 5090 (or other Blackwell GPU)
- CUDA 12.8 installed
- Driver 567.72 or later
- Python 3.11

### Quick Start
```cmd
cd C:\Users\5090\MASt3R-SLAM-WINBUILD

REM Activate virtual environment
.\venv\Scripts\activate.bat

REM Verify installation
python quick_test.py

REM Run SLAM on TUM dataset
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_xyz --config config\base.yaml --no-viz
```

### Important: Shell Environment
**Use Windows Command Prompt or PowerShell for SLAM runs**

❌ **DO NOT use Git Bash/MINGW64** for running full SLAM:
- Causes segmentation faults due to CUDA/PyTorch interaction issues
- DLL loading differences between MINGW64 and native Windows

✅ **Use Windows native shells:**
- Command Prompt (cmd.exe)
- PowerShell

**Note:** Git Bash/MINGW64 works fine for testing, imports, and simple scripts - just not for full SLAM pipeline execution.

---

## 🎬 Usage

### Basic SLAM Run
```cmd
REM From Windows Command Prompt or PowerShell
cd C:\Users\5090\MASt3R-SLAM-WINBUILD
.\venv\Scripts\activate.bat

REM Run on TUM dataset (no visualization)
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_xyz --config config\base.yaml --no-viz

REM Run with visualization (ImGui window)
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_xyz --config config\base.yaml

REM Custom output path
python main.py --dataset datasets\tum\rgbd_dataset_freiburg1_xyz --config config\base.yaml --no-viz --save-as results\my_run
```

### Output Files
SLAM generates two files in `results/<sequence_name>/`:
1. **PLY point cloud** - Binary colored point cloud (x,y,z,r,g,b)
   - Typical size: 10-500 MB depending on sequence length
   - Viewable in MeshLab, CloudCompare, Open3D, Blender, TouchDesigner
2. **Trajectory file** - TUM RGB-D format (timestamp, tx, ty, tz, qx, qy, qz, qw)
   - Camera poses in world frame
   - Compatible with evo toolkit for evaluation

See `SLAM_OUTPUT_GUIDE.md` for detailed output documentation.

### OSC Streaming (TouchDesigner Integration)

The build includes real-time OSC streaming for integration with TouchDesigner, Max/MSP, and other creative tools:

```python
from osc_streamer import OSCStreamer

# Create OSC streamer
streamer = OSCStreamer("127.0.0.1", 9000, enabled=True)

# Test connection
if streamer.test_connection():
    print("OSC ready!")
```

**OSC Messages:**
- `/slam/camera/pose` - 7-DOF camera pose [tx, ty, tz, qx, qy, qz, qw]
- `/slam/pointcloud/chunk` - Point cloud data (chunked, 1000 points/chunk)
- `/slam/status` - SLAM state, FPS, point count, confidence
- `/slam/keyframe/new` - Keyframe events
- `/slam/tracking/quality` - Tracking quality metrics
- `/slam/complete` - Completion signal with output path

**Standalone Test:**
```cmd
python osc_streamer.py
```

See `IMPLEMENTATION_STATUS.md` for integration examples and TouchDesigner setup.

---

## 🧪 Testing

### Quick Test (30 seconds)
```cmd
.\venv\Scripts\activate.bat
python quick_test.py
```

Tests 7 components:
1. Core imports (numpy, torch, opencv, scipy, matplotlib)
2. CUDA availability and GPU detection
3. lietorch (SE3 operations)
4. curope (rotation encodings)
5. mast3r_slam_backends
6. MASt3R model imports
7. Checkpoint files

### Full Test Run
```cmd
python test_installation.py
```

Comprehensive test including CUDA tensor operations, LieTorch SE3 operations, and more. Test logs saved to `test_logs/`.

**Note:** If tests fail, clear Python bytecode cache:
```cmd
rmdir /s /q mast3r_slam\__pycache__
python quick_test.py
```

---

## 📊 Datasets

### TUM RGB-D Dataset
Already included in `datasets/tum/rgbd_dataset_freiburg1_xyz/` (798 frames)

Download additional sequences:
```bash
# Use Git Bash for download scripts
bash ./scripts/download_tum.sh
```

### Other Datasets
- **7-Scenes:** `bash ./scripts/download_7_scenes.sh`
- **EuRoC:** `bash ./scripts/download_euroc.sh`
- **ETH3D:** `bash ./scripts/download_eth3d.sh`

### RealSense Camera (Live)
```cmd
python main.py --dataset realsense --config config\base.yaml
```

### MP4 Video or Image Folder
```cmd
python main.py --dataset path\to\video.mp4 --config config\base.yaml
python main.py --dataset path\to\image\folder --config config\base.yaml
```

---

## 🔧 Troubleshooting

### Issue: Segmentation Fault
**Cause:** Running from Git Bash/MINGW64

**Solution:** Use Windows Command Prompt or PowerShell (see Shell Environment section)

### Issue: IndexError when loading dataset
**Cause:** Python bytecode cache using old code

**Solution:**
```cmd
rmdir /s /q mast3r_slam\__pycache__
python main.py <your args>
```

### Issue: CUDA out of memory
**Solution:** Reduce image size in config or downsample dataset
```yaml
# In config/base.yaml
img_downsample: 2  # Downsample images by factor of 2
```

### Issue: Missing checkpoints
**Solution:** Download model weights:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

---

## 🚧 In Development

### Enhanced GUI Launcher
Currently implementing ImGui-based launcher with:
- Dataset source selection (TUM, EuRoC, ETH3D, RealSense, Webcam, MP4, folder)
- Output path control with browse dialog
- PLY export options (confidence threshold, format, downsampling)
- OSC/Network configuration (IP, port, presets)
- Real-time monitoring (progress, FPS, GPU usage, VRAM)
- Configuration file management

See `ENHANCED_GUI_DESIGN.md` for complete specification.

### Planned Features
- NDI video input support
- Batch processing
- Trajectory comparison tools
- Quality assessment automation
- HTML report generation

---

## 📚 Documentation

- **TEST_OVERVIEW.md** - Testing results and verification
- **SLAM_OUTPUT_GUIDE.md** - Output file formats, PLY specification, viewing tools
- **ENHANCED_GUI_DESIGN.md** - Complete GUI design with 5 tabs and 10+ features
- **GUI_IMPLEMENTATION_GUIDE.md** - Implementation guide with code samples
- **IMPLEMENTATION_STATUS.md** - Current development status and next steps
- **osc_streamer.py** - OSC streaming module documentation (docstrings)

---

## 🎥 Creative Pipeline Integration

This build is designed for integration into creative pipelines:

### TouchDesigner
- Real-time OSC streaming of camera poses and point clouds
- Live 3D reconstruction visualization
- PLY import for static geometry
- See `IMPLEMENTATION_STATUS.md` for TouchDesigner setup

### Max/MSP
- OSC message format compatible with Max OSC objects
- Audio-reactive 3D reconstruction

### Unreal Engine / Unity
- PLY export for static mesh import
- Trajectory data for camera animation

### Blender
- PLY import for point cloud visualization and rendering
- Trajectory export for camera path

---

## 🙏 Acknowledgements

This Windows build is based on the original MASt3R-SLAM:
- [MASt3R](https://github.com/naver/mast3r)
- [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)
- [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [ModernGL](https://github.com/moderngl/moderngl)

Additional libraries:
- [python-osc](https://pypi.org/project/python-osc/) for OSC streaming

---

## 📝 Citation

If you use this code in your research, please cite the original MASt3R-SLAM paper:

```bibtex
@inproceedings{murai2024_mast3rslam,
  title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
  author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```

---

## 📧 Contact

For Windows build-specific issues, refer to the documentation files in this repository.

For general MASt3R-SLAM questions, see the [original repository](https://github.com/rmurai0610/MASt3R-SLAM).

---

**Build Version:** Windows Native Build for RTX 5090
**Last Updated:** December 2, 2025
**Status:** Fully Functional - SLAM Ready, GUI In Development
