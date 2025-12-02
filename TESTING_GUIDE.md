# MASt3R-SLAM Testing Guide

## ✅ Installation Complete!

Your MAst3R-SLAM installation has been completed and verified. All tests passed successfully.

---

## 🎮 GUI/Visualization

**YES!** The visualization GUI is built-in and enabled by default!

### Visualization Features
- ✅ Real-time 3D reconstruction viewer
- ✅ Camera trajectory display
- ✅ Point cloud visualization
- ✅ Keyframe display
- ✅ Interactive controls

### How to Use Visualization

**Enable (default):**
```bash
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz --config config/base.yaml
```

**Disable (headless mode):**
```bash
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz --config config/base.yaml --no-viz
```

The visualization runs in a separate process using `mast3r_slam/visualization.py`.

---

## 📦 Sample Datasets Available

The repository includes download scripts for several benchmark datasets:

### 1. TUM RGB-D Dataset (Smallest and Recommended for Testing)
**Location:** `scripts/download_tum.sh`
**Sequences:**
- `freiburg1_xyz` - Simple XYZ motion (SMALLEST, ~460 MB)
- `freiburg1_360` - 360° rotation
- `freiburg1_desk` - Desktop scene
- `freiburg1_floor` - Floor scene
- `freiburg1_room` - Room scene
- `freiburg1_teddy` - Teddy bear
- Plus more...

**Quick Download (Windows-compatible):**
```bash
# Download smallest sequence (xyz - recommended for first test)
mkdir -p datasets/tum
cd datasets/tum
curl -O https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
cd ../..
```

**Dataset Info:**
- Format: TUM RGB-D
- Size: ~460 MB (xyz sequence)
- Frames: ~800 images
- Resolution: 640x480
- Has ground truth trajectory

### 2. EuRoC MAV Dataset
**Location:** `scripts/download_euroc.sh`
**Sequences:** Various indoor/outdoor flying robot sequences

### 3. ETH3D Dataset
**Location:** `scripts/download_eth3d.sh`
**Sequences:** High-quality indoor scenes

### 4. 7-Scenes Dataset
**Location:** `scripts/download_7_scenes.sh`
**Sequences:** Indoor relocalization scenarios

---

## 🚀 Quick Test with Sample Data

### Step 1: Download TUM XYZ (Smallest Dataset)

```bash
cd C:\Users\5090\MASt3R-SLAM-WINBUILD

# Create datasets directory
mkdir datasets
mkdir datasets\tum
cd datasets\tum

# Download smallest TUM sequence (~460 MB)
curl -O https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz

# Extract
tar -xzf rgbd_dataset_freiburg1_xyz.tgz

cd ..\..
```

**Alternative (using PowerShell):**
```powershell
# Download using PowerShell
New-Item -ItemType Directory -Force datasets\tum
Invoke-WebRequest -Uri "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz" -OutFile "datasets\tum\rgbd_dataset_freiburg1_xyz.tgz"

# Extract with tar (Windows 10+ has built-in tar)
tar -xzf datasets\tum\rgbd_dataset_freiburg1_xyz.tgz -C datasets\tum
```

### Step 2: Download Model Checkpoints

MASt3R-SLAM requires pre-trained model weights. Download from Hugging Face:

```bash
# Create checkpoints directory
mkdir checkpoints
cd checkpoints

# Download MASt3R model (use curl or browser)
# Visit: https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
# Download: MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

cd ..
```

**Using Python (easier):**
```python
from huggingface_hub import hf_hub_download

# Download MASt3R checkpoint
hf_hub_download(
    repo_id="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    filename="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    local_dir="checkpoints"
)
```

### Step 3: Run SLAM with GUI

```bash
# Activate environment
venv\Scripts\activate

# Run SLAM with visualization
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz --config config/base.yaml
```

**Expected behavior:**
1. Loads MASt3R model
2. Processes RGB-D frames
3. Opens visualization window showing:
   - 3D point cloud
   - Camera trajectory
   - Current keyframes
4. Saves trajectory output

---

## ⚙️ Configuration Files

Located in `config/` directory:

### `base.yaml` - Default configuration
- Standard SLAM parameters
- Good for most datasets
- Balanced speed/accuracy

### `calib.yaml` - With camera calibration
- Use when you have accurate intrinsics
- Better accuracy

### `eval_no_calib.yaml` - Evaluation without calibration
- For benchmarking
- Assumes unknown intrinsics

### `intrinsics.yaml` - Intrinsic parameters
- Camera calibration settings
- Focal length, principal point

### `more_keyframes.yaml` - More keyframe selection
- Denser reconstruction
- Higher memory usage

### `eth3d.yaml` - ETH3D dataset specific
- Tuned for ETH3D benchmark

---

## 🎯 Test Commands

### Basic Test (With GUI)
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_xyz \
    --config config/base.yaml
```

### Save Trajectory
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_xyz \
    --config config/base.yaml \
    --save-as results/trajectory_xyz.txt
```

### Headless Mode (No GUI)
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_xyz \
    --config config/base.yaml \
    --no-viz
```

### With Custom Camera Calibration
```bash
python main.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_xyz \
    --config config/calib.yaml \
    --calib config/intrinsics.yaml
```

---

## 📊 Evaluation

After running SLAM, evaluate against ground truth:

```bash
# Using evo (already installed)
evo_ape tum \
    datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
    results/trajectory_xyz.txt \
    --plot --verbose --align
```

**Metrics provided:**
- Absolute Pose Error (APE)
- Relative Pose Error (RPE)
- Trajectory visualization
- Error statistics

---

## 🎮 GUI Controls

Once the visualization window opens:

### Mouse Controls
- **Left Click + Drag:** Rotate view
- **Right Click + Drag:** Pan view
- **Scroll Wheel:** Zoom in/out

### Keyboard Shortcuts
- **Space:** Pause/resume processing
- **R:** Reset camera view
- **S:** Save current view
- **Q/Esc:** Quit

### Visualization Elements
- **Blue points:** Current 3D reconstruction
- **Green line:** Camera trajectory
- **Red points:** Current keyframes
- **Yellow:** Active tracking features

---

## 📁 Expected Directory Structure After Setup

```
MASt3R-SLAM-WINBUILD/
├── venv/                          # Virtual environment
├── checkpoints/                   # Model weights
│   └── MASt3R_ViTLarge_*.pth     # Downloaded model
├── datasets/                      # Downloaded datasets
│   └── tum/
│       └── rgbd_dataset_freiburg1_xyz/
│           ├── rgb/               # Color images
│           ├── depth/             # Depth images
│           ├── rgb.txt            # RGB timestamps
│           ├── depth.txt          # Depth timestamps
│           └── groundtruth.txt    # GT trajectory
├── config/                        # Configuration files
│   ├── base.yaml
│   ├── calib.yaml
│   └── ...
├── results/                       # Output trajectories
│   └── trajectory_xyz.txt
├── main.py                        # Main SLAM script
└── mast3r_slam/                   # Source code
    ├── visualization.py           # GUI code
    └── ...
```

---

## 🐛 Troubleshooting

### Model checkpoint not found
```
Error: Could not find checkpoint...
```
**Solution:** Download model weights to `checkpoints/` directory

### Dataset not found
```
Error: Dataset path does not exist
```
**Solution:** Verify dataset path, ensure it's extracted correctly

### GUI doesn't open
```
Warning: No display found
```
**Solution:**
- On WSL: Install X server (VcXsrv, Xming)
- On Windows: Should work directly
- Use `--no-viz` for headless mode

### Out of memory
```
CUDA out of memory
```
**Solution:**
- Reduce image resolution in config: `img_downsample: 2`
- Process fewer keyframes: `window_size: 100000`

### Slow performance
**Solution:**
- Check GPU usage: `nvidia-smi`
- Enable FP16 in config (if supported)
- Reduce `max_iters` in tracking

---

## 📝 Next Steps

1. ✅ **Installation complete** - Done!
2. ⏭️ **Download sample dataset** - Use commands above
3. ⏭️ **Download model weights** - Required for inference
4. ⏭️ **Run first test** - TUM xyz sequence
5. ⏭️ **Evaluate results** - Compare with ground truth
6. ⏭️ **Try other datasets** - EuRoC, ETH3D, etc.
7. ⏭️ **Customize configs** - Tune for your data

---

## 💡 Tips for Best Results

### For RTX 5090 (Your GPU)
- Use high resolution: `img_downsample: 1`
- Increase iterations: `max_iters: 100`
- Enable more keyframes: use `config/more_keyframes.yaml`
- Process full sequences without subsampling

### For Faster Testing
- Use `img_downsample: 2` or `4`
- Reduce `max_iters` to `20`
- Enable `subsample: 2` in config

### For Best Accuracy
- Use `config/calib.yaml` with accurate intrinsics
- Disable image downsampling
- Increase optimization iterations
- Use dense keyframe selection

---

## 📚 Additional Resources

- **TUM Dataset:** https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- **EuRoC Dataset:** https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
- **ETH3D Dataset:** https://www.eth3d.net/
- **MASt3R Model:** https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

---

**Ready to test!** Start with downloading the TUM xyz dataset (~460 MB) and model checkpoints, then run your first SLAM test with the built-in GUI visualization.

---

*Last Updated: December 2, 2025*
*System: Windows 11, Python 3.11.9, CUDA 12.8, RTX 5090*
