# MASt3R-SLAM Quick Start Guide

## Installation Verification

### Quick Test
```bash
# Activate virtual environment
cd C:\Users\5090\MASt3R-SLAM-WINBUILD
venv\Scripts\activate

# Test imports
python -c "import mast3r, mast3r_slam, lietorch, torch; print('Installation OK! CUDA:', torch.cuda.is_available())"
```

Expected output:
```
Installation OK! CUDA: True
```

---

## Environment Activation

Every time you want to use MASt3R-SLAM:

```bash
cd C:\Users\5090\MASt3R-SLAM-WINBUILD
venv\Scripts\activate
```

You'll see `(venv)` prefix in your terminal.

---

## Running MASt3R-SLAM

### 1. Download Model Checkpoints

First-time setup - download pre-trained models:

```bash
mkdir checkpoints
cd checkpoints

# Download MASt3R model (visit repository for official links)
# Place downloaded .pth files in this directory
```

### 2. Prepare Your Data

MASt3R-SLAM supports:
- **TUM RGB-D format**
- **EuRoC MAV format**
- **Custom RGB-D sequences**

Example TUM format structure:
```
dataset/
├── rgb/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── depth/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── rgb.txt          # timestamp filename
├── depth.txt        # timestamp filename
└── groundtruth.txt  # optional
```

### 3. Run SLAM

```bash
# Basic usage
python run_slam.py --dataset <path_to_dataset> --config configs/default.yaml

# With visualization
python run_slam.py --dataset <path_to_dataset> --config configs/default.yaml --visualize

# Save trajectory
python run_slam.py --dataset <path_to_dataset> --output trajectory.txt
```

### 4. Evaluate Results

If you have ground truth:

```bash
# Using evo (already installed)
evo_ape tum groundtruth.txt trajectory.txt -v --plot
```

---

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Image resolution
image_width: 640
image_height: 480

# Keyframe selection
keyframe_interval: 5

# Optimization
max_iterations: 20
learning_rate: 1e-4

# GPU settings
device: "cuda:0"
batch_size: 1
```

---

## Troubleshooting

### CUDA Out of Memory
Reduce image resolution or batch size in config:
```yaml
image_width: 320
image_height: 240
batch_size: 1
```

### Slow Performance
Check GPU usage:
```bash
nvidia-smi
```

Should show Python process using GPU. If not, verify CUDA:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

### Module Import Errors
Ensure virtual environment is activated:
```bash
venv\Scripts\activate
```

### Missing Checkpoints
Download required model weights from the official repository.

---

## Directory Structure

```
MASt3R-SLAM-WINBUILD/
├── venv/              # Virtual environment (activate this!)
├── checkpoints/       # Model weights (download here)
├── configs/           # Configuration files
├── datasets/          # Your datasets (create this)
├── outputs/           # Results will be saved here
├── mast3r_slam/       # Source code
└── thirdparty/        # Dependencies
```

---

## Common Commands

```bash
# Activate environment
venv\Scripts\activate

# Deactivate environment
deactivate

# Check installation
python -c "import mast3r_slam; print('OK')"

# List available configs
dir configs\*.yaml

# Run with default settings
python run_slam.py --dataset datasets/your_data

# Run with custom config
python run_slam.py --dataset datasets/your_data --config configs/custom.yaml

# Visualize in real-time
python run_slam.py --dataset datasets/your_data --visualize

# Save output mesh
python run_slam.py --dataset datasets/your_data --save_mesh outputs/mesh.ply
```

---

## GPU Information

Your system:
- **GPU:** NVIDIA GeForce RTX 5090
- **CUDA:** 12.8
- **Memory:** 32 GB GDDR7
- **Compute Capability:** 12.0 (sm_120)

Optimal settings for RTX 5090:
```yaml
# Can handle higher resolutions
image_width: 1280
image_height: 960

# Can process multiple frames
batch_size: 4

# Faster optimization
max_iterations: 50
```

---

## Example Workflows

### Process TUM Dataset
```bash
# Download TUM fr1/xyz dataset (example)
# Extract to datasets/tum_fr1_xyz/

python run_slam.py \
    --dataset datasets/tum_fr1_xyz \
    --config configs/tum.yaml \
    --output outputs/tum_fr1_xyz.txt \
    --visualize

# Evaluate
evo_ape tum datasets/tum_fr1_xyz/groundtruth.txt outputs/tum_fr1_xyz.txt -va --plot
```

### Process Your Own Data
```bash
# 1. Organize your images
datasets/my_data/
├── rgb/
└── depth/

# 2. Run SLAM
python run_slam.py --dataset datasets/my_data --visualize

# 3. Results in outputs/
```

---

## Performance Tips

1. **Use FP16 for faster inference** (if supported):
   ```yaml
   use_fp16: true
   ```

2. **Enable CUDNN benchmarking**:
   ```yaml
   cudnn_benchmark: true
   ```

3. **Adjust keyframe rate** based on motion:
   - Fast motion: `keyframe_interval: 3`
   - Slow motion: `keyframe_interval: 10`

4. **Monitor GPU memory**:
   ```bash
   nvidia-smi -l 1  # Update every second
   ```

---

## Getting Help

- **Installation issues:** See `INSTALLATION_TEST_RESULTS.md`
- **Runtime errors:** Check log files in `outputs/logs/`
- **Configuration:** See examples in `configs/`
- **Dataset format:** Check `docs/dataset_format.md`

---

## Updates

To update MASt3R-SLAM:

```bash
cd C:\Users\5090\MASt3R-SLAM-WINBUILD
git pull
venv\Scripts\activate
pip install -e . --no-deps --force-reinstall
```

---

**Installation Date:** 2025-12-02
**System:** Windows 11, Python 3.11.9, CUDA 12.8
**Status:** ✅ Ready to use
