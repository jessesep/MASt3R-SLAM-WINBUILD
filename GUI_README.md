# MASt3R-SLAM GUI Launcher

A user-friendly Windows GUI for launching MASt3R-SLAM without command-line hassle.

---

## Quick Start

### Option 1: Double-Click Launch (Easiest)

Simply double-click `launch_gui.bat` to start the GUI.

### Option 2: Create Desktop Shortcut

1. Right-click on `launch_gui.bat`
2. Select "Send to" → "Desktop (create shortcut)"
3. (Optional) Rename the shortcut to "MASt3R-SLAM"
4. (Optional) Right-click the shortcut → Properties → Change Icon

---

## GUI Features

### 🎯 Main Features

- **Dataset Selection**: Browse or select from preset TUM datasets
- **Configuration**: Choose YAML config files
- **Options Panel**:
  - Disable visualization (recommended for Windows)
  - Single-thread mode (stable, no crashes)
  - Threading mode (alternative to multiprocessing)
  - Save results (trajectory and reconstruction)
  - Optional calibration file
- **Live Output Console**: Real-time SLAM output display
- **Process Control**: Start, Stop, and Clear output buttons
- **Status Monitoring**: Visual status indicator

### 📋 Options Explained

#### Disable Visualization (--no-viz)
- **Recommended:** ✅ Always enabled on Windows
- **Why:** Windows visualization can be unstable
- **Effect:** Runs SLAM without 3D viewer

#### Disable Backend (--no-backend)
- **Recommended:** ✅ Enabled for stability
- **Why:** Single-thread mode is most stable on Windows
- **Effect:** Runs tracking only (no global optimization)
- **Trade-off:** Slightly less accurate over long sequences

#### Use Threading (--use-threading)
- **Recommended:** ⚠️ Alternative option
- **Why:** Threading instead of multiprocessing
- **Effect:** May be more stable than multiprocessing on Windows
- **Note:** Mutually exclusive with --no-backend

#### Save Results
- **Recommended:** Optional
- **What it saves:**
  - Camera trajectory (.txt)
  - 3D reconstruction (.ply)
  - Keyframe images
- **Location:** `results/` directory

---

## Usage Guide

### Step 1: Select Dataset

**Option A - Use Preset:**
Click one of the preset buttons:
- "TUM xyz" - Walking in XYZ pattern
- "TUM desk" - Desktop sequence
- "TUM room" - Room sequence

**Option B - Browse:**
Click "Browse..." to select a custom dataset directory

**Dataset Structure:**
```
dataset_folder/
├── rgb/           # RGB images
├── depth/         # Depth images (optional)
└── associations.txt
```

### Step 2: Choose Configuration

Default: `config/base.yaml` (works for most cases)

To use custom config:
1. Click "Browse..." next to Config File
2. Select your YAML config file

### Step 3: Set Options

**Recommended Settings for Windows:**
- ✅ Disable Visualization
- ✅ Disable Backend
- ❌ Use Threading (leave unchecked)
- ✅ Save Results (if you want output files)

### Step 4: Launch SLAM

1. Click **"Start SLAM"** button
2. Watch the output console for progress
3. Wait for completion or click **"Stop SLAM"** to cancel

### Step 5: View Results

If "Save Results" was enabled, find outputs in:
```
results/
├── [dataset_name]/
│   ├── [dataset_name].txt       # Camera trajectory
│   ├── [dataset_name].ply       # 3D point cloud
│   └── keyframes/               # Keyframe images
```

---

## Troubleshooting

### GUI doesn't open
- **Check:** Is virtual environment activated?
- **Fix:** Run `launch_gui.bat` which auto-activates venv

### "Dataset Not Found" error
- **Check:** Does the dataset directory exist?
- **Fix:** Click "Browse..." and select correct directory

### SLAM crashes during run
- **Check:** Are recommended options enabled?
- **Fix:** Enable "Disable Visualization" and "Disable Backend"

### "Module not found" errors
- **Check:** Are all dependencies installed?
- **Fix:**
  ```bash
  venv\Scripts\activate.bat
  pip install -r requirements.txt
  ```

### Output console shows errors
- **Check:** Review error messages in console
- **Common fixes:**
  - Enable `--no-viz` if visualization errors
  - Enable `--no-backend` if multiprocessing errors
  - Check dataset path is correct

---

## Keyboard Shortcuts

- **Ctrl+C** (in console): Stop SLAM process
- **Alt+F4**: Close GUI window

---

## Output Console Tips

### Reading the Output

**Initialization:**
```
[DEBUG 1] Starting main...
[DEBUG 2] set_start_method done
...
[DEBUG 18] Loading MASt3R model...
```

**Tracking:**
```
Frame   0: INIT complete
Frame   1: tracked
Frame   2: tracked
...
```

**Completion:**
```
================================================================================
[SUCCESS] MASt3R-SLAM WORKS ON WINDOWS!
================================================================================
```

### Log Colors

The console uses a terminal theme:
- **Green text** on black background for easy reading
- All output is logged in real-time
- Scroll to view full history

---

## Advanced Usage

### Custom Calibration

If you have camera calibration:
1. Prepare YAML file with intrinsics:
   ```yaml
   width: 640
   height: 480
   calibration:
     fx: 517.3
     fy: 516.5
     cx: 318.6
     cy: 255.3
   ```
2. Click "Browse..." next to Calibration
3. Select your calibration file

### Running Multiple Datasets

1. Complete first dataset
2. Click "Clear Output" to reset console
3. Select new dataset
4. Click "Start SLAM" again

### Batch Processing

For processing multiple datasets automatically, use command-line instead:
```bash
venv\Scripts\activate.bat
python main.py --dataset dataset1 --no-viz --no-backend
python main.py --dataset dataset2 --no-viz --no-backend
```

---

## Performance

### Expected FPS

- **Single-thread mode:** 0.5-1.0 FPS
- **With backend:** 0.3-0.8 FPS (slower but more accurate)
- **Actual speed depends on:**
  - GPU (RTX 5090 is fast)
  - Image resolution
  - Number of features

### Processing Time Examples

| Dataset | Frames | Time (--no-backend) |
|---------|--------|---------------------|
| TUM xyz (50 frames) | 50 | ~1-2 minutes |
| TUM desk (full) | 500+ | ~10-20 minutes |

---

## Technical Details

### What the GUI Does

1. **Validates inputs**: Checks dataset and config exist
2. **Builds command**: Constructs `main.py` command with options
3. **Launches subprocess**: Runs SLAM in separate process
4. **Captures output**: Streams stdout/stderr to console
5. **Monitors process**: Tracks running state and exit code

### GUI vs Command-Line

**GUI Advantages:**
- ✅ User-friendly interface
- ✅ No need to remember commands
- ✅ Visual status monitoring
- ✅ Easy dataset browsing

**Command-Line Advantages:**
- ✅ Scriptable/automatable
- ✅ Can run in background
- ✅ SSH-friendly
- ✅ Better for batch processing

Both use the same underlying `main.py` script!

---

## Files

### Launcher Files
- `launch_gui.bat` - Windows batch launcher
- `slam_launcher.py` - Python GUI application
- `GUI_README.md` - This documentation

### SLAM Files
- `main.py` - Main SLAM pipeline
- `config/base.yaml` - Default configuration
- `datasets/` - Dataset directory
- `results/` - Output directory

---

## Requirements

- Windows 10/11
- Python 3.11+
- Virtual environment with dependencies installed
- CUDA-capable GPU (recommended)

---

## Credits

- **MASt3R-SLAM:** https://github.com/edexheim/MASt3R-SLAM
- **Windows Port:** Claude Code
- **GUI:** Tkinter (Python standard library)

---

## Support

For issues:
1. Check this README
2. Review console output for errors
3. Verify all recommended options are enabled
4. See `WINDOWS_SLAM_SUCCESS.md` for detailed fixes

---

## Version History

**v1.0** (December 2, 2025)
- Initial Windows GUI launcher
- Dataset presets for TUM datasets
- Real-time output console
- Process control (start/stop)
- All Windows-specific options

---

*Last Updated: December 2, 2025*
