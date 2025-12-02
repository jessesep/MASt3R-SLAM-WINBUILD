# Quick Start: GUI Launcher

Get MASt3R-SLAM running on Windows in 3 easy steps!

---

## Step 1: Launch GUI

**Double-click** `launch_gui.bat`

That's it! The GUI will open automatically.

---

## Step 2: Configure

### Recommended Settings (Pre-selected):

✅ **Dataset:** TUM xyz (or click "TUM xyz" preset)
✅ **Config:** config/base.yaml
✅ **Disable Visualization** (checked)
✅ **Disable Backend** (checked)

**Leave these settings as-is for best stability!**

---

## Step 3: Run

Click the big green **"Start SLAM"** button.

Watch the output console for progress:
- Frame 0: INIT complete
- Frame 1-N: tracked
- Done: Success message

---

## That's It!

### What You'll See

```
Frame   0: INIT complete
Frame   1: tracked
Frame   2: tracked
Frame   3: tracked
...
Frame  49: tracked

[SUCCESS] MASt3R-SLAM WORKS ON WINDOWS!
```

### Typical Processing Time

- **50 frames:** 1-2 minutes
- **500 frames:** 10-20 minutes

---

## Troubleshooting

### Problem: GUI doesn't open
**Solution:** Run from Command Prompt:
```cmd
cd C:\Users\5090\MASt3R-SLAM-WINBUILD
launch_gui.bat
```

### Problem: "Dataset not found"
**Solution:**
1. Click "Browse..."
2. Navigate to: `datasets/tum/rgbd_dataset_freiburg1_xyz`

### Problem: SLAM crashes
**Solution:** Make sure these are checked:
- ✅ Disable Visualization
- ✅ Disable Backend

---

## Want More?

- **Full Documentation:** See `GUI_README.md`
- **Technical Details:** See `WINDOWS_SLAM_SUCCESS.md`
- **Test Results:** See `TEST_RESULTS.md`

---

## Advanced: Command-Line Alternative

If you prefer command-line:
```cmd
venv\Scripts\activate.bat
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz ^
               --config config/base.yaml ^
               --no-viz --no-backend
```

But the GUI is easier! 😊

---

*Enjoy SLAM on Windows!*
