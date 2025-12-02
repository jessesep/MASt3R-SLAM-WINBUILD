# MASt3R-SLAM Windows GUI Implementation Guide

**Date:** December 2, 2025
**Purpose:** Port Ubuntu GUI launcher to Windows with dataset/source selection
**Status:** Planning Document

---

## Overview

This guide explains how to implement a GUI launcher for MASt3R-SLAM on Windows, similar to the Ubuntu build, with support for:
- Dataset selection (TUM, EuRoC, ETH3D, 7-Scenes)
- Video source selection (RealSense, webcam, video files)
- NDI source integration (to be added)
- Configuration management
- SLAM launch and monitoring

---

## Current State Analysis

### What's Already Available ✅

Based on exploration of the Ubuntu repository at https://github.com/jessesep/MASt3R-SLAM and the current Windows build:

**Visualization Components (Already Present):**
- ✅ `mast3r_slam/visualization.py` - ImGui-based 3D visualization
- ✅ `mast3r_slam/visualization_utils.py` - Visualization helpers
- ✅ `thirdparty/in3d/` - Complete 3D rendering library
- ✅ ImGui integration with ModernGL
- ✅ Real-time 3D point cloud rendering

**Installed Dependencies:**
- ✅ `imgui 2.0.0` - GUI framework
- ✅ `moderngl 5.12.0` - OpenGL rendering
- ✅ `moderngl-window 2.4.6` - Window management
- ✅ `pyrealsense2` - Intel RealSense support
- ✅ `opencv-python` - Video/webcam support

**Data Source Support:**
- ✅ TUM RGB-D datasets
- ✅ EuRoC datasets
- ✅ ETH3D datasets
- ✅ 7-Scenes datasets
- ✅ RealSense cameras
- ✅ Webcams
- ✅ Video files (MP4, AVI, MOV)
- ✅ Image directories

### What's Missing ❌

**From Ubuntu Build:**
- ❌ No GUI launcher/selector (uses command-line only)
- ❌ No file browser dialog
- ❌ No dataset type dropdown
- ❌ No source selection UI
- ❌ No configuration editor GUI

**NDI Support:**
- ❌ No NDI libraries
- ❌ No NDI dataloader class
- ❌ No NDI configuration

**Windows-Specific:**
- ❌ No Windows-native file dialogs
- ❌ No Windows batch launcher

---

## Proposed GUI Architecture

### Option 1: ImGui Launcher (Recommended)

**Rationale:** Already have ImGui installed, consistent with existing visualization

**Components:**
```
launcher.py (New File)
├── ImGui window with tabs:
│   ├── Dataset Tab
│   │   ├── Browse button (file dialog)
│   │   ├── Dataset type dropdown (Auto, TUM, EuRoC, ETH3D, 7-Scenes)
│   │   └── Recent datasets list
│   ├── Live Source Tab
│   │   ├── RealSense radio button
│   │   ├── Webcam radio button (with device list)
│   │   ├── NDI radio button (with source list)
│   │   └── Video file button (file dialog)
│   ├── Config Tab
│   │   ├── Config file dropdown
│   │   ├── Image downsampling slider
│   │   ├── Use CUDA checkbox
│   │   └── Enable visualization checkbox
│   └── Run Tab
│       ├── Launch button
│       ├── Status display
│       └── Stop button
```

**Advantages:**
- ✅ Consistent with existing visualization style
- ✅ No additional dependencies
- ✅ Cross-platform (works on Ubuntu too)
- ✅ GPU-accelerated rendering
- ✅ Modern appearance

**Disadvantages:**
- ⚠️ Requires OpenGL context
- ⚠️ More complex than traditional GUI

### Option 2: PyQt5/PySide6 Launcher

**Rationale:** Native Windows look-and-feel, simpler development

**Components:**
```
launcher_qt.py (New File)
├── QMainWindow
│   ├── QTabWidget
│   │   ├── Dataset Tab
│   │   │   ├── QPushButton (Browse)
│   │   │   ├── QComboBox (Dataset type)
│   │   │   └── QListWidget (Recent)
│   │   ├── Live Source Tab
│   │   │   ├── QRadioButton group
│   │   │   └── QComboBox (Device list)
│   │   ├── Config Tab
│   │   │   └── QFormLayout (settings)
│   │   └── Run Tab
│   │       ├── QPushButton (Launch)
│   │       ├── QTextEdit (Log output)
│   │       └── QProgressBar
```

**Advantages:**
- ✅ Native Windows appearance
- ✅ Built-in file dialogs
- ✅ Easier development
- ✅ Rich widget library
- ✅ Qt Designer for visual design

**Disadvantages:**
- ❌ Large dependency (~100 MB)
- ❌ Different from Ubuntu ImGui style
- ❌ License considerations (LGPL/Commercial)

### Option 3: Gradio Web UI

**Rationale:** Web-based, easy deployment, mobile-friendly

**Components:**
```
launcher_gradio.py (New File)
├── gr.Blocks() interface
│   ├── Dataset selection (file upload or path)
│   ├── Source selection (dropdown)
│   ├── Config editor (text area)
│   ├── Launch button
│   └── Log viewer
```

**Advantages:**
- ✅ Web-based (access from browser)
- ✅ Already installed (gradio 6.0.1)
- ✅ Mobile-friendly
- ✅ Easy to share/deploy

**Disadvantages:**
- ❌ Requires web server
- ❌ Less integrated feel
- ❌ Network latency

---

## Recommended Implementation: ImGui Launcher

Based on analysis, **ImGui launcher** is recommended because:
1. Already installed and working
2. Consistent with existing visualization
3. No additional dependencies
4. Modern, GPU-accelerated interface

---

## Implementation Plan

### Phase 1: Basic ImGui Launcher (1-2 days)

**File:** `launcher.py`

```python
"""
MASt3R-SLAM GUI Launcher for Windows
Uses ImGui for consistent interface with visualization
"""

import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import subprocess
import os
from pathlib import Path

class SLAMLauncher:
    def __init__(self):
        self.dataset_path = ""
        self.dataset_type = "Auto"
        self.config_file = "config/base.yaml"
        self.use_viz = True
        self.source_type = "Dataset"  # Dataset, RealSense, Webcam, NDI, Video
        self.recent_datasets = self.load_recent()

    def load_recent(self):
        """Load recent dataset paths from config"""
        recent_file = Path.home() / ".mast3r_slam" / "recent.txt"
        if recent_file.exists():
            return recent_file.read_text().strip().split('\n')
        return []

    def save_recent(self, path):
        """Save dataset path to recent list"""
        recent_file = Path.home() / ".mast3r_slam" / "recent.txt"
        recent_file.parent.mkdir(exist_ok=True)

        recent = self.recent_datasets
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        recent = recent[:10]  # Keep last 10

        recent_file.write_text('\n'.join(recent))
        self.recent_datasets = recent

    def render_dataset_tab(self):
        """Dataset selection tab"""
        imgui.text("Dataset Selection")
        imgui.separator()

        # Browse button
        if imgui.button("Browse for Dataset Folder", width=300):
            # TODO: Implement file dialog
            # For Windows, use tkinter.filedialog or win32 API
            pass

        # Manual path entry
        changed, self.dataset_path = imgui.input_text(
            "Dataset Path",
            self.dataset_path,
            256
        )

        # Dataset type selection
        imgui.text("Dataset Type:")
        clicked, self.dataset_type = imgui.combo(
            "##dataset_type",
            ["Auto", "TUM", "EuRoC", "ETH3D", "7-Scenes"].index(self.dataset_type),
            ["Auto", "TUM", "EuRoC", "ETH3D", "7-Scenes"]
        )

        # Recent datasets
        imgui.text("\nRecent Datasets:")
        for i, path in enumerate(self.recent_datasets[:5]):
            if imgui.selectable(f"{i+1}. {path}")[0]:
                self.dataset_path = path

    def render_source_tab(self):
        """Live source selection tab"""
        imgui.text("Live Source Selection")
        imgui.separator()

        # Source type radio buttons
        clicked, source_idx = imgui.radio_button("Dataset File",
                                                 ["Dataset", "RealSense", "Webcam", "NDI", "Video"].index(self.source_type))
        if clicked:
            self.source_type = ["Dataset", "RealSense", "Webcam", "NDI", "Video"][source_idx]

        imgui.same_line()
        clicked, source_idx = imgui.radio_button("RealSense Camera",
                                                 ["Dataset", "RealSense", "Webcam", "NDI", "Video"].index(self.source_type))
        if clicked:
            self.source_type = "RealSense"

        imgui.same_line()
        clicked, source_idx = imgui.radio_button("Webcam",
                                                 ["Dataset", "RealSense", "Webcam", "NDI", "Video"].index(self.source_type))
        if clicked:
            self.source_type = "Webcam"

        # Source-specific options
        if self.source_type == "Webcam":
            # TODO: Enumerate webcam devices
            imgui.text("Device: /dev/video0 (or 0 for Windows)")

        elif self.source_type == "NDI":
            imgui.text("NDI Sources:")
            # TODO: Enumerate NDI sources
            imgui.text("  (NDI support not yet implemented)")

    def render_config_tab(self):
        """Configuration tab"""
        imgui.text("SLAM Configuration")
        imgui.separator()

        # Config file selection
        configs = list(Path("config").glob("*.yaml"))
        config_names = [c.name for c in configs]

        current_idx = config_names.index(Path(self.config_file).name) if Path(self.config_file).name in config_names else 0
        clicked, new_idx = imgui.combo("Config File", current_idx, config_names)
        if clicked:
            self.config_file = f"config/{config_names[new_idx]}"

        # Options
        clicked, self.use_viz = imgui.checkbox("Enable 3D Visualization", self.use_viz)

        imgui.text("\nNote: Advanced settings can be edited in config YAML files")

    def render_run_tab(self):
        """Launch tab"""
        imgui.text("Launch SLAM")
        imgui.separator()

        # Status display
        imgui.text("Current Settings:")
        imgui.bullet_text(f"Source: {self.source_type}")
        if self.source_type == "Dataset":
            imgui.bullet_text(f"Path: {self.dataset_path or '(not set)'}")
        imgui.bullet_text(f"Config: {self.config_file}")
        imgui.bullet_text(f"Visualization: {'Enabled' if self.use_viz else 'Disabled'}")

        imgui.separator()

        # Launch button
        if imgui.button("Launch SLAM", width=200, height=40):
            self.launch_slam()

        imgui.same_line()
        if imgui.button("Cancel", width=100, height=40):
            pass  # TODO: Cancel/stop

    def launch_slam(self):
        """Launch SLAM with current settings"""
        if not self.dataset_path and self.source_type == "Dataset":
            print("Error: No dataset path specified")
            return

        # Build command
        cmd = ["python", "main.py"]

        if self.source_type == "Dataset":
            cmd.extend(["--dataset", self.dataset_path])
        elif self.source_type == "RealSense":
            cmd.extend(["--dataset", "realsense"])
        elif self.source_type == "Webcam":
            cmd.extend(["--dataset", "webcam"])

        cmd.extend(["--config", self.config_file])

        if not self.use_viz:
            cmd.append("--no-viz")

        # Save to recent
        if self.dataset_path:
            self.save_recent(self.dataset_path)

        # Launch
        print(f"Launching: {' '.join(cmd)}")
        subprocess.Popen(cmd, cwd=os.getcwd())

    def render(self):
        """Main render loop"""
        imgui.begin("MASt3R-SLAM Launcher", True,
                   imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)

        imgui.set_window_size(800, 600)

        # Tab bar
        if imgui.begin_tab_bar("MainTabs"):
            if imgui.begin_tab_item("Dataset").selected:
                self.render_dataset_tab()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Live Source").selected:
                self.render_source_tab()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Config").selected:
                self.render_config_tab()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Run").selected:
                self.render_run_tab()
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()

def main():
    """Main application loop"""
    if not glfw.init():
        return

    # Create window
    window = glfw.create_window(820, 640, "MASt3R-SLAM Launcher", None, None)
    glfw.make_context_current(window)

    # Setup ImGui
    imgui.create_context()
    impl = GlfwRenderer(window)

    launcher = SLAMLauncher()

    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()
        launcher.render()

        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
```

### Phase 2: File Dialog Integration (1 day)

**Windows File Dialog using tkinter (lightest option):**

```python
from tkinter import Tk, filedialog

def browse_dataset():
    """Open Windows file dialog for dataset selection"""
    root = Tk()
    root.withdraw()  # Hide main window
    root.wm_attributes('-topmost', 1)  # Bring to front

    folder = filedialog.askdirectory(
        title="Select Dataset Folder",
        initialdir=str(Path.cwd() / "datasets")
    )

    root.destroy()
    return folder

def browse_video():
    """Open Windows file dialog for video file selection"""
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    file = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.MOV"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file
```

### Phase 3: NDI Integration (2-3 days)

**NDI SDK for Windows:**

**Install NDI SDK:**
1. Download NDI SDK from https://ndi.video/for-developers/
2. Install to `C:\Program Files\NDI\NDI 6 SDK`
3. Add Python wrapper

**Python NDI Wrapper:**

```python
# Install NDI Python wrapper
pip install ndi-python
```

**NDI Dataloader Class:**

```python
# Add to mast3r_slam/dataloader.py

import NDIlib as ndi

class NDIDataset:
    """NDI Network Source Dataset"""

    def __init__(self, source_name=None):
        """
        Args:
            source_name: NDI source name (None = first available)
        """
        self.source_name = source_name
        self.save_results = False

        # Initialize NDI
        if not ndi.initialize():
            raise RuntimeError("Failed to initialize NDI")

        # Find sources
        self.ndi_find = ndi.find_create_v2()
        if not self.ndi_find:
            raise RuntimeError("Failed to create NDI finder")

        # Wait for sources
        time.sleep(1)
        sources = ndi.find_get_current_sources(self.ndi_find)

        if not sources:
            raise RuntimeError("No NDI sources found")

        # Select source
        if source_name:
            source = next((s for s in sources if s.ndi_name == source_name), None)
            if not source:
                raise ValueError(f"NDI source '{source_name}' not found")
        else:
            source = sources[0]

        print(f"Connecting to NDI source: {source.ndi_name}")

        # Create receiver
        self.ndi_recv = ndi.recv_create_v3()
        ndi.recv_connect(self.ndi_recv, source)

        # Configure
        ndi.recv_set_tally(self.ndi_recv, True)

        # Frame info
        self.frame_count = 0
        self.timestamps = []

        # Get first frame for dimensions
        frame = self._get_next_frame()
        if frame is None:
            raise RuntimeError("Failed to receive first NDI frame")

        self.H, self.W = frame.shape[:2]
        self.K = self._estimate_intrinsics(self.W, self.H)

    def _estimate_intrinsics(self, width, height):
        """Estimate camera intrinsics (assume 60° FOV)"""
        fx = fy = width / (2 * np.tan(np.deg2rad(60) / 2))
        cx = width / 2
        cy = height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def _get_next_frame(self):
        """Get next NDI video frame"""
        video_frame = ndi.VideoFrameV2()

        # Timeout 5 seconds
        frame_type = ndi.recv_capture_v2(self.ndi_recv, video_frame, 5000)

        if frame_type != ndi.FRAME_TYPE_VIDEO:
            return None

        # Convert to numpy array
        frame = np.copy(video_frame.data)
        frame = frame.reshape((video_frame.yres, video_frame.xres, 4))[:, :, :3]  # RGBA -> RGB

        # Free frame
        ndi.recv_free_video_v2(self.ndi_recv, video_frame)

        return frame

    def __len__(self):
        return float('inf')  # Continuous stream

    def __getitem__(self, idx):
        frame = self._get_next_frame()
        if frame is None:
            raise StopIteration

        timestamp = time.time()
        self.timestamps.append(timestamp)
        self.frame_count += 1

        # Convert to torch tensor
        img = torch.from_numpy(frame).float() / 255.0
        img = img.permute(2, 0, 1)  # HWC -> CHW

        return {
            'img': img,
            'K': torch.from_numpy(self.K),
            'timestamp': timestamp,
            'idx': self.frame_count
        }

    def __del__(self):
        """Cleanup NDI resources"""
        if hasattr(self, 'ndi_recv'):
            ndi.recv_destroy(self.ndi_recv)
        if hasattr(self, 'ndi_find'):
            ndi.find_destroy(self.ndi_find)
        ndi.destroy()

def get_ndi_sources():
    """Get list of available NDI sources"""
    if not ndi.initialize():
        return []

    ndi_find = ndi.find_create_v2()
    if not ndi_find:
        return []

    time.sleep(1)  # Wait for discovery
    sources = ndi.find_get_current_sources(ndi_find)

    source_names = [s.ndi_name for s in sources]

    ndi.find_destroy(ndi_find)
    ndi.destroy()

    return source_names
```

**Update launcher.py for NDI:**

```python
def render_source_tab(self):
    """Live source selection tab"""
    # ... existing code ...

    elif self.source_type == "NDI":
        imgui.text("Available NDI Sources:")

        if imgui.button("Refresh NDI Sources"):
            self.ndi_sources = get_ndi_sources()

        if not hasattr(self, 'ndi_sources'):
            self.ndi_sources = get_ndi_sources()

        if self.ndi_sources:
            for i, source in enumerate(self.ndi_sources):
                if imgui.selectable(f"{i+1}. {source}")[0]:
                    self.ndi_source = source
        else:
            imgui.text("  No NDI sources found")
            imgui.text("  Make sure NDI-enabled devices are on the network")
```

### Phase 4: Testing & Polish (1 day)

**Testing Checklist:**
- [ ] Dataset folder selection works
- [ ] Video file selection works
- [ ] RealSense camera detection
- [ ] Webcam detection
- [ ] NDI source enumeration
- [ ] Config file switching
- [ ] SLAM launches correctly
- [ ] Recent datasets saved/loaded
- [ ] Window positioning and sizing
- [ ] Error handling and user feedback

---

## File Structure

```
MASt3R-SLAM-WINBUILD/
├── launcher.py                 # NEW: ImGui GUI launcher
├── launcher.bat                # NEW: Windows batch launcher
├── gui/                        # NEW: GUI-specific modules
│   ├── __init__.py
│   ├── dialogs.py              # File dialogs
│   ├── ndi_utils.py            # NDI helper functions
│   └── recent_manager.py       # Recent datasets manager
├── main.py                     # Existing CLI entry point
├── mast3r_slam/
│   ├── dataloader.py           # MODIFY: Add NDIDataset class
│   ├── visualization.py        # Existing visualization
│   └── ...
└── config/
    └── ...
```

---

## Dependencies

**Already Installed:**
- ✅ imgui 2.0.0
- ✅ moderngl 5.12.0
- ✅ moderngl-window 2.4.6
- ✅ pyrealsense2
- ✅ opencv-python

**Need to Install:**
- ❌ ndi-python (for NDI support)
- ❌ pyglfw (if not already present)

**Installation:**
```bash
pip install ndi-python pyglfw
```

---

## Launch Methods

### Method 1: Python Script
```bash
python launcher.py
```

### Method 2: Batch File
```cmd
REM launcher.bat
@echo off
call venv\Scripts\activate.bat
python launcher.py
```

### Method 3: Desktop Shortcut
Create shortcut to `launcher.bat` for easy access

---

## Future Enhancements

**Phase 5+ (Optional):**
- [ ] Configuration editor GUI (edit YAML in-app)
- [ ] Live SLAM monitoring dashboard
- [ ] Result viewer (trajectory plots, point cloud preview)
- [ ] Batch processing (multiple datasets)
- [ ] Remote SLAM (network dataset streaming)
- [ ] TouchDesigner OSC output toggle
- [ ] Point cloud export options
- [ ] Benchmark comparison tools

---

## Summary

**Current Status:**
- ✅ All GUI dependencies installed
- ✅ Visualization components present
- ✅ Multi-source dataloader architecture ready

**Implementation Time Estimate:**
- Basic ImGui launcher: 1-2 days
- File dialogs: 1 day
- NDI integration: 2-3 days
- Testing & polish: 1 day
- **Total: 5-7 days**

**Priority Implementation Order:**
1. Basic ImGui launcher with dataset selection
2. File browser integration
3. Config selection
4. RealSense/Webcam support
5. NDI integration (requires SDK)

The Windows build is **ready for GUI implementation** - all required libraries are installed and the architecture supports it!

---

*Document Version: 1.0*
*Last Updated: December 2, 2025*
*Status: Planning Complete - Ready for Implementation*
