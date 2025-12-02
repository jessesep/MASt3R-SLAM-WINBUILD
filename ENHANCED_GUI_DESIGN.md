# MASt3R-SLAM Enhanced GUI Design

**Date:** December 2, 2025
**Purpose:** Advanced GUI features for production workflows
**Focus:** Export control, OSC/TouchDesigner integration, real-time monitoring

---

## Overview

This document expands on the basic GUI launcher with professional features for:
- Export path and format control
- OSC output for TouchDesigner/Max/MSP integration
- Real-time monitoring and status
- Advanced configuration
- Workflow automation
- Performance monitoring

---

## Enhanced GUI Layout

### Tab Structure (Expanded)

```
┌─────────────────────────────────────────────────────────────┐
│ MASt3R-SLAM Pro Launcher                          [_][□][×] │
├─────────────────────────────────────────────────────────────┤
│ [Input] [Output] [OSC/Network] [Config] [Monitor] [About]  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Current Tab Content...                                      │
│                                                               │
│                                                               │
│                                                               │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│ Status: Ready          GPU: RTX 5090 (12 GB free)  [Launch] │
└─────────────────────────────────────────────────────────────┘
```

---

## Tab 1: Input (Enhanced Dataset/Source Selection)

```
┌─────────────────────────────────────────────────────────────┐
│ DATA SOURCE                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ Source Type:  ◉ Dataset File   ○ Live Camera   ○ Network    │
│                                                               │
│ ┌─ Dataset Selection ──────────────────────────────────────┐│
│ │                                                           ││
│ │ Path: [C:\datasets\tum\freiburg1_xyz...] [Browse]        ││
│ │                                                           ││
│ │ Type: [Auto Detect ▼]  Format: [TUM RGB-D ▼]            ││
│ │                                                           ││
│ │ Preview:                                                  ││
│ │   • 798 frames found                                     ││
│ │   • Resolution: 640x480                                  ││
│ │   • Ground truth: Available ✓                            ││
│ │   • Estimated processing time: 3-5 minutes               ││
│ │                                                           ││
│ │ Recent Datasets:                                          ││
│ │   1. ★ freiburg1_xyz (Last used: 2 hours ago)           ││
│ │   2. freiburg2_desk (Last used: 1 day ago)              ││
│ │   3. Custom_Recording_001                                ││
│ │                                                           ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Live Camera ────────────────────────────────────────────┐│
│ │ Camera Type: [RealSense D455 ▼]                          ││
│ │ Resolution:  [1280x720 ▼]    FPS: [30 ▼]                ││
│ │ Depth Mode:  [High Accuracy ▼]                           ││
│ │ [Test Camera]  Status: ● Connected                       ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Network Streaming ──────────────────────────────────────┐│
│ │ Protocol: [NDI ▼]  [UDP ▼]  [RTSP ▼]                    ││
│ │                                                           ││
│ │ NDI Sources: [Refresh]                                    ││
│ │   • BlackmagicCam (1920x1080@30)                         ││
│ │   • OBSStudio (1280x720@60)                              ││
│ │   • vMix Output                                           ││
│ │                                                           ││
│ │ Custom Stream URL:                                        ││
│ │ [rtsp://192.168.1.100:8554/stream]                       ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**New Features:**
- ✨ Source preview with frame count and specs
- ✨ Estimated processing time
- ✨ Recent datasets with stars/favorites
- ✨ Camera device enumeration and testing
- ✨ NDI source discovery with specs
- ✨ Custom RTSP/UDP stream support

---

## Tab 2: Output (NEW - Export Control)

```
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT CONFIGURATION                                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ ┌─ Output Directory ───────────────────────────────────────┐│
│ │ Base Path: [C:\MASt3R-Results\]              [Browse]    ││
│ │                                                           ││
│ │ Naming:    ◉ Auto (timestamp)                            ││
│ │            ○ Custom: [my_scan_____]                      ││
│ │                                                           ││
│ │ Preview: C:\MASt3R-Results\2025-12-02_091530_freiburg1\  ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Point Cloud Export ─────────────────────────────────────┐│
│ │ ☑ Export PLY point cloud                                 ││
│ │   Format: [Binary PLY ▼]                                 ││
│ │   Confidence threshold: [──●────] 0.5                    ││
│ │   Color space: [RGB ▼]  Encoding: [8-bit ▼]             ││
│ │                                                           ││
│ │ ☑ Export subsampled version                              ││
│ │   Voxel size: [0.05] meters (for preview/streaming)      ││
│ │                                                           ││
│ │ ☐ Export OBJ mesh (Poisson reconstruction)               ││
│ │   Depth: [9 ▼]  Scale: [1.1 ▼]                          ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Trajectory Export ──────────────────────────────────────┐│
│ │ ☑ Export camera trajectory                               ││
│ │   Format: [TUM (txt) ▼]  [KITTI ▼]  [EuRoC ▼]          ││
│ │                                                           ││
│ │ ☑ Export keyframe poses only                             ││
│ │ ☑ Export covariance estimates                            ││
│ │ ☐ Export in TouchDesigner format (JSON)                  ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Additional Exports ─────────────────────────────────────┐│
│ │ ☑ Save configuration snapshot (config.yaml)              ││
│ │ ☑ Save processing log (log.txt)                          ││
│ │ ☑ Export depth maps (keyframes only)                     ││
│ │ ☑ Export confidence maps                                 ││
│ │ ☐ Export normal maps                                     ││
│ │ ☑ Generate HTML report with visualizations               ││
│ │ ☐ Export to Alembic (.abc) for animation software        ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Real-time Streaming ────────────────────────────────────┐│
│ │ ☑ Stream point cloud during processing                   ││
│ │   Protocol: [OSC ▼]  Update rate: [10 Hz ▼]             ││
│ │   (See OSC/Network tab for destination settings)         ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ Estimated output size: ~150 MB                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**New Features:**
- ✨ Custom output directory selection
- ✨ Automatic or custom naming with preview
- ✨ Multiple point cloud export formats
- ✨ Confidence threshold control
- ✨ Subsampled/preview cloud generation
- ✨ Multiple trajectory formats (TUM, KITTI, EuRoC, JSON)
- ✨ Optional mesh export (Poisson)
- ✨ Depth/confidence/normal map exports
- ✨ HTML report generation
- ✨ Alembic export for animation pipelines
- ✨ Real-time streaming toggle

---

## Tab 3: OSC/Network (NEW - TouchDesigner Integration)

```
┌─────────────────────────────────────────────────────────────┐
│ NETWORK & OSC OUTPUT                                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ ☑ Enable OSC Output                                         │
│                                                               │
│ ┌─ OSC Configuration ──────────────────────────────────────┐│
│ │                                                           ││
│ │ Destination IP:   [127.0.0.1        ▼] (localhost)      ││
│ │                   [192.168.1.100     ] (TouchDesigner)   ││
│ │                   [192.168.1.101     ] (Max/MSP)         ││
│ │                   [Custom...         ]                    ││
│ │                                                           ││
│ │ Port:             [9000    ]  [Test Connection]          ││
│ │                                                           ││
│ │ Preset Targets:   [TouchDesigner ▼]                      ││
│ │                   • TouchDesigner (127.0.0.1:9000)       ││
│ │                   • Max/MSP (127.0.0.1:7400)             ││
│ │                   • Processing (127.0.0.1:12000)         ││
│ │                   • Resolume (127.0.0.1:7000)            ││
│ │                   • Custom...                             ││
│ │                                                           ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Data Streaming ─────────────────────────────────────────┐│
│ │                                                           ││
│ │ What to stream:                                           ││
│ │   ☑ Camera pose (real-time)                              ││
│ │       Address: /slam/camera/pose                         ││
│ │       Format: [x y z qx qy qz qw]                        ││
│ │       Rate: [30 Hz ▼]                                    ││
│ │                                                           ││
│ │   ☑ Point cloud (chunked)                                ││
│ │       Address: /slam/pointcloud/chunk                    ││
│ │       Format: [x y z r g b] per point                    ││
│ │       Chunk size: [1000 ▼] points                        ││
│ │       Update rate: [5 Hz ▼]                              ││
│ │       Voxel downsample: [0.05] m                         ││
│ │                                                           ││
│ │   ☑ Keyframe events                                      ││
│ │       Address: /slam/keyframe/new                        ││
│ │       Format: [frame_id timestamp]                       ││
│ │                                                           ││
│ │   ☑ SLAM status                                          ││
│ │       Address: /slam/status                              ││
│ │       Data: [state, fps, points, confidence]             ││
│ │       Rate: [2 Hz ▼]                                     ││
│ │                                                           ││
│ │   ☐ Tracking quality metrics                             ││
│ │       Address: /slam/tracking/quality                    ││
│ │       Data: [inliers, matches, error]                    ││
│ │                                                           ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ TouchDesigner Integration ──────────────────────────────┐│
│ │                                                           ││
│ │ ☑ Send as TouchDesigner Table DAT format                 ││
│ │ ☑ Include column headers                                 ││
│ │ ☐ Send texture data (RGB images)                         ││
│ │   • Via UDP (faster, lossy)                              ││
│ │   • Via TCP (reliable)                                    ││
│ │   • Via Shared Memory (lowest latency)                   ││
│ │                                                           ││
│ │ [Export TouchDesigner Template.toe]                      ││
│ │                                                           ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Connection Status ──────────────────────────────────────┐│
│ │ OSC Client: ● Connected to 127.0.0.1:9000                ││
│ │ Messages sent: 1,247                                      ││
│ │ Data rate: 2.3 MB/s                                       ││
│ │ Latency: 1.2 ms                                           ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Advanced ───────────────────────────────────────────────┐│
│ │ Buffer size: [8192] bytes                                 ││
│ │ ☑ Use bundle messages (more efficient)                   ││
│ │ ☑ Auto-reconnect on connection loss                      ││
│ │ Timestamp mode: [SLAM time ▼] [System time ▼]           ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**OSC Message Formats:**

```python
# Camera Pose
/slam/camera/pose f f f f f f f
  [tx, ty, tz, qx, qy, qz, qw]

# Point Cloud Chunk (1000 points)
/slam/pointcloud/chunk i [f f f i i i] * 1000
  chunk_id, [x, y, z, r, g, b] * n_points

# Keyframe Event
/slam/keyframe/new i f i
  [keyframe_id, timestamp, point_count]

# Status Update
/slam/status s f i f
  [state, fps, total_points, avg_confidence]
  state: "initializing" | "tracking" | "lost" | "complete"

# Tracking Quality
/slam/tracking/quality i i f
  [num_inliers, num_matches, reprojection_error]

# Complete Scan Signal
/slam/complete s
  [output_filepath]
```

**New Features:**
- ✨ Easy IP and port selection
- ✨ Preset targets for common software
- ✨ Connection testing
- ✨ Configurable data streaming (pose, points, status)
- ✨ Adjustable update rates
- ✨ Point cloud chunking for network efficiency
- ✨ TouchDesigner-specific formats
- ✨ Real-time connection monitoring
- ✨ Export TouchDesigner template project

---

## Tab 4: Config (Enhanced)

```
┌─────────────────────────────────────────────────────────────┐
│ SLAM CONFIGURATION                                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ ┌─ Configuration Preset ───────────────────────────────────┐│
│ │ [base.yaml            ▼] [Edit] [Save As] [Delete]      ││
│ │   • base.yaml (Default settings)                         ││
│ │   • eth3d.yaml (ETH3D optimization)                      ││
│ │   • eval_calib.yaml (With calibration)                   ││
│ │   • high_quality.yaml (Slow but accurate)                ││
│ │   • realtime.yaml (Fast tracking)                        ││
│ │   • custom_rtx5090.yaml ★                                ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Performance ────────────────────────────────────────────┐│
│ │ GPU: [RTX 5090 ▼]  VRAM: [32 GB] [Monitor]              ││
│ │                                                           ││
│ │ Image Resolution:                                         ││
│ │   Downsample: [None ▼]  (1x, 2x, 4x)                    ││
│ │   Target: 1280x960 (Original)                            ││
│ │                                                           ││
│ │ Batch Size: [4 ▼]  (Recommended: 4 for RTX 5090)        ││
│ │                                                           ││
│ │ Max Iterations:                                           ││
│ │   Tracking: [50 ▼]                                       ││
│ │   Local Opt: [10 ▼]                                      ││
│ │                                                           ││
│ │ ☑ Use FP16 precision (faster)                            ││
│ │ ☑ Enable cuDNN benchmark mode                            ││
│ │                                                           ││
│ │ Expected FPS: ~12-15 @ 1280x960                          ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Reconstruction Quality ─────────────────────────────────┐│
│ │ Keyframe Selection:                                       ││
│ │   Strategy: [Adaptive ▼]  (Adaptive, Fixed, Dense)      ││
│ │   Spacing: [──●──] (meters or frames)                    ││
│ │                                                           ││
│ │ Confidence Filtering:                                     ││
│ │   Threshold: [──●────] 0.5                               ││
│ │   Keep low-conf: [10%] (for completeness)                ││
│ │                                                           ││
│ │ Loop Closure:                                             ││
│ │   ☑ Enable loop detection                                ││
│ │   Min separation: [20 ▼] frames                          ││
│ │   Candidates: [3 ▼]                                      ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Visualization ──────────────────────────────────────────┐│
│ │ ☑ Enable 3D visualization window                         ││
│ │   Update rate: [30 Hz ▼]                                 ││
│ │   Point render: [Surfels ▼] (Points, Surfels, Mesh)     ││
│ │                                                           ││
│ │ ☑ Show camera frustums                                   ││
│ │ ☑ Show trajectory path                                   ││
│ │ ☐ Show depth uncertainty                                 ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Advanced ───────────────────────────────────────────────┐│
│ │ [Show Advanced Settings ▼]                                ││
│ │   Camera Intrinsics:                                      ││
│ │     ◉ Auto (from dataset/camera)                         ││
│ │     ○ Manual: fx[___] fy[___] cx[___] cy[___]           ││
│ │                                                           ││
│ │   Measurement Uncertainty:                                ││
│ │     Sigma ray: [0.1]  Sigma dist: [0.01]                ││
│ │                                                           ││
│ │   Memory Management:                                      ││
│ │     Keyframe window: [1000000]  (unlimited)              ││
│ │     ☑ Clear GPU cache between frames                     ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ [Reset to Defaults]  [Import Config]  [Export Config]       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**New Features:**
- ✨ Configuration presets with descriptions
- ✨ Visual performance tuning
- ✨ GPU monitoring integration
- ✨ Expected FPS estimates
- ✨ Quality vs. speed sliders
- ✨ Advanced settings collapsible section
- ✨ Import/Export configs

---

## Tab 5: Monitor (NEW - Real-time Status)

```
┌─────────────────────────────────────────────────────────────┐
│ SLAM MONITORING                                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ ┌─ Status ─────────────────────────────────────────────────┐│
│ │ State: ● TRACKING  [Pause] [Resume] [Stop]               ││
│ │ Progress: [████████████──────] 65% (520/798 frames)      ││
│ │ Elapsed: 00:03:42  Remaining: ~00:02:10                   ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Performance Metrics ────────────────────────────────────┐│
│ │  FPS: 14.2 ████████████████░░  [Live Graph ▼]           ││
│ │  GPU: 78%  ██████████████████░                           ││
│ │  VRAM: 11.2 / 32 GB  ███████░░░░░░░░░░░                  ││
│ │  CPU: 23%  ██████░░░░░░░░░░░░░░░                         ││
│ │  RAM: 5.8 / 64 GB                                         ││
│ │                                                           ││
│ │  Avg frame time: 70 ms                                    ││
│ │    Tracking: 35 ms                                        ││
│ │    Optimization: 25 ms                                    ││
│ │    Rendering: 10 ms                                       ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ SLAM Statistics ────────────────────────────────────────┐│
│ │  Keyframes: 124                                           ││
│ │  Points: 1,247,852  (↑ 3,421 this frame)                ││
│ │  Avg confidence: 0.73                                     ││
│ │                                                           ││
│ │  Tracking quality: ████████████████░░ 82%                ││
│ │    Feature matches: 847 / 1000                           ││
│ │    Inliers: 812 (96%)                                    ││
│ │    Reprojection error: 0.34 px                           ││
│ │                                                           ││
│ │  Loop closures: 3 detected                                ││
│ │  Lost tracking events: 0                                  ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Output Preview ─────────────────────────────────────────┐│
│ │  [3D View Preview]                                        ││
│ │  ┌─────────────────────────────────────────────────────┐ ││
│ │  │                                                       │ ││
│ │  │         [Real-time point cloud rendering]            │ ││
│ │  │                                                       │ ││
│ │  │     (Click to open full visualization window)        │ ││
│ │  │                                                       │ ││
│ │  └─────────────────────────────────────────────────────┘ ││
│ │                                                           ││
│ │  Current view: Top-down      [Rotate] [Reset Camera]     ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Logs ───────────────────────────────────────────────────┐│
│ │  [All ▼] [Errors ▼] [Warnings ▼] [Info ▼]  [Clear]     ││
│ │  ┌───────────────────────────────────────────────────┐   ││
│ │  │ [09:15:30] Initialized SLAM system                │   ││
│ │  │ [09:15:32] Loaded 798 frames                      │   ││
│ │  │ [09:15:35] ⚠ Low GPU memory warning               │   ││
│ │  │ [09:15:40] Keyframe 50 added                      │   ││
│ │  │ [09:15:45] Loop closure detected at frame 245     │   ││
│ │  │ [09:15:48] Optimizing global map...               │   ││
│ │  │ [09:15:50] Keyframe 100 added                     │   ││
│ │  │ [Auto-scroll ☑]                                    │   ││
│ │  └───────────────────────────────────────────────────┘   ││
│ │                                                           ││
│ │  [Export Log] [Save Screenshot]                          ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**New Features:**
- ✨ Real-time progress tracking
- ✨ Live performance graphs (FPS, GPU, VRAM, CPU)
- ✨ Detailed timing breakdown
- ✨ SLAM statistics (keyframes, points, quality)
- ✨ Feature matching visualization
- ✨ Embedded 3D preview
- ✨ Filterable log viewer
- ✨ Pause/Resume/Stop controls
- ✨ Export logs and screenshots

---

## Additional Advanced Features

### 1. Batch Processing Mode

```
┌─ Batch Processing ──────────────────────────────────────────┐
│ ☑ Enable batch mode                                          │
│                                                               │
│ Datasets Queue:  [Add] [Remove] [Clear] [Import List]       │
│   1. ☑ freiburg1_xyz          → output_001/                 │
│   2. ☑ freiburg2_desk         → output_002/                 │
│   3. ☐ custom_recording_001   → output_003/  (PENDING)      │
│   4. ☑ euroc_MH01             → output_004/                 │
│                                                               │
│ Process: [Sequential ▼] [Parallel (2 GPUs) ▼]               │
│ On complete: [Shutdown PC ▼] [Send email ▼] [None ▼]       │
│                                                               │
│ Total estimated time: 2h 45m                                  │
│ [Start Batch Processing]                                      │
└───────────────────────────────────────────────────────────────┘
```

### 2. Cloud/Remote Processing

```
┌─ Cloud Processing ──────────────────────────────────────────┐
│ Target: ◉ Local  ○ Remote Server  ○ AWS  ○ Google Cloud    │
│                                                               │
│ Remote Server:                                                │
│   Address: [192.168.1.50:8080]  [Test Connection]           │
│   API Key: [••••••••••••••••]                                │
│                                                               │
│ ☑ Upload dataset automatically                               │
│ ☑ Download results when complete                            │
│ ☑ Delete remote files after download                        │
│                                                               │
│ Upload progress: [████████░░] 87% (2.3 / 2.6 GB)            │
└───────────────────────────────────────────────────────────────┘
```

### 3. Dataset Recording Tool

```
┌─ Record New Dataset ────────────────────────────────────────┐
│ Camera: [RealSense D455 ▼]                                   │
│ Save to: [C:\datasets\new_scan_001\] [Browse]               │
│                                                               │
│ Recording mode:                                               │
│   ◉ Continuous  ○ Triggered  ○ Time-lapse                   │
│                                                               │
│ Duration: [──●────] 60 seconds  [∞ Unlimited]               │
│                                                               │
│ Preview: [Live camera feed]                                   │
│ Status: Ready  Frames: 0  Size: 0 MB                         │
│                                                               │
│ [● REC] [⏸ Pause] [⏹ Stop]                                   │
│                                                               │
│ ☑ Auto-process after recording                               │
└───────────────────────────────────────────────────────────────┘
```

### 4. Quality Assurance

```
┌─ QA Checks ─────────────────────────────────────────────────┐
│ Post-processing validation:                                   │
│   ☑ Check point cloud density                                │
│   ☑ Validate trajectory smoothness                           │
│   ☑ Detect motion blur frames                                │
│   ☑ Check feature distribution                               │
│   ☑ Measure reconstruction completeness                      │
│                                                               │
│ Quality score: ████████░░ 85/100                             │
│                                                               │
│ Issues found:                                                 │
│   ⚠ Sparse coverage in region [2.5, 1.2, 0.8]               │
│   ⚠ Low feature count in frames 234-267                      │
│   ✓ No trajectory discontinuities                            │
│   ✓ Good loop closure distribution                           │
│                                                               │
│ [Generate QA Report] [Re-process Problem Areas]              │
└───────────────────────────────────────────────────────────────┘
```

### 5. Comparison Tool

```
┌─ Compare Runs ──────────────────────────────────────────────┐
│ Run A: [output_001 ▼]  vs  Run B: [output_002 ▼]           │
│                                                               │
│ Metric           Run A      Run B      Difference            │
│ ───────────────────────────────────────────────────────────  │
│ Points           124,532    156,721    +32,189 (+26%)       │
│ Keyframes        87         102        +15 (+17%)           │
│ Processing time  3m 42s     4m 15s     +33s                 │
│ Avg confidence   0.73       0.68       -0.05                │
│ ATE (if GT)      0.034 m    0.028 m    -0.006 m (better)   │
│                                                               │
│ [Export Comparison] [Overlay 3D View]                        │
└───────────────────────────────────────────────────────────────┘
```

### 6. Automated Workflows

```
┌─ Workflow Automation ───────────────────────────────────────┐
│ Template: [Custom Workflow ▼]                                │
│                                                               │
│ Steps:                                                        │
│   1. ☑ Process SLAM                                          │
│   2. ☑ Export full point cloud (PLY)                        │
│   3. ☑ Generate preview cloud (subsampled)                  │
│   4. ☑ Stream to TouchDesigner via OSC                      │
│   5. ☑ Export trajectory (TUM + JSON)                       │
│   6. ☑ Generate HTML report                                  │
│   7. ☑ Upload to cloud storage                               │
│   8. ☐ Send notification email                               │
│                                                               │
│ [Save Workflow] [Load Workflow] [Share Workflow]            │
└───────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Core Enhancements (Week 1)
1. ✅ Output tab with export control
2. ✅ OSC basic implementation
3. ✅ Enhanced input preview
4. ✅ Monitor tab with real-time stats

### Phase 2: TouchDesigner Integration (Week 2)
1. ✅ OSC point cloud streaming
2. ✅ Camera pose streaming
3. ✅ TouchDesigner template export
4. ✅ Connection monitoring

### Phase 3: Advanced Features (Week 3)
1. ✅ Batch processing
2. ✅ Dataset recording tool
3. ✅ Quality assurance
4. ✅ Comparison tool

### Phase 4: Automation & Polish (Week 4)
1. ✅ Workflow templates
2. ✅ Cloud integration
3. ✅ Performance optimization
4. ✅ User testing and refinement

---

## Technical Implementation Notes

### OSC Library Installation
```bash
pip install python-osc
```

### OSC Server Implementation (Example)
```python
from pythonosc import udp_client
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder
import numpy as np

class OSCStreamer:
    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.enabled = True

    def send_camera_pose(self, T_WC):
        """Send 7-DOF camera pose"""
        # Extract position and quaternion from SE(3)
        t = T_WC[:3, 3]
        q = rotation_to_quaternion(T_WC[:3, :3])

        self.client.send_message(
            "/slam/camera/pose",
            [float(t[0]), float(t[1]), float(t[2]),
             float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        )

    def send_pointcloud_chunk(self, points, colors, chunk_id):
        """Send point cloud chunk"""
        bundle = osc_bundle_builder.OscBundleBuilder(
            osc_bundle_builder.IMMEDIATELY
        )

        msg = osc_message_builder.OscMessageBuilder(
            address="/slam/pointcloud/chunk"
        )
        msg.add_arg(chunk_id)

        for i in range(len(points)):
            msg.add_arg(float(points[i, 0]))
            msg.add_arg(float(points[i, 1]))
            msg.add_arg(float(points[i, 2]))
            msg.add_arg(int(colors[i, 0]))
            msg.add_arg(int(colors[i, 1]))
            msg.add_arg(int(colors[i, 2]))

        bundle.add_content(msg.build())
        self.client.send(bundle.build())

    def send_status(self, state, fps, points, confidence):
        """Send SLAM status"""
        self.client.send_message(
            "/slam/status",
            [state, float(fps), int(points), float(confidence)]
        )
```

### TouchDesigner Template Structure
```
SLAM_Receiver.toe
├── OSC In CHOP (/slam/*)
├── Table DAT (camera poses)
├── SOP (point cloud geometry)
├── Material (point rendering)
└── Render TOP (output)
```

---

## Summary: Enhanced Features

**Export Control:**
- ✨ Custom output paths
- ✨ Multiple format support (PLY, OBJ, TUM, KITTI, JSON, Alembic)
- ✨ Confidence threshold control
- ✨ Subsampled preview generation
- ✨ HTML report generation

**OSC/Network Streaming:**
- ✨ Real-time camera pose streaming
- ✨ Point cloud chunked streaming
- ✨ Status and metrics streaming
- ✨ TouchDesigner preset integration
- ✨ Connection monitoring

**Enhanced Monitoring:**
- ✨ Real-time performance graphs
- ✨ SLAM statistics dashboard
- ✨ Embedded 3D preview
- ✨ Filterable log viewer
- ✨ Quality metrics

**Workflow Tools:**
- ✨ Batch processing
- ✨ Dataset recording
- ✨ Quality assurance
- ✨ Run comparison
- ✨ Automated workflows

---

*Document Version: 1.0*
*Last Updated: December 2, 2025*
*Status: Design Complete - Ready for Review*
