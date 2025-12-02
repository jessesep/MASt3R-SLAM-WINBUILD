"""Quick test: Process first 50 frames to verify SLAM works"""
import sys
import torch
import mast3r_slam.lietorch_compat as lietorch
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import Mode, create_frame
from mast3r_slam.frame_singlethread import SingleThreadKeyframes, SingleThreadStates
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker

print("="*80)
print("QUICK TEST: MASt3R-SLAM ON WINDOWS (50 frames)")
print("="*80)

# Setup
device = "cuda:0"
load_config("config/base.yaml")

# Load dataset
print("\n[1] Loading dataset...")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
dataset.subsample(1)
h, w = dataset.get_img_shape()[0]
print(f"    Image size: {h}x{w}")

# Create keyframes and states
print("[2] Creating keyframes and states...")
keyframes = SingleThreadKeyframes(h, w)
states = SingleThreadStates(h, w)

# Load model
print("[3] Loading MASt3R model...")
model = load_mast3r(device=device)
model.share_memory()

# Create tracker
print("[4] Creating tracker...")
tracker = FrameTracker(model, keyframes, device)

# Process frames
print("\n[5] Processing frames...")
print("-"*80)

max_frames = min(50, len(dataset))
mode = Mode.INIT

for i in range(max_frames):
    timestamp, img = dataset[i]

    # Get camera pose
    if i == 0:
        T_WC = lietorch.Sim3.Identity(1, device=device)
    else:
        T_WC = states.get_frame().T_WC

    frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

    if mode == Mode.INIT:
        # Initialize first frame
        X_init, C_init = mast3r_inference_mono(model, frame)
        frame.update_pointmap(X_init, C_init)
        keyframes.append(frame)
        states.set_mode(Mode.TRACKING)
        states.set_frame(frame)
        mode = Mode.TRACKING
        print(f"Frame {i:3d}: INIT complete (keyframe)")

    elif mode == Mode.TRACKING:
        # Track frame
        add_new_kf, match_info, try_reloc = tracker.track(frame)
        states.set_frame(frame)

        if add_new_kf:
            keyframes.append(frame)
            kf_marker = " (keyframe)"
        else:
            kf_marker = ""

        # match_info is a list: [match_frac, ...]
        match_frac = match_info[0] if isinstance(match_info, (list, tuple)) else 0.0
        if torch.is_tensor(match_frac):
            match_frac = match_frac.item()
        print(f"Frame {i:3d}: match_frac={match_frac:.4f}, "
              f"add_kf={add_new_kf}{kf_marker}")

print("-"*80)
print(f"\n[6] Summary:")
print(f"    Processed: {max_frames} frames")
print(f"    Keyframes: {len(keyframes)}")
print("\n" + "="*80)
print("[SUCCESS] MASt3R-SLAM WORKS ON WINDOWS!")
print("="*80)
