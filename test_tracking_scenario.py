"""Test the exact tracking scenario that fails in main.py"""
import torch
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame, SharedKeyframes
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker
import lietorch
import torch.multiprocessing as mp

print("Loading config and dataset...")
load_config("config/base.yaml")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")

print("Loading MASt3R model...")
device = "cuda:0"
model = load_mast3r(device=device)

# Create shared keyframes (like main.py does)
h, w = dataset.get_img_shape()[0]
manager = mp.Manager()
keyframes = SharedKeyframes(manager, h, w)

print("\nInitializing with frame 0...")
timestamp0, img0 = dataset[0]
T_WC0 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)

# Initialize like main.py does
X_init, C_init = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X_init, C_init)
keyframes.append(frame0)
print(f"Frame 0 added as keyframe")
print(f"  X: min={X_init.min():.3f}, max={X_init.max():.3f}")

print("\nCreating tracker...")
tracker = FrameTracker(model, keyframes, device)

print("\nTracking frame 1...")
timestamp1, img1 = dataset[1]
T_WC1 = keyframes.last_keyframe().T_WC  # Use last keyframe pose
frame1 = create_frame(1, img1, T_WC1, img_size=dataset.img_size, device=device)

print(f"Frame 1 created, calling tracker.track()...")
add_new_kf, match_info, try_reloc = tracker.track(frame1)

print(f"\nTracking result:")
print(f"  add_new_kf: {add_new_kf}")
print(f"  try_reloc: {try_reloc}")
print(f"  match_info: {match_info}")

if try_reloc:
    print("\n[ERROR] Tracking failed, requested relocalization")
    print("This is where the SLAM gets stuck in the relocalization loop")
else:
    print("\n[OK] Tracking succeeded!")
