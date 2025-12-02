"""Test tracker WITHOUT FactorGraph"""
import sys
import torch
import lietorch
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.frame_singlethread import SingleThreadKeyframes
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker

torch.set_grad_enabled(False)
device = "cuda:0"

print("Testing tracker WITHOUT FactorGraph", flush=True)
print("="*60, flush=True)

load_config("config/base.yaml")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
h, w = dataset.get_img_shape()[0]

print("Loading model...", flush=True)
model = load_mast3r(device=device)

print("Creating keyframes...", flush=True)
keyframes = SingleThreadKeyframes(h, w, device=device)

# NO FactorGraph created!

print("Creating tracker...", flush=True)
tracker = FrameTracker(model, keyframes, device)

print("Initializing Frame 0...", flush=True)
timestamp0, img0 = dataset[0]
T_WC0 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
X0, C0 = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X0, C0)
keyframes.append(frame0)
print(f"Frame 0 initialized: {len(keyframes)} keyframes", flush=True)

print("Loading Frame 1...", flush=True)
timestamp1, img1 = dataset[1]
frame1 = create_frame(1, img1, frame0.T_WC, img_size=dataset.img_size, device=device)

print("Tracking Frame 1 (WITHOUT FactorGraph)...", flush=True)
sys.stdout.flush()
add_new_kf, match_info, try_reloc = tracker.track(frame1)
print(f"SUCCESS! Tracking worked: match_frac={match_info['match_frac']:.4f}", flush=True)
