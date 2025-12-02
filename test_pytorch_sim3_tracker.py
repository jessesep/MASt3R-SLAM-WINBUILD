"""Test the tracker with PyTorch Sim3 implementation"""
import sys
import torch
from mast3r_slam.config import load_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.frame import SharedKeyframes
import mast3r_slam.lietorch_compat as lietorch

print("="*80, flush=True)
print("TESTING TRACKER WITH PYTORCH SIM3", flush=True)
print("="*80, flush=True)

# Load config
print("\n[1] Loading config...", flush=True)
load_config("config/base.yaml")

# Load model
print("[2] Loading MASt3R model...", flush=True)
device = "cuda:0"
model = load_mast3r(device=device)

# Load dataset
print("[3] Loading dataset...", flush=True)
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
timestamp0, img0 = dataset[0]
timestamp1, img1 = dataset[1]

# Create frames
print("[4] Creating frames...", flush=True)
T_WC0 = lietorch.Sim3.Identity(1, device=device)
T_WC1 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
frame1 = create_frame(1, img1, T_WC1, img_size=dataset.img_size, device=device)

# Initialize with mono inference
print("[5] Running mono inference on frames...", flush=True)
X0, C0 = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X0, C0)
X1, C1 = mast3r_inference_mono(model, frame1)
frame1.update_pointmap(X1, C1)

# Create keyframes container
print("[6] Creating keyframes...", flush=True)
keyframes = SharedKeyframes(device=device)
keyframes.append(frame0)

# Create tracker
print("[7] Creating tracker...", flush=True)
tracker = FrameTracker(model, keyframes, device)

# Test tracking
print("\n[8] Testing tracker.track()...", flush=True)
print("="*80, flush=True)
try:
    new_kf, data, skip = tracker.track(frame1)
    print("="*80, flush=True)
    print(f"[SUCCESS] Tracking completed!", flush=True)
    print(f"  new_kf: {new_kf}", flush=True)
    print(f"  skip: {skip}", flush=True)
    print(f"  frame1 pose: {frame1.T_WC.data}", flush=True)
    print("\n" + "="*80, flush=True)
    print("[OK] ALL TESTS PASSED! TRACKER WORKS WITH PYTORCH SIM3!", flush=True)
    print("="*80, flush=True)
    sys.exit(0)
except Exception as e:
    print("="*80, flush=True)
    print(f"[FAILED] Tracking failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
