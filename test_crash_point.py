"""Find exact crash point - step by step"""
import sys
print("Step 1: Imports...", flush=True)

import torch
print("  torch imported", flush=True)

import lietorch
print("  lietorch imported", flush=True)

from mast3r_slam.config import load_config, config
print("  config imported", flush=True)

from mast3r_slam.dataloader import load_dataset
print("  dataloader imported", flush=True)

from mast3r_slam.frame import create_frame
print("  frame imported", flush=True)

from mast3r_slam.frame_singlethread import SingleThreadKeyframes
print("  SingleThreadKeyframes imported", flush=True)

from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
print("  mast3r_utils imported", flush=True)

from mast3r_slam.tracker import FrameTracker
print("  tracker imported", flush=True)

from mast3r_slam.global_opt import FactorGraph
print("  FactorGraph imported", flush=True)

print("\nStep 2: Basic setup...", flush=True)
torch.set_grad_enabled(False)
device = "cuda:0"
print(f"  device: {device}", flush=True)

print("\nStep 3: Load config...", flush=True)
load_config("config/base.yaml")
print("  config loaded", flush=True)

print("\nStep 4: Load dataset...", flush=True)
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
h, w = dataset.get_img_shape()[0]
print(f"  dataset loaded: {h}x{w}", flush=True)

print("\nStep 5: Load model...", flush=True)
model = load_mast3r(device=device)
print("  model loaded", flush=True)

print("\nStep 6: Create SingleThreadKeyframes...", flush=True)
keyframes = SingleThreadKeyframes(h, w, device=device)
print(f"  keyframes created", flush=True)

print("\nStep 7: Create FactorGraph...", flush=True)
sys.stdout.flush()
factor_graph = FactorGraph(model, keyframes, None, device)
print("  factor_graph created", flush=True)

print("\nStep 8: Create FrameTracker...", flush=True)
tracker = FrameTracker(model, keyframes, device)
print("  tracker created", flush=True)

print("\nStep 9: Load Frame 0...", flush=True)
timestamp0, img0 = dataset[0]
T_WC0 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
print("  frame0 created", flush=True)

print("\nStep 10: MASt3R mono inference...", flush=True)
X0, C0 = mast3r_inference_mono(model, frame0)
print(f"  inference done: X0 shape={X0.shape}", flush=True)

print("\nStep 11: Update pointmap...", flush=True)
frame0.update_pointmap(X0, C0)
print("  pointmap updated", flush=True)

print("\nStep 12: Append to keyframes...", flush=True)
keyframes.append(frame0)
print(f"  appended: {len(keyframes)} keyframes", flush=True)

print("\nStep 13: Load Frame 1...", flush=True)
timestamp1, img1 = dataset[1]
frame1 = create_frame(1, img1, frame0.T_WC, img_size=dataset.img_size, device=device)
print("  frame1 created", flush=True)

print("\nStep 14: Track frame 1...", flush=True)
sys.stdout.flush()
add_new_kf, match_info, try_reloc = tracker.track(frame1)
print(f"  tracking done: match_frac={match_info['match_frac']:.4f}", flush=True)

if add_new_kf:
    print("\nStep 15: Append frame1 to keyframes...", flush=True)
    keyframes.append(frame1)
    print(f"  appended: {len(keyframes)} keyframes", flush=True)

    print("\nStep 16: Add factors...", flush=True)
    sys.stdout.flush()
    frame_idx = [1]
    kf_idx = [0]
    result = factor_graph.add_factors(frame_idx, kf_idx, 0.05, is_reloc=False)
    print(f"  add_factors returned: {result}", flush=True)

    if result:
        print("\nStep 17: Optimize (solve_GN_rays)...", flush=True)
        sys.stdout.flush()
        factor_graph.solve_GN_rays()
        print("  optimization done", flush=True)

print("\n" + "="*60, flush=True)
print("SUCCESS! All steps completed without crash.", flush=True)
print("="*60, flush=True)
