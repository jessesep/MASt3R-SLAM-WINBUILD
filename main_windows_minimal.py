"""Minimal working SLAM for Windows - Start simple and build up"""
import torch
import lietorch
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.frame_singlethread import SingleThreadKeyframes
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.global_opt import FactorGraph

torch.set_grad_enabled(False)
device = "cuda:0"

print("="*60)
print("Minimal Windows SLAM - Testing Tracker")
print("="*60)

load_config("config/base.yaml")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
h, w = dataset.get_img_shape()[0]

print(f"\nLoading model...")
model = load_mast3r(device=device)

print("Creating structures...")
keyframes = SingleThreadKeyframes(h, w, device=device)
factor_graph = FactorGraph(model, keyframes, None, device)
tracker = FrameTracker(model, keyframes, device)

print("\nProcessing frames...")
print("="*60)

# Frame 0 - Initialize
timestamp0, img0 = dataset[0]
T_WC0 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
X0, C0 = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X0, C0)
keyframes.append(frame0)
print(f"Frame 0: INIT - {len(keyframes)} keyframes")

# Frame 1 - Track
timestamp1, img1 = dataset[1]
frame1 = create_frame(1, img1, frame0.T_WC, img_size=dataset.img_size, device=device)

print(f"\nFrame 1: Tracking...")
try:
    add_new_kf, match_info, try_reloc = tracker.track(frame1)
    print(f"  match_frac={match_info['match_frac']:.4f}")
    print(f"  add_kf={add_new_kf}, reloc={try_reloc}")

    if add_new_kf:
        keyframes.append(frame1)
        # Add factors
        frame_idx = [1]
        kf_idx = [0]
        if factor_graph.add_factors(frame_idx, kf_idx, 0.05, is_reloc=False):
            print(f"  Factors added, optimizing...")
            factor_graph.solve_GN_rays()
            print(f"  Optimization done")
        print(f"  Keyframes: {len(keyframes)}")

    print("\n[SUCCESS] Frame 1 tracked successfully!")

except Exception as e:
    print(f"\n[FAIL] Frame 1 tracking failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Try a few more frames
for i in range(2, min(10, len(dataset))):
    timestamp, img = dataset[i]
    prev_frame = keyframes[len(keyframes)-1]
    frame = create_frame(i, img, prev_frame.T_WC, img_size=dataset.img_size, device=device)

    try:
        add_new_kf, match_info, try_reloc = tracker.track(frame)
        status = "ADD_KF" if add_new_kf else "SKIP"
        print(f"Frame {i}: {status} - match_frac={match_info['match_frac']:.4f}")

        if add_new_kf:
            keyframes.append(frame)
            # Simple optimization with previous keyframe
            frame_idx = [len(keyframes)-1]
            kf_idx = [len(keyframes)-2]
            if factor_graph.add_factors(frame_idx, kf_idx, 0.05, is_reloc=False):
                factor_graph.solve_GN_rays()
    except Exception as e:
        print(f"Frame {i}: [ERROR] {e}")
        break

print("\n" + "="*60)
print(f"Processed successfully! Total keyframes: {len(keyframes)}")
print("="*60)
