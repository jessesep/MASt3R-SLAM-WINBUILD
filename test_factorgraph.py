"""Test FactorGraph to isolate segfault cause"""
import torch
import lietorch
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.frame_singlethread import SingleThreadKeyframes
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph

torch.set_grad_enabled(False)

print("="*60)
print("Testing FactorGraph - Isolating Segfault")
print("="*60)

device = "cuda:0"

# Load config and dataset
print("\n1. Loading config and dataset...")
load_config("config/base.yaml")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
h, w = dataset.get_img_shape()[0]
print(f"   Loaded: {len(dataset)} frames")

# Load model
print("\n2. Loading MASt3R model...")
model = load_mast3r(device=device)
print("   Model loaded")

# Create keyframes
print("\n3. Creating keyframes structure...")
keyframes = SingleThreadKeyframes(h, w, device=device)
print("   Keyframes created")

# Initialize first frame
print("\n4. Initializing Frame 0...")
timestamp0, img0 = dataset[0]
T_WC0 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
X0, C0 = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X0, C0)
keyframes.append(frame0)
print(f"   Frame 0 added: {len(keyframes)} keyframes")

# THIS IS WHERE IT LIKELY CRASHES
print("\n5. Creating FactorGraph...")
try:
    K = None  # No calibration
    factor_graph = FactorGraph(model, keyframes, K, device)
    print("   [OK] FactorGraph created successfully")
except Exception as e:
    print(f"   [FAIL] FactorGraph creation FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Try to run optimization
print("\n6. Running first optimization (Frame 0)...")
try:
    if config["use_calib"]:
        factor_graph.solve_GN_calib()
    else:
        factor_graph.solve_GN_rays()
    print("   [OK] Optimization completed successfully")
except Exception as e:
    print(f"   [FAIL] Optimization FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n7. Adding second frame...")
timestamp1, img1 = dataset[1]
T_WC1 = frame0.T_WC
frame1 = create_frame(1, img1, T_WC1, img_size=dataset.img_size, device=device)
X1, C1 = mast3r_inference_mono(model, frame1)
frame1.update_pointmap(X1, C1)
keyframes.append(frame1)
print(f"   Frame 1 added: {len(keyframes)} keyframes")

print("\n8. Adding factors between Frame 0 and Frame 1...")
try:
    frame_idx = [1]
    kf_idx = [0]
    min_match_frac = 0.05
    result = factor_graph.add_factors(frame_idx, kf_idx, min_match_frac, is_reloc=False)
    print(f"   [OK] Factors added: {result}")
except Exception as e:
    print(f"   [FAIL] add_factors FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n9. Running optimization with 2 frames...")
try:
    if config["use_calib"]:
        factor_graph.solve_GN_calib()
    else:
        factor_graph.solve_GN_rays()
    print("   [OK] Optimization completed successfully")
except Exception as e:
    print(f"   [FAIL] Optimization FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("SUCCESS! FactorGraph works without segfault")
print("="*60)
print("\nIf you see this, the issue is elsewhere!")
