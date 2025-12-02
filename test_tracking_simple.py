"""Test tracking without multiprocessing - simpler approach"""
import torch
import mast3r_slam.lietorch_compat as lietorch  # PyTorch-based Sim3
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, mast3r_match_asymmetric

torch.set_grad_enabled(False)

print("="*60)
print("Testing Tracking WITHOUT Shared Memory/Multiprocessing")
print("="*60)

device = "cuda:0"

# Load config and dataset
print("\n1. Loading config and dataset...")
load_config("config/base.yaml")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
print(f"   Dataset loaded: {len(dataset)} frames")

# Load model
print("\n2. Loading MASt3R model...")
model = load_mast3r(device=device)
print("   Model loaded successfully")

# Initialize first frame
print("\n3. Initializing Frame 0...")
timestamp0, img0 = dataset[0]
T_WC0 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
X0, C0 = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X0, C0)
print(f"   Frame 0: X min={X0.min():.3f}, max={X0.max():.3f}")

# Create second frame
print("\n4. Creating Frame 1...")
timestamp1, img1 = dataset[1]
T_WC1 = frame0.T_WC
frame1 = create_frame(1, img1, T_WC1, img_size=dataset.img_size, device=device)

# Test asymmetric matching directly
print("\n5. Testing mast3r_match_asymmetric (Frame 0 -> Frame 1)...")
idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji = mast3r_match_asymmetric(
    model, frame0, frame1
)

# Calculate match statistics
valid_match_count = valid_match_j.sum().item()
total_points = valid_match_j.numel()
match_frac = valid_match_count / total_points if total_points > 0 else 0.0

print("\n" + "="*60)
print("MATCHING RESULTS:")
print("="*60)
print(f"Total points: {total_points}")
print(f"Valid matches: {valid_match_count}")
print(f"Match fraction: {match_frac:.4f}")
print(f"\nXii stats: min={Xii.min():.3f}, max={Xii.max():.3f}")
print(f"Xji stats: min={Xji.min():.3f}, max={Xji.max():.3f}")
print(f"Qii max: {Qii.max():.3f}")
print(f"Qji max: {Qji.max():.3f}")

if match_frac > 0.05:
    print("\n✓ SUCCESS! Matching works without multiprocessing")
    print("  The core MASt3R functions are fine")
    print("  The issue is specifically with Windows multiprocessing/shared memory")
else:
    print("\n✗ FAILED - Even without multiprocessing, matching fails")
    print("  This would indicate a deeper issue")
