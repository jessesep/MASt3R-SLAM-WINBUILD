"""Test MASt3R symmetric matching (used in relocalization)"""
import torch
from mast3r_slam.config import load_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, mast3r_match_symmetric
import lietorch

print("Loading config and dataset...")
load_config("config/base.yaml")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")

print("Loading MASt3R model...")
device = "cuda:0"
model = load_mast3r(device=device)

print("Getting frames 0 and 1...")
timestamp0, img0 = dataset[0]
timestamp1, img1 = dataset[1]
T_WC0 = lietorch.Sim3.Identity(1, device=device)
T_WC1 = lietorch.Sim3.Identity(1, device=device)
frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
frame1 = create_frame(1, img1, T_WC1, img_size=dataset.img_size, device=device)

# Initialize frames with mono inference
print("\nInitializing frames...")
X0, C0 = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X0, C0)
X1, C1 = mast3r_inference_mono(model, frame1)
frame1.update_pointmap(X1, C1)

print("\n" + "="*60)
print("Testing SYMMETRIC matching (frame 0 <-> frame 1)...")
print("="*60)

# Test with batch
frame_idx = [0]
kf_idx = [1]

# Need to extract features first (like SLAM does)
frame0.feat, frame0.pos, _ = model._encode_image(frame0.img, frame0.img_true_shape)
frame1.feat, frame1.pos, _ = model._encode_image(frame1.img, frame1.img_true_shape)

frames = [frame0]
keyframes_batch = [frame1]

print("\nCalling mast3r_match_symmetric...")
try:
    # Call with correct signature - returns 8 values
    # Need to pass shapes as lists (like add_factors does)
    idx_i2j, idx_j2i, valid_match_j, valid_match_i, Qii, Qjj, Qji, Qij = mast3r_match_symmetric(
        model,
        frame0.feat, frame0.pos,  # frame i features
        frame1.feat, frame1.pos,  # frame j features
        [frame0.img_true_shape], [frame1.img_true_shape]  # shapes as lists
    )

    print(f"\nResults:")
    print(f"  idx_i2j shape: {idx_i2j.shape}")
    print(f"  idx_j2i shape: {idx_j2i.shape}")
    print(f"  valid_match_j: {valid_match_j.sum()}/{valid_match_j.numel()}")
    print(f"  valid_match_i: {valid_match_i.sum()}/{valid_match_i.numel()}")

    print(f"\n  Qii shape: {Qii.shape}")
    print(f"    min: {Qii.min():.3f}, max: {Qii.max():.3f}")

    print(f"  Qji shape: {Qji.shape}")
    print(f"    min: {Qji.min():.3f}, max: {Qji.max():.3f}")

    if valid_match_j.sum() == 0 and valid_match_i.sum() == 0:
        print("\n[ERROR] Symmetric matching found ZERO matches!")
        print("This is the bug that causes relocalization to fail!")
    else:
        print("\n[OK] Symmetric matching produces matches")

except Exception as e:
    print(f"\n[ERROR] Symmetric matching failed: {e}")
    import traceback
    traceback.print_exc()
