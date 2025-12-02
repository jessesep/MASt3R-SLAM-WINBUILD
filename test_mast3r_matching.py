"""Test MASt3R asymmetric/symmetric inference"""
import torch
import numpy as np
from mast3r_slam.config import load_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, mast3r_match_asymmetric, mast3r_match_symmetric
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
print("\nInitializing frame 0 with mono inference...")
X0, C0 = mast3r_inference_mono(model, frame0)
frame0.update_pointmap(X0, C0)
print(f"Frame 0: X min={X0.min():.3f}, max={X0.max():.3f}, non-zero={((X0 != 0).sum())}/{X0.numel()}")

print("\nInitializing frame 1 with mono inference...")
X1, C1 = mast3r_inference_mono(model, frame1)
frame1.update_pointmap(X1, C1)
print(f"Frame 1: X min={X1.min():.3f}, max={X1.max():.3f}, non-zero={((X1 != 0).sum())}/{X1.numel()}")

print("\n" + "="*60)
print("Testing ASYMMETRIC matching (frame 0 -> frame 1)...")
print("="*60)
idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji = mast3r_match_asymmetric(model, frame0, frame1)

print(f"\nResults:")
print(f"  Xii (frame 0 points) shape: {Xii.shape}")
print(f"    min: {Xii.min():.3f}, max: {Xii.max():.3f}, non-zero: {((Xii != 0).sum())}/{Xii.numel()}")

print(f"  Xji (frame 1 points) shape: {Xji.shape}")
print(f"    min: {Xji.min():.3f}, max: {Xji.max():.3f}, non-zero: {((Xji != 0).sum())}/{Xji.numel()}")

print(f"  valid_match_j: {valid_match_j.sum()}/{valid_match_j.numel()} valid matches")

if Xji.abs().max() == 0:
    print("\n[ERROR] Xji is ALL ZEROS - this is the bug!")
else:
    print("\n[OK] Asymmetric matching produces non-zero points")
