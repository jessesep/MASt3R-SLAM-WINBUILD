"""Test MASt3R model output directly"""
import torch
import numpy as np
from mast3r_slam.config import load_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
import lietorch

print("Loading config and dataset...")
load_config("config/base.yaml")
dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")

print(f"Dataset loaded: {len(dataset)} frames\n")

print("Loading MASt3R model...")
device = "cuda:0"
model = load_mast3r(device=device)
print("Model loaded\n")

print("Getting first frame...")
timestamp, img = dataset[0]
T_WC = lietorch.Sim3.Identity(1, device=device)
frame = create_frame(0, img, T_WC, img_size=dataset.img_size, device=device)
print(f"Frame created: {frame.img.shape}\n")

print("Running MASt3R mono inference...")
X, C = mast3r_inference_mono(model, frame)
print(f"Inference complete!")
print(f"  X (3D points) shape: {X.shape}")
print(f"  X min: {X.min():.6f}, max: {X.max():.6f}, mean: {X.mean():.6f}")
print(f"  X has NaN: {torch.isnan(X).any()}")
print(f"  X has Inf: {torch.isinf(X).any()}")
print(f"  Non-zero elements: {(X != 0).sum()} / {X.numel()}")
print(f"\n  C (confidence) shape: {C.shape}")
print(f"  C min: {C.min():.6f}, max: {C.max():.6f}, mean: {C.mean():.6f}")

if X.abs().max() == 0:
    print(f"\n✗ ERROR: MASt3R returned ALL ZERO 3D points!")
    print(f"  This is the Windows-specific bug.")
else:
    print(f"\n✓ MASt3R mono inference working - points are non-zero")
    print(f"  Sample points (first 5):")
    print(f"  {X[0, :5, :]}")
