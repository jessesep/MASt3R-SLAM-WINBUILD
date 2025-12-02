"""Quick test to see if file_system sharing strategy fixes the SLAM tracking issue"""
import sys
import torch
import torch.multiprocessing as mp
import lietorch
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import create_frame, SharedKeyframes, SharedStates, Mode
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker

if __name__ == "__main__":
    # Apply Windows fix
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_grad_enabled(False)

    print("="*60)
    print("Testing SLAM with file_system sharing strategy fix")
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

    # Create shared keyframes and states
    print("\n3. Creating shared structures...")
    manager = mp.Manager()
    keyframes = SharedKeyframes(device=device)
    states = SharedStates(manager)
    print("   Shared structures created")

    # Initialize first frame
    print("\n4. Initializing first frame (Frame 0)...")
    timestamp0, img0 = dataset[0]
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img0, T_WC0, img_size=dataset.img_size, device=device)
    X0, C0 = mast3r_inference_mono(model, frame0)
    frame0.update_pointmap(X0, C0)
    keyframes.append(frame0)
    print(f"   Frame 0 initialized: X min={X0.min():.3f}, max={X0.max():.3f}")

    # Create tracker
    print("\n5. Creating tracker...")
    tracker = FrameTracker(model, keyframes, device)
    print("   Tracker created")

    # Track second frame
    print("\n6. Tracking second frame (Frame 1)...")
    timestamp1, img1 = dataset[1]
    T_WC1 = frame0.T_WC.clone()
    frame1 = create_frame(1, img1, T_WC1, img_size=dataset.img_size, device=device)

    print("   Running tracker.track()...")
    add_new_kf, match_info, try_reloc = tracker.track(frame1)

    # Check results
    print("\n" + "="*60)
    print("TRACKING RESULTS:")
    print("="*60)
    print(f"add_new_kf: {add_new_kf}")
    print(f"match_info: valid={match_info['valid_match_k']}, match_frac={match_info['match_frac']:.4f}")
    print(f"try_reloc: {try_reloc}")

    if match_info['match_frac'] > 0.0:
        print("\n✓ SUCCESS! Tracking working with file_system sharing strategy!")
        print(f"  Valid matches: {match_info['valid_match_k']}")
        print(f"  Match fraction: {match_info['match_frac']:.4f}")
        sys.exit(0)
    else:
        print("\n✗ FAILED - Still getting zero matches")
        print("  The file_system sharing strategy did not fix the issue")
        sys.exit(1)
