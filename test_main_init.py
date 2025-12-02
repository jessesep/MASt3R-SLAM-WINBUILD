"""Test main.py initialization to find segfault cause"""
import sys
print("1. Starting imports...")

try:
    import torch
    print("2. Imported torch")

    import torch.multiprocessing as mp
    print("3. Imported multiprocessing")

    # Try set_start_method
    try:
        mp.set_start_method("spawn", force=True)
        print("4. Set start method to spawn")
    except RuntimeError as e:
        print(f"4. Start method already set: {e}")

    from mast3r_slam.config import load_config, config
    print("5. Imported config")

    load_config("config/base.yaml")
    print("6. Loaded config")
    print(f"   single_thread: {config['single_thread']}")

    from mast3r_slam.dataloader import load_dataset
    print("7. Imported dataloader")

    print("\n8. Creating Manager...")
    manager = mp.Manager()
    print("9. Manager created successfully!")

    print("\n10. Loading dataset...")
    dataset = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
    print(f"11. Dataset loaded: {len(dataset)} frames")

    h, w = dataset.get_img_shape()[0]
    print(f"12. Image shape: {h}x{w}")

    from mast3r_slam.frame import SharedKeyframes, SharedStates
    print("13. Imported frame classes")

    print("\n14. Creating SharedKeyframes...")
    keyframes = SharedKeyframes(manager, h, w)
    print("15. SharedKeyframes created!")

    print("\n16. Creating SharedStates...")
    states = SharedStates(manager, h, w)
    print("17. SharedStates created!")

    print("\n18. Loading MASt3R model...")
    from mast3r_slam.mast3r_utils import load_mast3r
    device = "cuda:0"
    model = load_mast3r(device=device)
    print("19. Model loaded!")

    print("\n[SUCCESS] All initialization steps completed!")

except Exception as e:
    print(f"\n[ERROR] Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
