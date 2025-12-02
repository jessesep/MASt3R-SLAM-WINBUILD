"""Test that mimics exactly what main.py does"""
import torch.multiprocessing as mp
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.multiprocess_utils import new_queue

print("="*60)
print("Mimicking main.py flow exactly...")
print("="*60)

# Mimic main.py setup
dataset_path = "datasets/tum/rgbd_dataset_freiburg1_xyz"
config_path = "config/base.yaml"
no_viz = True

print("\n1. Loading config...")
load_config(config_path)
print(f"   Config loaded. subsample = {config['dataset']['subsample']}")

print("\n2. Setting up multiprocessing...")
manager = mp.Manager()
main2viz = new_queue(manager, no_viz)
viz2main = new_queue(manager, no_viz)
print("   Multiprocessing queues created")

print("\n3. Loading dataset...")
dataset = load_dataset(dataset_path)
print(f"   Dataset type: {type(dataset)}")
print(f"   Dataset length: {len(dataset)}")
print(f"   rgb_files length: {len(dataset.rgb_files)}")

print("\n4. Subsampling dataset...")
print(f"   Before subsample: {len(dataset.rgb_files)} files")
dataset.subsample(config["dataset"]["subsample"])
print(f"   After subsample: {len(dataset.rgb_files)} files")

print("\n5. Getting image shape...")
try:
    h, w = dataset.get_img_shape()[0]
    print(f"   ✓ Success! Shape: {h}x{w}")
    print("\n✓ All steps completed successfully!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

    # Debug info
    print(f"\n   DEBUG INFO:")
    print(f"   rgb_files length: {len(dataset.rgb_files)}")
    print(f"   timestamps length: {len(dataset.timestamps)}")
    if len(dataset.rgb_files) > 0:
        print(f"   First file: {dataset.rgb_files[0]}")
    else:
        print(f"   rgb_files is EMPTY!")
