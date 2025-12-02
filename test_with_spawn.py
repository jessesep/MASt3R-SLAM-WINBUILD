"""Test with spawn multiprocessing like main.py"""
import torch.multiprocessing as mp

# THIS is what main.py does first!
mp.set_start_method("spawn")

print("Set multiprocessing to 'spawn' mode")
print("Now testing dataset loading...\n")

from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.multiprocess_utils import new_queue

dataset_path = "datasets/tum/rgbd_dataset_freiburg1_xyz"
config_path = "config/base.yaml"

print("1. Loading config...")
load_config(config_path)

print("2. Creating multiprocessing manager...")
manager = mp.Manager()
main2viz = new_queue(manager, True)
viz2main = new_queue(manager, True)

print("3. Loading dataset...")
dataset = load_dataset(dataset_path)
print(f"   Dataset length: {len(dataset)}")
print(f"   rgb_files length: {len(dataset.rgb_files)}")

print("4. Subsampling...")
dataset.subsample(config["dataset"]["subsample"])
print(f"   After subsample: {len(dataset.rgb_files)} files")

print("5. Getting image shape...")
try:
    h, w = dataset.get_img_shape()[0]
    print(f"   ✓ Success! Shape: {h}x{w}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"   rgb_files length at error: {len(dataset.rgb_files)}")
    import traceback
    traceback.print_exc()
