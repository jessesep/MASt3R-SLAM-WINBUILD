"""
Download smallest test dataset and model checkpoint for MASt3R-SLAM
This script downloads:
1. TUM freiburg1_xyz dataset (~460 MB) - smallest sequence
2. MASt3R model checkpoint from Hugging Face
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path

print("="*80)
print("MASt3R-SLAM Test Data Downloader")
print("="*80)
print()

# Create directories
print("Creating directories...")
os.makedirs("datasets/tum", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Download TUM xyz dataset
dataset_url = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
dataset_file = "datasets/tum/rgbd_dataset_freiburg1_xyz.tgz"

if not os.path.exists("datasets/tum/rgbd_dataset_freiburg1_xyz"):
    print(f"Downloading TUM freiburg1_xyz dataset (~460 MB)...")
    print(f"From: {dataset_url}")
    print(f"To: {dataset_file}")
    print()

    def download_progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}% ({count * block_size / 1024**2:.1f} MB / {total_size / 1024**2:.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(dataset_url, dataset_file, download_progress)
        print("\n  Download complete!")

        # Extract
        print("  Extracting...")
        with tarfile.open(dataset_file, 'r:gz') as tar:
            tar.extractall("datasets/tum")
        print("  Extraction complete!")

        # Cleanup
        print("  Cleaning up archive...")
        os.remove(dataset_file)
        print("  Done!")
    except Exception as e:
        print(f"\n  Error downloading dataset: {e}")
        print("  Please download manually from:")
        print(f"  {dataset_url}")
else:
    print("Dataset already exists. Skipping download.")

print()

# Download model checkpoint
print("Downloading MASt3R model checkpoint...")
print("This requires huggingface_hub package...")

try:
    from huggingface_hub import hf_hub_download

    checkpoint_path = Path("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")

    if not checkpoint_path.exists():
        print("Downloading from Hugging Face (this may take a while, ~1.5 GB)...")
        hf_hub_download(
            repo_id="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            filename="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            local_dir="checkpoints",
            local_dir_use_symlinks=False
        )
        print("  Model checkpoint downloaded!")
    else:
        print("Model checkpoint already exists. Skipping download.")
except ImportError:
    print("  ERROR: huggingface_hub not installed")
    print("  Install with: pip install huggingface-hub")
    print("  Or download manually from:")
    print("  https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
except Exception as e:
    print(f"  Error downloading model: {e}")
    print("  Please download manually from:")
    print("  https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")

print()
print("="*80)
print("Download Complete!")
print("="*80)
print()
print("Next steps:")
print("1. Activate environment: venv\\Scripts\\activate")
print("2. Run SLAM:")
print("   python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz --config config/base.yaml")
print()
print("The GUI will open automatically showing:")
print("  - 3D reconstruction")
print("  - Camera trajectory")
print("  - Real-time SLAM progress")
print()
print("To run without GUI (headless):")
print("   python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_xyz --config config/base.yaml --no-viz")
print()
