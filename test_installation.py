"""
MASt3R-SLAM Installation Test Suite
This script tests all components of the MASt3R-SLAM installation
"""

import sys
import os
from datetime import datetime

# Create logs directory
os.makedirs("test_logs", exist_ok=True)

# Setup logging
log_file = f"test_logs/installation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message, level="INFO"):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

def test_section(name):
    """Print test section header"""
    log("=" * 80)
    log(f"  {name}")
    log("=" * 80)

def test_import(module_name, description):
    """Test importing a module"""
    try:
        log(f"Testing import: {module_name} ({description})")
        module = __import__(module_name)
        version = getattr(module, "__version__", "N/A")
        log(f"  SUCCESS: {module_name} imported (version: {version})", "SUCCESS")
        return True, module
    except Exception as e:
        log(f"  FAILED: {module_name} - {str(e)}", "ERROR")
        return False, None

def main():
    log("MASt3R-SLAM Installation Test Suite")
    log(f"Python: {sys.version}")
    log(f"Platform: {sys.platform}")
    log("")

    results = {}

    # Test 1: Core Dependencies
    test_section("TEST 1: Core Python Dependencies")
    results["numpy"], np = test_import("numpy", "NumPy numerical library")
    results["torch"], torch = test_import("torch", "PyTorch deep learning framework")
    results["scipy"], _ = test_import("scipy", "Scientific computing library")
    results["matplotlib"], _ = test_import("matplotlib", "Plotting library")
    results["cv2"], cv2 = test_import("cv2", "OpenCV computer vision")

    # Test 2: PyTorch CUDA Support
    test_section("TEST 2: PyTorch CUDA Support")
    if torch:
        log(f"PyTorch version: {torch.__version__}")
        log(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"CUDA version: {torch.version.cuda}")
            log(f"cuDNN version: {torch.backends.cudnn.version()}")
            log(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                log(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
                log(f"  Compute capability: {props.major}.{props.minor}")
            results["cuda"] = True
        else:
            log("CUDA not available!", "WARNING")
            results["cuda"] = False

    # Test 3: CUDA Extensions
    test_section("TEST 3: CUDA Extensions")
    results["lietorch"], lietorch = test_import("lietorch", "Lie algebra CUDA operations")
    results["curope"], curope = test_import("curope", "CUDA rotary position embeddings")

    # Test 4: MASt3R Components
    test_section("TEST 4: MASt3R Components")
    results["dust3r"], _ = test_import("dust3r", "DUSt3R dependency")
    results["mast3r"], mast3r = test_import("mast3r", "MASt3R model")
    results["mast3r_slam"], mast3r_slam = test_import("mast3r_slam", "MASt3R-SLAM main package")

    # Test 5: Additional Dependencies
    test_section("TEST 5: Additional Dependencies")
    results["einops"], _ = test_import("einops", "Tensor operations")
    results["trimesh"], _ = test_import("trimesh", "3D mesh processing")
    results["gradio"], _ = test_import("gradio", "Web UI framework")
    results["plyfile"], _ = test_import("plyfile", "PLY file format")
    results["evo"], _ = test_import("evo", "Trajectory evaluation")

    # Test 6: CUDA Tensor Operations
    test_section("TEST 6: CUDA Tensor Operations")
    if torch and torch.cuda.is_available():
        try:
            log("Creating CUDA tensor...")
            x = torch.randn(100, 100, device='cuda')
            log(f"  CUDA tensor created: shape={x.shape}, device={x.device}")

            log("Testing CUDA computation...")
            y = torch.matmul(x, x.T)
            log(f"  Matrix multiplication successful: result shape={y.shape}")

            log("Testing CPU-GPU transfer...")
            z = y.cpu()
            log(f"  Transfer successful: result on CPU")
            results["cuda_ops"] = True
        except Exception as e:
            log(f"  FAILED: CUDA operations - {str(e)}", "ERROR")
            results["cuda_ops"] = False
    else:
        log("Skipping CUDA operations test (CUDA not available)", "WARNING")
        results["cuda_ops"] = False

    # Test 7: LieTorch Operations
    test_section("TEST 7: LieTorch Operations")
    if lietorch and torch and torch.cuda.is_available():
        try:
            log("Testing LieTorch SE3 operations...")
            from lietorch import SE3

            # Create random SE3 transformations
            poses = SE3.InitFromVec(torch.randn(10, 7, device='cuda'))
            log(f"  Created SE3 poses: shape={poses.shape}")

            # Test SE3 operations
            inv_poses = poses.inv()
            log(f"  SE3 inverse computed: shape={inv_poses.shape}")

            composed = poses * inv_poses
            log(f"  SE3 composition successful")
            results["lietorch_ops"] = True
        except Exception as e:
            log(f"  FAILED: LieTorch operations - {str(e)}", "ERROR")
            results["lietorch_ops"] = False
    else:
        log("Skipping LieTorch operations test", "WARNING")
        results["lietorch_ops"] = False

    # Test 8: MASt3R-SLAM Backend
    test_section("TEST 8: MASt3R-SLAM Backend")
    if mast3r_slam:
        try:
            log("Testing MASt3R-SLAM backend import...")
            import mast3r_slam_backends
            log(f"  Backend module imported successfully")
            results["backend"] = True
        except Exception as e:
            log(f"  FAILED: Backend import - {str(e)}", "ERROR")
            results["backend"] = False
    else:
        log("Skipping backend test (mast3r_slam not available)", "WARNING")
        results["backend"] = False

    # Test 9: Check Model Checkpoints
    test_section("TEST 9: Model Checkpoints")
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        log(f"Checkpoints directory exists: {checkpoints_dir}")
        files = os.listdir(checkpoints_dir)
        log(f"Found {len(files)} files:")
        for f in files:
            size = os.path.getsize(os.path.join(checkpoints_dir, f))
            log(f"  {f} ({size / 1024**2:.2f} MB)")
        results["checkpoints"] = len(files) > 0
    else:
        log(f"Checkpoints directory not found: {checkpoints_dir}", "WARNING")
        results["checkpoints"] = False

    # Summary
    test_section("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    log(f"Total tests: {total}")
    log(f"Passed: {passed}")
    log(f"Failed: {failed}")
    log(f"Success rate: {passed/total*100:.1f}%")
    log("")

    log("Detailed results:")
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        level = "SUCCESS" if result else "ERROR"
        log(f"  {test_name:20s}: {status}", level)

    log("")
    log(f"Full test log saved to: {log_file}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
