"""
Test lietorch operations in isolation to find which one crashes.
"""

import torch
import lietorch
import sys

def test_lietorch_operations():
    print("="*80)
    print("TESTING LIETORCH OPERATIONS ON WINDOWS")
    print("="*80)

    device = "cuda:0"
    print(f"\nDevice: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Test 1: Create identity Sim3
    print("\n[TEST 1] Creating Sim3 identity...")
    try:
        T1 = lietorch.Sim3.Identity(1, device=device)
        print(f"  SUCCESS: T1 created, shape={T1.shape}")
        print(f"  T1 data: {T1.data}")
        torch.cuda.synchronize()
        print(f"  CUDA sync OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # Test 2: Create another identity
    print("\n[TEST 2] Creating another Sim3 identity...")
    try:
        T2 = lietorch.Sim3.Identity(1, device=device)
        print(f"  SUCCESS: T2 created")
        torch.cuda.synchronize()
        print(f"  CUDA sync OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # Test 3: Invert T1
    print("\n[TEST 3] Testing T1.inv()...")
    try:
        torch.cuda.synchronize()
        print(f"  Before inv(): CUDA sync OK")
        T1_inv = T1.inv()
        torch.cuda.synchronize()
        print(f"  After inv(): CUDA sync OK")
        print(f"  SUCCESS: T1_inv created")
        print(f"  T1_inv data: {T1_inv.data}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 4: Multiply two Sim3
    print("\n[TEST 4] Testing T1 * T2...")
    try:
        torch.cuda.synchronize()
        print(f"  Before multiply: CUDA sync OK")
        T3 = T1 * T2
        torch.cuda.synchronize()
        print(f"  After multiply: CUDA sync OK")
        print(f"  SUCCESS: T3 = T1 * T2")
        print(f"  T3 data: {T3.data}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 5: Act on points
    print("\n[TEST 5] Testing T1.act(points)...")
    try:
        points = torch.randn(100, 3, device=device)
        print(f"  Created random points: shape={points.shape}")
        torch.cuda.synchronize()
        print(f"  Before act(): CUDA sync OK")

        transformed = T1.act(points)

        torch.cuda.synchronize()
        print(f"  After act(): CUDA sync OK")
        print(f"  SUCCESS: transformed points shape={transformed.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 6: Retraction
    print("\n[TEST 6] Testing T1.retr(delta)...")
    try:
        delta = torch.randn(1, 7, device=device) * 0.01  # Small perturbation
        print(f"  Created delta: shape={delta.shape}")
        torch.cuda.synchronize()
        print(f"  Before retr(): CUDA sync OK")

        T1_new = T1.retr(delta)

        torch.cuda.synchronize()
        print(f"  After retr(): CUDA sync OK")
        print(f"  SUCCESS: T1_new created")
        print(f"  T1_new data: {T1_new.data}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 7: Large batch of points
    print("\n[TEST 7] Testing T1.act() with large batch (196608 points)...")
    try:
        large_points = torch.randn(196608, 3, device=device)
        print(f"  Created large_points: shape={large_points.shape}")
        torch.cuda.synchronize()
        print(f"  Before act(): CUDA sync OK")

        transformed_large = T1.act(large_points)

        torch.cuda.synchronize()
        print(f"  After act(): CUDA sync OK")
        print(f"  SUCCESS: transformed_large shape={transformed_large.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*80)
    print("ALL LIETORCH TESTS PASSED!")
    print("="*80)

if __name__ == "__main__":
    try:
        test_lietorch_operations()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
