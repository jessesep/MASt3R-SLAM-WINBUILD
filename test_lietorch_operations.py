"""Test each lietorch operation individually to find which ones crash"""
import sys
import torch
import lietorch

print("="*80, flush=True)
print("TESTING INDIVIDUAL LIETORCH OPERATIONS", flush=True)
print("="*80, flush=True)

device = "cuda:0"
print(f"\nDevice: {device}", flush=True)
print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)

# Create test data
print("\n[SETUP] Creating Sim3 matrices...", flush=True)
T1 = lietorch.Sim3.Identity(1, device=device)
T2 = lietorch.Sim3.Identity(1, device=device)
points = torch.randn(1000, 3, device=device)
delta = torch.randn(1, 7, device=device) * 0.01
print(f"  T1: {T1.data}", flush=True)
print(f"  T2: {T2.data}", flush=True)
print(f"  points shape: {points.shape}", flush=True)
print(f"  delta shape: {delta.shape}", flush=True)

# Test each operation individually
tests = []

# Test 1: Multiplication
print("\n[TEST 1] T1 * T2 (multiplication)...", flush=True)
try:
    torch.cuda.synchronize()
    result = T1 * T2
    torch.cuda.synchronize()
    print(f"  ✓ SUCCESS: {result.data}", flush=True)
    tests.append(("multiplication", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    tests.append(("multiplication", False))

# Test 2: Act on points
print("\n[TEST 2] T1.act(points) (point transformation)...", flush=True)
try:
    torch.cuda.synchronize()
    result = T1.act(points)
    torch.cuda.synchronize()
    print(f"  ✓ SUCCESS: transformed shape={result.shape}", flush=True)
    tests.append(("act", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    tests.append(("act", False))

# Test 3: Retraction
print("\n[TEST 3] T1.retr(delta) (retraction)...", flush=True)
try:
    torch.cuda.synchronize()
    result = T1.retr(delta)
    torch.cuda.synchronize()
    print(f"  ✓ SUCCESS: {result.data}", flush=True)
    tests.append(("retraction", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    tests.append(("retraction", False))

# Test 4: Adjoint
print("\n[TEST 4] T1.adjT(delta) (adjoint transpose)...", flush=True)
try:
    torch.cuda.synchronize()
    result = T1.adjT(delta)
    torch.cuda.synchronize()
    print(f"  ✓ SUCCESS: {result.shape}", flush=True)
    tests.append(("adjoint", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    tests.append(("adjoint", False))

# Test 5: Inversion (we know this crashes, test it last)
print("\n[TEST 5] T1.inv() (inversion)...", flush=True)
try:
    torch.cuda.synchronize()
    result = T1.inv()
    torch.cuda.synchronize()
    print(f"  ✓ SUCCESS: {result.data}", flush=True)
    tests.append(("inversion", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}", flush=True)
    tests.append(("inversion", False))

# Summary
print("\n" + "="*80, flush=True)
print("TEST SUMMARY", flush=True)
print("="*80, flush=True)
for name, success in tests:
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {status}: {name}", flush=True)

passed = sum(1 for _, s in tests if s)
total = len(tests)
print(f"\nTotal: {passed}/{total} tests passed", flush=True)

if passed == total:
    print("\n✓ ALL TESTS PASSED!", flush=True)
    sys.exit(0)
else:
    print(f"\n✗ {total - passed} TESTS FAILED", flush=True)
    sys.exit(1)
