"""Test lietorch operations to see if they cause segfault"""
import torch
import lietorch

print("="*60)
print("Testing lietorch Sim3 operations")
print("="*60)

device = "cuda:0"

print("\n1. Creating Sim3 identity...")
T1 = lietorch.Sim3.Identity(1, device=device)
print(f"   OK: {T1}")

print("\n2. Creating another Sim3...")
T2 = lietorch.Sim3.Identity(1, device=device)
print(f"   OK: {T2}")

print("\n3. Testing Sim3 inverse...")
try:
    T_inv = T1.inv()
    print(f"   OK: inverse computed")
except Exception as e:
    print(f"   FAIL: {e}")
    exit(1)

print("\n4. Testing Sim3 multiplication...")
try:
    T3 = T1 * T2
    print(f"   OK: multiplication computed")
except Exception as e:
    print(f"   FAIL: {e}")
    exit(1)

print("\n5. Testing Sim3.act (transform points)...")
try:
    points = torch.randn(100, 3, device=device)
    points_transformed = T1.act(points)
    print(f"   OK: act computed, shape={points_transformed.shape}")
except Exception as e:
    print(f"   FAIL: {e}")
    exit(1)

print("\n6. Testing Sim3.retr (retraction)...")
try:
    tau = torch.randn(1, 7, device=device) * 0.01
    T_new = T1.retr(tau)
    print(f"   OK: retr computed")
except Exception as e:
    print(f"   FAIL: {e}")
    exit(1)

print("\n7. Testing Sim3.matrix...")
try:
    mat = T1.matrix()
    print(f"   OK: matrix computed, shape={mat.shape}")
except Exception as e:
    print(f"   FAIL: {e}")
    exit(1)

print("\n8. Testing Sim3 with batch operations...")
try:
    T_batch = lietorch.Sim3.Identity(10, device=device)
    points_batch = torch.randn(10, 100, 3, device=device)
    result = T_batch.act(points_batch)
    print(f"   OK: batch operations work, shape={result.shape}")
except Exception as e:
    print(f"   FAIL: {e}")
    exit(1)

print("\n" + "="*60)
print("SUCCESS! All lietorch Sim3 operations work")
print("="*60)
