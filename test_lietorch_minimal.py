"""Minimal lietorch test"""
import sys
print("STEP 1: Starting script", flush=True)

import torch
print("STEP 2: Imported torch", flush=True)

print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

import lietorch
print("STEP 3: Imported lietorch", flush=True)

device = "cuda:0"
print(f"STEP 4: About to create Sim3 identity", flush=True)

T1 = lietorch.Sim3.Identity(1, device=device)
print(f"STEP 5: Created T1: {T1.data}", flush=True)

torch.cuda.synchronize()
print(f"STEP 6: CUDA synchronized", flush=True)

print(f"STEP 7: About to call T1.inv()", flush=True)
T1_inv = T1.inv()
print(f"STEP 8: T1.inv() completed: {T1_inv.data}", flush=True)

torch.cuda.synchronize()
print(f"STEP 9: CUDA synchronized after inv()", flush=True)

print("SUCCESS: All tests passed!", flush=True)
