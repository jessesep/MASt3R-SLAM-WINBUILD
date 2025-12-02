"""
Deep debugging of the tracker crash.
This adds comprehensive error checking at every CUDA operation.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from mast3r_slam.frame import Frame
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.geometry import act_Sim3, point_to_ray_dist
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric, load_mast3r
from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import load_dataset
import mast3r_slam.lietorch_compat as lietorch  # PyTorch-based Sim3

# Add CUDA error checking
def check_cuda_error(msg=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        error = torch.cuda.get_device_properties(0)
        print(f"[CUDA CHECK] {msg}: OK", flush=True)

def safe_lietorch_op(func, *args, msg="", **kwargs):
    """Wrapper for lietorch operations with error checking"""
    try:
        print(f"[LIETORCH] Before {msg}", flush=True)
        check_cuda_error(f"Before {msg}")
        result = func(*args, **kwargs)
        print(f"[LIETORCH] After {msg} - SUCCESS", flush=True)
        check_cuda_error(f"After {msg}")
        return result
    except Exception as e:
        print(f"[LIETORCH] CRASH in {msg}: {e}", flush=True)
        raise

def safe_torch_op(func, *args, msg="", **kwargs):
    """Wrapper for torch operations with error checking"""
    try:
        print(f"[TORCH] Before {msg}", flush=True)
        check_cuda_error(f"Before {msg}")
        result = func(*args, **kwargs)
        print(f"[TORCH] After {msg} - SUCCESS", flush=True)
        check_cuda_error(f"After {msg}")
        return result
    except Exception as e:
        print(f"[TORCH] CRASH in {msg}: {e}", flush=True)
        raise

def test_opt_pose_manually():
    """Manually reproduce the opt_pose_ray_dist_sim3 function with detailed checks"""

    print("="*80, flush=True)
    print("LOADING CONFIG AND MODEL", flush=True)
    print("="*80, flush=True)

    load_config("config/base.yaml")
    device = torch.device("cuda:0")

    model = load_mast3r(device=device)
    check_cuda_error("Model loaded")

    print("\n" + "="*80, flush=True)
    print("LOADING DATASET", flush=True)
    print("="*80, flush=True)

    dataloader = load_dataset("datasets/tum/rgbd_dataset_freiburg1_xyz")
    timestamp0, img0 = dataloader[0]
    timestamp1, img1 = dataloader[1]

    # Create frames
    from mast3r_slam.frame import create_frame
    from mast3r_slam.mast3r_utils import mast3r_inference_mono
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    T_WC1 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img0, T_WC0, img_size=dataloader.img_size, device=device)
    frame1 = create_frame(1, img1, T_WC1, img_size=dataloader.img_size, device=device)

    # Initialize with mono inference
    X0, C0 = mast3r_inference_mono(model, frame0)
    frame0.update_pointmap(X0, C0)
    X1, C1 = mast3r_inference_mono(model, frame1)
    frame1.update_pointmap(X1, C1)

    print(f"Frame 0: {frame0.frame_id}", flush=True)
    print(f"Frame 1: {frame1.frame_id}", flush=True)

    print("\n" + "="*80, flush=True)
    print("RUNNING MAST3R MATCHING", flush=True)
    print("="*80, flush=True)

    idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
        model, frame1, frame0, idx_i2j_init=None
    )
    check_cuda_error("mast3r_match_asymmetric")

    idx_f2k = idx_f2k[0]
    valid_match_k = valid_match_k[0]
    Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

    frame1.update_pointmap(Xff, Cff)

    # Get points and poses
    Xf = frame1.X_canon[idx_f2k]
    Xk = frame0.X_canon
    T_WCf = frame1.T_WC
    T_WCk = frame0.T_WC

    # Get valid mask
    Cf = frame1.get_average_conf()[idx_f2k]
    Ck = frame0.get_average_conf()

    cfg = config["tracking"]
    valid_Cf = Cf > cfg["C_conf"]
    valid_Ck = Ck > cfg["C_conf"]
    valid_Q = Qk > cfg["Q_conf"]
    valid = valid_match_k & valid_Cf & valid_Ck & valid_Q

    match_frac = valid.sum() / valid.numel()
    print(f"Match fraction: {match_frac:.4f}", flush=True)
    print(f"Valid matches: {valid.sum()}/{valid.numel()}", flush=True)

    print("\n" + "="*80, flush=True)
    print("STARTING POSE OPTIMIZATION", flush=True)
    print("="*80, flush=True)

    # Setup for optimization (from opt_pose_ray_dist_sim3)
    sqrt_info_ray = 1 / cfg["sigma_ray"] * valid * torch.sqrt(Qk)
    sqrt_info_dist = 1 / cfg["sigma_dist"] * valid * torch.sqrt(Qk)
    sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

    print(f"Xf shape: {Xf.shape}, device: {Xf.device}", flush=True)
    print(f"Xk shape: {Xk.shape}, device: {Xk.device}", flush=True)
    print(f"T_WCf type: {type(T_WCf)}", flush=True)
    print(f"T_WCk type: {type(T_WCk)}", flush=True)

    # Step 1: Test T_WCk.inv()
    print("\n[STEP 1] Testing T_WCk.inv()", flush=True)
    T_WCk_inv = safe_lietorch_op(T_WCk.inv, msg="T_WCk.inv()")

    # Step 2: Test relative pose computation
    print("\n[STEP 2] Testing T_CkCf = T_WCk.inv() * T_WCf", flush=True)
    T_CkCf = safe_lietorch_op(lambda: T_WCk_inv * T_WCf, msg="Relative pose multiply")

    # Step 3: Test point_to_ray_dist on keyframe points
    print("\n[STEP 3] Testing point_to_ray_dist(Xk, jacobian=False)", flush=True)
    rd_k = safe_torch_op(point_to_ray_dist, Xk, jacobian=False, msg="point_to_ray_dist(Xk)")
    print(f"rd_k shape: {rd_k.shape}", flush=True)

    # Step 4: Start optimization loop
    print("\n[STEP 4] Starting optimization loop", flush=True)
    old_cost = float("inf")
    for step in range(min(cfg["max_iters"], 3)):  # Only 3 iterations for testing
        print(f"\n--- Iteration {step} ---", flush=True)

        # Step 4a: act_Sim3 with jacobian
        print(f"[STEP 4a] Testing act_Sim3(T_CkCf, Xf, jacobian=True)", flush=True)

        try:
            print(f"  Calling T_CkCf.act(Xf)...", flush=True)
            check_cuda_error("Before T_CkCf.act")
            Xf_Ck = T_CkCf.act(Xf)
            check_cuda_error("After T_CkCf.act")
            print(f"  T_CkCf.act(Xf) SUCCESS", flush=True)
            print(f"  Xf_Ck shape: {Xf_Ck.shape}", flush=True)
        except Exception as e:
            print(f"  CRASH in T_CkCf.act(Xf): {e}", flush=True)
            raise

        try:
            print(f"  Computing jacobian manually...", flush=True)
            dpC_dt = torch.eye(3, device=Xf_Ck.device).repeat(*Xf_Ck.shape[:-1], 1, 1)
            from mast3r_slam.geometry import skew_sym
            dpC_dR = -skew_sym(Xf_Ck)
            dpc_ds = Xf_Ck.reshape(*Xf_Ck.shape[:-1], -1, 1)
            dXf_Ck_dT_CkCf = torch.cat([dpC_dt, dpC_dR, dpc_ds], dim=-1)
            print(f"  Jacobian computation SUCCESS", flush=True)
        except Exception as e:
            print(f"  CRASH in jacobian computation: {e}", flush=True)
            raise

        # Step 4b: point_to_ray_dist with jacobian
        print(f"[STEP 4b] Testing point_to_ray_dist(Xf_Ck, jacobian=True)", flush=True)
        rd_f_Ck, drd_f_Ck_dXf_Ck = safe_torch_op(
            point_to_ray_dist, Xf_Ck, jacobian=True,
            msg="point_to_ray_dist(Xf_Ck, jacobian=True)"
        )

        # Step 4c: Compute residual and Jacobian
        print(f"[STEP 4c] Computing residual and Jacobian", flush=True)
        r = safe_torch_op(lambda: rd_k - rd_f_Ck, msg="residual computation")
        J = safe_torch_op(lambda: -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf, msg="Jacobian matmul")

        # Step 4d: Solve linear system
        print(f"[STEP 4d] Solving linear system (Cholesky)", flush=True)
        whitened_r = sqrt_info * r
        from mast3r_slam.nonlinear_optimizer import huber
        robust_sqrt_info = sqrt_info * torch.sqrt(huber(whitened_r, k=cfg["huber"]))

        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)
        b = (robust_sqrt_info * r).view(-1, 1)
        H = safe_torch_op(lambda: A.T @ A, msg="H = A.T @ A")
        g = safe_torch_op(lambda: -A.T @ b, msg="g = -A.T @ b")
        cost = 0.5 * (b.T @ b).item()

        print(f"  Cost: {cost}", flush=True)
        print(f"  H shape: {H.shape}, condition: {torch.linalg.cond(H).item():.2e}", flush=True)

        try:
            print(f"  Attempting Cholesky decomposition...", flush=True)
            check_cuda_error("Before Cholesky")
            L = torch.linalg.cholesky(H, upper=False)
            check_cuda_error("After Cholesky")
            print(f"  Cholesky SUCCESS", flush=True)
        except Exception as e:
            print(f"  Cholesky FAILED: {e}", flush=True)
            print(f"  Matrix is likely not positive definite", flush=True)
            # Try to continue anyway
            print(f"  Skipping this iteration", flush=True)
            break

        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)
        print(f"  tau_j shape: {tau_j.shape}, norm: {tau_j.norm().item():.6f}", flush=True)

        # Step 4e: Retraction
        print(f"[STEP 4e] Testing T_CkCf.retr(tau_j)", flush=True)
        try:
            print(f"  Calling T_CkCf.retr...", flush=True)
            check_cuda_error("Before retr")
            T_CkCf = T_CkCf.retr(tau_j)
            check_cuda_error("After retr")
            print(f"  T_CkCf.retr SUCCESS", flush=True)
        except Exception as e:
            print(f"  CRASH in T_CkCf.retr: {e}", flush=True)
            raise

        # Check convergence
        if abs(old_cost - cost) < 1e-6:
            print(f"  Converged!", flush=True)
            break
        old_cost = cost

    print("\n" + "="*80, flush=True)
    print("TEST COMPLETED SUCCESSFULLY!", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    try:
        test_opt_pose_manually()
    except Exception as e:
        print(f"\n{'='*80}", flush=True)
        print(f"TEST FAILED WITH EXCEPTION:", flush=True)
        print(f"{'='*80}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
