"""Single-thread version of MASt3R-SLAM for Windows
This version bypasses multiprocessing entirely to avoid Windows spawn mode issues.
"""
import argparse
import datetime
import pathlib
import sys
import time
import lietorch
import torch
import yaml

from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, create_frame
from mast3r_slam.frame_singlethread import SingleThreadKeyframes, SingleThreadStates
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.global_opt import FactorGraph


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    """Relocalization function (runs inline in main thread)"""
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print(f"RELOCALIZING against kf {n_kf - 1} and {kf_idx}")
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]]
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)
    print("="*60)
    print("MASt3R-SLAM - Single-Thread Mode (Windows)")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print()

    # Load dataset
    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    has_calib = hasattr(dataset, "camera_intrinsics")
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)

    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )

    # Create single-thread structures (no multiprocessing)
    print("Initializing (single-thread mode)...")
    keyframes = SingleThreadKeyframes(h, w, device=device)
    states = SingleThreadStates(h, w, device=device)

    if K is not None:
        keyframes.set_intrinsics(K)

    # Load model
    print("Loading MASt3R model...")
    model = load_mast3r(device=device)
    print("Model loaded")

    # Prepare save directory
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    # Create tracker and optimization structures
    tracker = FrameTracker(model, keyframes, device)
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    print(f"\nProcessing {len(dataset)} frames...")
    print("="*60)

    fps_timer = time.time()
    mode = Mode.INIT

    for i in range(len(dataset)):
        timestamp, img = dataset[i]

        # Get previous camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize first frame
            print(f"Frame {i}: INIT")
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)

            # Run initial optimization inline
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()

            mode = Mode.TRACKING
            states.set_mode(mode)
            states.set_frame(frame)
            continue

        if mode == Mode.TRACKING:
            # Track frame
            add_new_kf, match_info, try_reloc = tracker.track(frame)

            print(f"Frame {i}: TRACKING - match_frac={match_info['match_frac']:.4f}, add_kf={add_new_kf}, reloc={try_reloc}")

            if try_reloc:
                mode = Mode.RELOC
                states.set_mode(mode)

            states.set_frame(frame)

            if add_new_kf:
                keyframes.append(frame)
                kf_idx = len(keyframes) - 1

                # Run local optimization inline
                if kf_idx > 0:
                    frame_idx = [kf_idx]
                    kf_idx_list = [kf_idx - 1]
                    factor_graph.add_factors(
                        frame_idx,
                        kf_idx_list,
                        config["local_opt"]["min_match_frac"],
                        is_reloc=False,
                    )
                    if config["use_calib"]:
                        factor_graph.solve_GN_calib()
                    else:
                        factor_graph.solve_GN_rays()

        elif mode == Mode.RELOC:
            # Relocalize
            print(f"Frame {i}: RELOCALIZING...")
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)

            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                mode = Mode.TRACKING
                states.set_mode(mode)
                print(f"  -> Relocalization successful, back to TRACKING")
            else:
                print(f"  -> Relocalization failed, will retry")

        # Print progress every 10 frames
        if (i + 1) % 10 == 0:
            elapsed = time.time() - fps_timer
            fps = 10 / elapsed if elapsed > 0 else 0
            print(f"  Progress: {i+1}/{len(dataset)} frames, {fps:.2f} fps, {len(keyframes)} keyframes")
            fps_timer = time.time()

    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Total keyframes: {len(keyframes)}")

    # Save results
    if dataset.save_results:
        print(f"\nSaving results to {save_dir}/")

        # Save trajectory
        with open(traj_file, "w") as f:
            for i in range(len(keyframes)):
                kf = keyframes[i]
                T = kf.T_WC.matrix().cpu().numpy()[0]
                t = T[:3, 3]
                R = T[:3, :3]
                from scipy.spatial.transform import Rotation
                q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
                timestamp = kf.frame_id  # Use frame index as timestamp
                f.write(f"{timestamp} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

        print(f"  Trajectory saved: {traj_file}")

        # Save point cloud
        import numpy as np
        from plyfile import PlyData, PlyElement

        all_points = []
        all_colors = []
        for i in range(len(keyframes)):
            kf = keyframes[i]
            X = kf.X_canon.cpu().numpy()
            C = kf.C.cpu().numpy()
            img = kf.uimg.cpu().numpy()

            # Filter by confidence
            valid = C[:, 0] > 2.0
            X_valid = X[valid]

            # Get colors
            h, w = kf.img_shape[0].cpu().numpy()
            colors = img.reshape(-1, 3)[valid]

            all_points.append(X_valid)
            all_colors.append(colors)

        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        colors = (colors * 255).astype(np.uint8)

        vertex = np.array(
            [(points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
             for i in range(len(points))],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(str(recon_file))

        print(f"  Point cloud saved: {recon_file} ({len(points)} points)")

    print("\nDone!")


if __name__ == "__main__":
    main()
