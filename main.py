import argparse
import datetime
import pathlib
import sys
import time
import cv2
import mast3r_slam.lietorch_compat as lietorch  # PyTorch-based Sim3
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.frame_singlethread import SingleThreadKeyframes, SingleThreadStates
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
import threading


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
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
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
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
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        # DIAGNOSTIC: Track pose changes before/after optimization
        import numpy as np
        poses_before = {}
        for i in range(len(keyframes)):
            try:
                kf = keyframes[i]
                if kf is not None and hasattr(kf, 'T_WC') and kf.T_WC is not None:
                    mat = kf.T_WC.matrix()
                    if mat.dim() == 3:
                        poses_before[i] = mat[0, :3, 3].cpu().numpy().copy()
                    else:
                        poses_before[i] = mat[:3, 3].cpu().numpy().copy()
            except:
                pass

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        # DIAGNOSTIC: Compare poses after optimization
        poses_after = {}
        max_change = 0.0
        changed_kfs = []
        for i in range(len(keyframes)):
            try:
                kf = keyframes[i]
                if kf is not None and hasattr(kf, 'T_WC') and kf.T_WC is not None:
                    mat = kf.T_WC.matrix()
                    if mat.dim() == 3:
                        poses_after[i] = mat[0, :3, 3].cpu().numpy().copy()
                    else:
                        poses_after[i] = mat[:3, 3].cpu().numpy().copy()

                    if i in poses_before:
                        diff = np.linalg.norm(poses_after[i] - poses_before[i])
                        if diff > 0.01:  # More than 1cm change
                            changed_kfs.append((i, diff))
                            max_change = max(max_change, diff)
            except:
                pass

        if changed_kfs:
            print(f"[DIAGNOSTIC] Backend optimization changed {len(changed_kfs)} keyframe poses:")
            print(f"  Max change: {max_change:.3f}m")
            print(f"  Changed keyframes: {[kf for kf, _ in changed_kfs[:10]]}")  # Show first 10
            if max_change > 0.5:
                print(f"  ** WARNING: Large jump detected! ({max_change:.3f}m) **")

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # WINDOWS FIX: Use file_system sharing strategy for CUDA tensors in spawn mode
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--use-threading", action="store_true", help="Use threading.Thread instead of multiprocessing (better for Windows)")
    parser.add_argument("--no-backend", action="store_true", help="Run without backend thread (true single-thread for Windows)")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    # WINDOWS FIX: Use threading-compatible queues if using threading mode
    if args.use_threading:
        print("=" * 60)
        print("WINDOWS THREADING MODE")
        print("Using threading instead of multiprocessing")
        print("=" * 60)
        manager = None  # Will use threading primitives instead
        # Use threading-safe queues for visualization
        import queue
        main2viz = queue.Queue() if not args.no_viz else None
        viz2main = queue.Queue() if not args.no_viz else None
    else:
        manager = mp.Manager()
        main2viz = new_queue(manager, args.no_viz)
        viz2main = new_queue(manager, args.no_viz)

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

    # WINDOWS FIX: Use threading-compatible versions when in threading mode or no-backend mode
    if args.use_threading or args.no_backend:
        keyframes = SingleThreadKeyframes(h, w)
        states = SingleThreadStates(h, w)
    else:
        keyframes = SharedKeyframes(manager, h, w)
        states = SharedStates(manager, h, w)

    if not args.no_viz:
        if args.use_threading:
            # Use threading.Thread for visualization in threading mode
            viz = threading.Thread(
                target=run_visualization,
                args=(config, states, keyframes, main2viz, viz2main),
                daemon=True
            )
        else:
            # Use multiprocessing.Process for standard mode
            viz = mp.Process(
                target=run_visualization,
                args=(config, states, keyframes, main2viz, viz2main),
            )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    # WINDOWS FIX: Use threading.Thread instead of multiprocessing for better Windows compatibility
    if args.no_backend:
        print("Running WITHOUT backend thread (true single-thread mode for Windows)")
        backend = None
    elif args.use_threading:
        print("Using threading.Thread for backend (Windows mode)")
        backend = threading.Thread(target=run_backend, args=(config, model, states, keyframes, K), daemon=True)
        backend.start()
    else:
        backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
        backend.start()

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main) if viz2main is not None else None
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            # Only enter RELOC mode if backend exists to handle it
            if try_reloc and backend is not None:
                states.set_mode(Mode.RELOC)
            elif try_reloc and backend is None:
                # In --no-backend mode, if tracking fails, reinitialize with current frame
                print(f"[WARNING] Tracking failed for frame {i}, reinitializing as new keyframe (no backend)")
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                add_new_kf = True  # Force this frame to be a keyframe
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            # Only queue reloc if backend exists
            if backend is not None:
                states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"] and backend is not None:
                with states.lock:
                    # Handle both SingleThreadStates (int) and SharedStates (Value)
                    reloc_val = states.reloc_sem if isinstance(states.reloc_sem, int) else states.reloc_sem.value
                    if reloc_val == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            # Only queue optimization if backend exists
            if backend is not None:
                states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"] and backend is not None:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes, dataset
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
