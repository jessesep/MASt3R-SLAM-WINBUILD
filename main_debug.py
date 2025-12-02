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


if __name__ == "__main__":
    print("[DEBUG 1] Starting main...", flush=True)
    mp.set_start_method("spawn")
    print("[DEBUG 2] set_start_method done", flush=True)

    # WINDOWS FIX: Use file_system sharing strategy for CUDA tensors in spawn mode
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("[DEBUG 3] set_sharing_strategy done", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")
    print("[DEBUG 4] Basic setup done", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--use-threading", action="store_true", help="Use threading.Thread instead of multiprocessing (better for Windows)")
    parser.add_argument("--no-backend", action="store_true", help="Run without backend thread (true single-thread for Windows)")

    args = parser.parse_args()
    print("[DEBUG 5] Args parsed", flush=True)

    load_config(args.config)
    print("[DEBUG 6] Config loaded", flush=True)
    print(args.dataset)
    print(config)

    # WINDOWS FIX: Don't use Manager if using threading mode
    if args.use_threading or args.no_backend:
        print("[DEBUG 7] Using single-thread mode (no Manager)", flush=True)
        manager = None
        main2viz = None
        viz2main = None
        args.no_viz = True
    else:
        print("[DEBUG 7] Creating multiprocessing Manager", flush=True)
        manager = mp.Manager()
        main2viz = new_queue(manager, args.no_viz)
        viz2main = new_queue(manager, args.no_viz)

    print("[DEBUG 8] Loading dataset...", flush=True)
    dataset = load_dataset(args.dataset)
    print("[DEBUG 9] Dataset loaded, subsampling...", flush=True)
    dataset.subsample(config["dataset"]["subsample"])
    print("[DEBUG 10] Getting image shape...", flush=True)
    h, w = dataset.get_img_shape()[0]
    print(f"[DEBUG 11] Image shape: {h}x{w}", flush=True)

    if args.calib:
        print("[DEBUG 12] Loading calibration...", flush=True)
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
        print("[DEBUG 13] Calibration loaded", flush=True)

    # WINDOWS FIX: Use threading-compatible versions when in threading mode or no-backend mode
    if args.use_threading or args.no_backend:
        print("[DEBUG 14] Creating SingleThread keyframes/states...", flush=True)
        keyframes = SingleThreadKeyframes(h, w)
        states = SingleThreadStates(h, w)
    else:
        print("[DEBUG 14] Creating Shared keyframes/states...", flush=True)
        keyframes = SharedKeyframes(manager, h, w)
        states = SharedStates(manager, h, w)
    print("[DEBUG 15] Keyframes/states created", flush=True)

    if not args.no_viz:
        print("[DEBUG 16] Starting visualization process...", flush=True)
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()
        print("[DEBUG 17] Viz started", flush=True)
    else:
        print("[DEBUG 16] Skipping visualization (--no-viz)", flush=True)

    print("[DEBUG 18] Loading MASt3R model...", flush=True)
    model = load_mast3r(device=device)
    print("[DEBUG 19] Model loaded, calling share_memory()...", flush=True)
    model.share_memory()
    print("[DEBUG 20] share_memory() done", flush=True)

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]
    print(f"[DEBUG 21] has_calib={has_calib}, use_calib={use_calib}", flush=True)

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        print("[DEBUG 22] Setting up intrinsics...", flush=True)
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)
        print("[DEBUG 23] Intrinsics set", flush=True)

    # remove the trajectory from the previous run
    if dataset.save_results:
        print("[DEBUG 24] Preparing save directory...", flush=True)
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()
        print("[DEBUG 25] Save directory ready", flush=True)

    print("[DEBUG 26] Creating FrameTracker...", flush=True)
    tracker = FrameTracker(model, keyframes, device)
    print("[DEBUG 27] Tracker created", flush=True)

    last_msg = WindowMsg()

    # WINDOWS FIX: Use threading.Thread instead of multiprocessing for better Windows compatibility
    if args.no_backend:
        print("[DEBUG 28] Running WITHOUT backend thread", flush=True)
        backend = None
    elif args.use_threading:
        print("[DEBUG 28] Starting threading backend...", flush=True)
        backend = threading.Thread(target=run_backend, args=(config, model, states, keyframes, K), daemon=True)
        backend.start()
        print("[DEBUG 29] Threading backend started", flush=True)
    else:
        print("[DEBUG 28] Starting multiprocessing backend...", flush=True)
        backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
        backend.start()
        print("[DEBUG 29] Backend started", flush=True)

    i = 0
    fps_timer = time.time()
    frames = []

    print("[DEBUG 30] Entering main loop...", flush=True)
    while True:
        print(f"[DEBUG 31] Loop iteration {i}", flush=True)
        mode = states.get_mode()
        print(f"[DEBUG 32] Mode: {mode}", flush=True)

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
            print("[DEBUG 33] Reached end of dataset", flush=True)
            states.set_mode(Mode.TERMINATED)
            break

        print(f"[DEBUG 34] Loading frame {i}...", flush=True)
        timestamp, img = dataset[i]
        print(f"[DEBUG 35] Frame loaded, img shape: {img.shape}", flush=True)

        if save_frames:
            frames.append(img)

        # get frames last camera pose
        print(f"[DEBUG 36] Getting camera pose (i={i})...", flush=True)
        if i == 0:
            print("[DEBUG 37] Creating identity Sim3...", flush=True)
            T_WC = lietorch.Sim3.Identity(1, device=device)
            print("[DEBUG 38] Identity Sim3 created", flush=True)
        else:
            print("[DEBUG 37] Getting T_WC from previous frame...", flush=True)
            T_WC = states.get_frame().T_WC
            print("[DEBUG 38] T_WC retrieved", flush=True)

        print(f"[DEBUG 39] Creating frame {i}...", flush=True)
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        print(f"[DEBUG 40] Frame {i} created", flush=True)

        if mode == Mode.INIT:
            print("[DEBUG 41] INIT mode: running mono inference...", flush=True)
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            print("[DEBUG 42] Mono inference done, updating pointmap...", flush=True)
            frame.update_pointmap(X_init, C_init)
            print("[DEBUG 43] Appending to keyframes...", flush=True)
            keyframes.append(frame)
            print("[DEBUG 44] Queueing global optimization...", flush=True)
            states.queue_global_optimization(len(keyframes) - 1)
            print("[DEBUG 45] Setting mode to TRACKING...", flush=True)
            states.set_mode(Mode.TRACKING)
            print("[DEBUG 46] Setting frame...", flush=True)
            states.set_frame(frame)
            print("[DEBUG 47] INIT complete, continuing...", flush=True)
            i += 1
            continue

        if mode == Mode.TRACKING:
            print("[DEBUG 48] TRACKING mode: calling tracker.track()...", flush=True)
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            print(f"[DEBUG 49] track() done: add_new_kf={add_new_kf}, try_reloc={try_reloc}", flush=True)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            print("[DEBUG 50] RELOC mode", flush=True)
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            print("[DEBUG 51] Adding new keyframe...", flush=True)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
            print("[DEBUG 52] New keyframe added", flush=True)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    print("[DEBUG 53] Main loop finished", flush=True)
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

    print("[DEBUG 54] Cleanup starting...", flush=True)
    print("done")
    if backend is not None:
        backend.join()
    if not args.no_viz:
        viz.join()
    print("[DEBUG 55] All done!", flush=True)
