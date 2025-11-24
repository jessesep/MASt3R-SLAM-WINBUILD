import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes, dataset=None):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    
    # MODIFICATION by Ben Williams (2025-11-21 & 2025-11-24): Save keyframe mapping file
    # This records which original high-res image corresponds to each keyframe
    # Format: timestamp frame_id original_filename
    # This enables using full-resolution images for splatting while keeping
    # MASt3R-SLAM's downsampled keyframes for pose estimation
    mapping_file = savedir.parent / "keyframe_mapping.txt"
    
    with open(mapping_file, 'w') as f:
        f.write("# Keyframe mapping: timestamp → original image\n")
        f.write("# Format: timestamp frame_id original_filename\n")
        f.write("# Use this to map MASt3R-SLAM keyframes to original high-res images\n")
        f.write(f"# Total keyframes: {len(keyframes)}\n")
        f.write("m-slam_file index original_file\n")
        
        for i in range(len(keyframes)):
            keyframe = keyframes[i]
            t = timestamps[keyframe.frame_id]
            frame_id = keyframe.frame_id
            
            # Save downsampled keyframe (current behavior)
            filename = savedir / f"{t}.png"
            cv2.imwrite(
                str(filename),
                cv2.cvtColor(
                    (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                ),
            )
            
            # Write mapping entry with actual original filename if dataset is provided
            if dataset is not None and hasattr(dataset, 'rgb_files') and frame_id < len(dataset.rgb_files):
                # Get the original filename from dataset.rgb_files[frame_id]
                original_file = dataset.rgb_files[frame_id]
                original_filename = original_file.name if hasattr(original_file, 'name') else str(original_file).split('/')[-1]
                f.write(f'{t} {frame_id} "{original_filename}"\n')
            else:
                # Fallback: just write frame_id if dataset not available
                f.write(f'{t} {frame_id} ""\n')
    
    print(f"✓ Saved keyframe mapping to {mapping_file}")


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
