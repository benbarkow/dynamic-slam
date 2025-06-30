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
        
        # Filter out dynamic regions using dynamic_mask
        if keyframe.dynamic_mask is not None:
            print("reconstruct with dynamic_mask")
            static_mask = ~keyframe.dynamic_mask.cpu().numpy().reshape(-1)
            valid = valid & static_mask
        
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        
        original_img = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        
        if keyframe.dynamic_mask is not None:
            dynamic_mask = keyframe.dynamic_mask.cpu().numpy().squeeze()  # Remove the first dimension
            mask_img = (dynamic_mask * 255).astype(np.uint8)
            mask_img_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
            combined_img = np.hstack([original_img, mask_img_rgb])
        else:
            combined_img = original_img
        
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        )
        
        # Save dynamic_masks grid visualization
        if keyframe.dynamic_masks is not None:
            dm_filename = savedir / f"{t}_dm.png"
            save_dynamic_masks_grid(dm_filename, keyframe.dynamic_masks)


def save_dynamic_masks_grid(filename, dynamic_masks):
    """Save all dynamic masks in a grid layout"""
    masks = dynamic_masks.cpu().numpy()  # Shape: (b, h, w)
    b, h, w = masks.shape
    
    if b == 0:
        return
    
    # Calculate grid dimensions (try to make it roughly square)
    grid_cols = int(np.ceil(np.sqrt(b)))
    grid_rows = int(np.ceil(b / grid_cols))
    
    # Create grid image
    grid_h = grid_rows * h
    grid_w = grid_cols * w
    grid_img = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    for idx in range(b):
        row = idx // grid_cols
        col = idx % grid_cols
        
        start_h = row * h
        end_h = start_h + h
        start_w = col * w
        end_w = start_w + w
        
        mask_img = (masks[idx] * 255).astype(np.uint8)
        grid_img[start_h:end_h, start_w:end_w] = mask_img
    
    # Convert to RGB and save
    grid_img_rgb = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(
        str(filename),
        cv2.cvtColor(grid_img_rgb, cv2.COLOR_RGB2BGR)
    )


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
