"""Render dense confocal transients on a virtual relay surface."""

import argparse
import os
import time

import numpy as np
import scipy.io
import torch
import yaml
from easydict import EasyDict
from scipy.io import savemat
from tqdm import tqdm

from save_load_model import load_adaptive_model
from utils import maybe_compile


def _load_virtual_wall(config):
    """Return virtual relay points and their normals as NumPy arrays."""
    resolution = int(config.output_res)
    if config.get("use_auto_wall", True):
        side_length = float(config.output_wallsize)
        axis = np.linspace(side_length / 2, -side_length / 2, resolution)
        x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
        points = np.stack(
            [x_grid, y_grid, np.zeros_like(x_grid)], axis=-1
        ).reshape(-1, 3)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(points), 1))
        print(
            f"Using an automatically generated {resolution}x{resolution} planar wall."
        )
        return points, normals

    wall_path = config.virtual_wall_path
    wall_data = scipy.io.loadmat(wall_path)
    required = {"virtual_cam_pts", "wall_normals"}
    missing = required.difference(wall_data)
    if missing:
        raise KeyError(f"Virtual-wall file is missing variables: {sorted(missing)}")
    points = np.ascontiguousarray(wall_data["virtual_cam_pts"])
    normals = np.ascontiguousarray(wall_data["wall_normals"])
    if points.shape != normals.shape or points.shape[1] != 3:
        raise ValueError(
            "virtual_cam_pts and wall_normals must both have shape (N, 3)."
        )
    print(f"Using virtual relay surface: {wall_path}")
    return points, normals


def generate_resampled_prediction(config):
    """Synthesize a dense confocal transient grid from a trained 3D-GTR model."""
    start_time = time.time()
    output_dir = config.output_dir
    resolution = int(config.output_res)
    output_path = os.path.join(
        output_dir, f"resample_pred_trans_{resolution}x{resolution}.mat"
    )
    wall_points, wall_normals = _load_virtual_wall(config)

    model_path = config.get("model_path", os.path.join(output_dir, "final_model.pth"))
    model = maybe_compile(load_adaptive_model(model_path, device=config.device))
    model.eval()

    device = torch.device(config.device)
    batch_size = int(config.get("predict_batch_size", 128))
    wall_points = torch.as_tensor(wall_points, dtype=torch.float32, device=device)
    wall_normals = torch.as_tensor(wall_normals, dtype=torch.float32, device=device)
    output_dt = float(config.get("output_dT", config.dT))
    model.ensure_timebin_buffer(
        config.output_t_range, device=device, dtype=torch.float32
    )

    predictions = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(wall_points), batch_size), desc="Rendering virtual wall"
        ):
            points = wall_points[start : start + batch_size]
            normals = wall_normals[start : start + batch_size]
            prediction = model(
                points,
                output_dt,
                config.output_t_range,
                laser_points=points,
                render_magnification=config.render_magnification,
                laser_normals=normals,
                camera_normals=normals,
            )
            predictions.append(prediction.cpu())

    predicted_transients = torch.cat(predictions, dim=0).numpy()
    savemat(
        output_path,
        {
            "camera_points": wall_points.cpu().numpy(),
            "pred_trans": predicted_transients,
            "new_res": resolution,
            "new_wallsize": float(config.output_wallsize),
            "dT": output_dt,
            "t_range": np.asarray(config.output_t_range),
        },
    )
    elapsed = time.time() - start_time
    print(f"Saved virtual-wall transients to {output_path}")
    print(f"Rendering time: {elapsed:.2f} seconds")
    print(
        "This file contains synthesized transients only. Run an external LCT or "
        "RSD solver to obtain the final volumetric reconstruction."
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to an experiment used_config.yaml."
    )
    parser.add_argument(
        "--model", help="Optional checkpoint override, for example a pruned model."
    )
    arguments = parser.parse_args()
    with open(arguments.config, "r", encoding="utf-8") as stream:
        configuration = EasyDict(yaml.safe_load(stream))
    if arguments.model:
        configuration.model_path = arguments.model
    generate_resampled_prediction(configuration)
