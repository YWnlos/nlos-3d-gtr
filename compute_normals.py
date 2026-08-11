"""Load 3D points from a .mat file, estimate outward-facing normals, and visualize."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.io import loadmat, savemat


def load_space_points(mat_path: Path, key: str) -> np.ndarray:
    """Load a (3, N) point array from a .mat file and return as (N, 3)."""
    data = loadmat(mat_path)
    if key not in data:
        available = ", ".join(sorted(k for k in data.keys() if not k.startswith("__")))
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available: {available}")

    arr = np.asarray(data[key])
    if arr.ndim != 2 or 3 not in arr.shape:
        raise ValueError(f"Expected array with one dimension of size 3; got shape {arr.shape}")

    # Normalize to shape (N, 3).
    if arr.shape[0] == 3:
        points = arr.T
    elif arr.shape[1] == 3:
        points = arr
    else:
        raise ValueError(f"Cannot reshape array of shape {arr.shape} to (N, 3)")

    return points.astype(np.float64, copy=False)


def estimate_outward_normals(
    points: np.ndarray,
    k_neighbors: int = 30,
    use_consistent: bool = False,
    voxel_size: Optional[float] = None,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Estimate normals; optionally downsample and optionally run the expensive
    orient_normals_consistent_tangent_plane step (can crash on huge/degenerate clouds).
    """
    # Remove invalid samples and optionally reduce point-cloud density.
    pts = np.asarray(points, dtype=np.float32)
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if voxel_size:
        pcd_ds = o3d.geometry.PointCloud()
        pcd_ds.points = o3d.utility.Vector3dVector(pts)
        pcd_ds = pcd_ds.voxel_down_sample(voxel_size)
        pts = np.asarray(pcd_ds.points, dtype=np.float32)

    if len(pts) < 3:
        raise ValueError("Not enough valid points after filtering/downsample.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    k_use = min(max(10, k_neighbors), len(pts) - 1)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_use))

    if use_consistent:
        k_consistent = min(max(10, k_use // 2), len(pts) - 1)
        try:
            pcd.orient_normals_consistent_tangent_plane(k=k_consistent)
        except Exception as e:
            print("orient_normals_consistent_tangent_plane failed; continue without it:", e)

    # Orient normals away from the centroid of a roughly convex surface.
    centroid = np.mean(pts, axis=0)
    normals = np.asarray(pcd.normals)
    vectors_to_points = pts - centroid
    signs = np.einsum("ij,ij->i", normals, vectors_to_points)
    normals[signs < 0] *= -1.0

    # Enforce the acquisition convention that normals point toward +z.
    flip_mask = normals[:, 2] < 0
    if np.any(flip_mask):
        normals[flip_mask] *= -1.0

    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd, normals


def visualize_point_cloud(pcd: o3d.geometry.PointCloud) -> None:
    """Display the point cloud with normals in Open3D."""
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud with Normals",
        point_show_normal=True,
        width=960,
        height=720,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input MATLAB point-cloud file.")
    parser.add_argument("--key", default="camera_points", help="Point variable name.")
    parser.add_argument("--output", type=Path, help="Output MATLAB file.")
    parser.add_argument("--k-neighbors", type=int, default=30)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--consistent", action="store_true")
    parser.add_argument("--view", action="store_true")
    arguments = parser.parse_args()

    mat_path = arguments.input
    output_path = arguments.output or mat_path.with_name(
        mat_path.stem + "_with_normals.mat"
    )
    points = load_space_points(mat_path, arguments.key)
    pcd, normals = estimate_outward_normals(
        points,
        k_neighbors=arguments.k_neighbors,
        use_consistent=arguments.consistent,
        voxel_size=arguments.voxel_size,
    )

    print(f"Loaded {len(points)} points from '{mat_path}' (key='{arguments.key}').")
    print(f"Estimated normals with k={arguments.k_neighbors}.")
    print(f"Centroid: {np.mean(points, axis=0)}")
    print("First 5 normals:\n", normals[:5])

    # Preserve original variables and add the estimated normals.
    original_data = loadmat(mat_path)
    save_dict = {k: v for k, v in original_data.items() if not k.startswith("__")}

    arr = np.asarray(original_data.get(arguments.key, points))
    normals_to_save = normals
    if arr.ndim == 2 and 3 in arr.shape:
        if arr.shape[0] == 3 and normals.shape[0] == arr.shape[1]:
            normals_to_save = normals.T
        elif arr.shape[1] == 3 and normals.shape[0] == arr.shape[0]:
            normals_to_save = normals

    save_dict["normals"] = normals_to_save
    savemat(output_path, save_dict)
    print(f"Saved normals and original variables to '{output_path}'.")

    if arguments.view:
        visualize_point_cloud(pcd)


if __name__ == "__main__":
    main()
