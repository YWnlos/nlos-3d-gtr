"""Fit a relay plane and generate a uniform virtual-wall sampling grid."""

import argparse

import numpy as np
from scipy.io import loadmat, savemat


def _as_points(array):
    """Normalize an array with one dimension of length three to shape (N,3)."""
    points = np.asarray(array, dtype=np.float64)
    if points.ndim != 2 or 3 not in points.shape:
        raise ValueError(f"Expected point array with shape (N,3) or (3,N); got {points.shape}")
    return points.T if points.shape[0] == 3 else points


def fit_plane_pca(points):
    """Fit n dot x + d = 0 by PCA and return fit diagnostics."""
    points = _as_points(points)
    centroid = points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(points - centroid, full_matrices=False)
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    signed_errors = points @ normal + offset
    return {
        "normal": normal,
        "d": offset,
        "center": centroid,
        "errors": signed_errors,
        "mse": float(np.mean(signed_errors**2)),
        "max_error": float(np.max(np.abs(signed_errors))),
    }


def build_plane_frame(normal):
    """Construct a right-handed local frame whose z axis is the plane normal."""
    z_axis = np.asarray(normal, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    reference = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(reference, z_axis)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def generate_snake_grid(side_length, resolution):
    """Generate a square z=0 grid in alternating row order."""
    axis = np.linspace(-side_length / 2, side_length / 2, resolution)
    rows = []
    for row_index, x_value in enumerate(axis[::-1]):
        y_values = axis[::-1] if row_index % 2 == 0 else axis
        rows.extend((x_value, y_value, 0.0) for y_value in y_values)
    return np.asarray(rows, dtype=np.float64)


def generate_virtual_wall(input_path, output_path, key, side_length, resolution):
    """Fit a plane to measured relay points and save a uniform virtual wall."""
    data = loadmat(input_path)
    if key not in data:
        available = sorted(name for name in data if not name.startswith("__"))
        raise KeyError(f"'{key}' not found in {input_path}. Available variables: {available}")

    measured_points = _as_points(data[key])
    fit = fit_plane_pca(measured_points)
    rotation = build_plane_frame(fit["normal"])
    projected_center = fit["center"] - (
        np.dot(fit["normal"], fit["center"]) + fit["d"]
    ) * fit["normal"]
    local_grid = generate_snake_grid(side_length, resolution)
    world_grid = local_grid @ rotation.T + projected_center
    wall_normals = np.tile(fit["normal"], (len(world_grid), 1))

    savemat(
        output_path,
        {
            "wall_size": float(side_length),
            "resolution": int(resolution),
            "virtual_cam_pts": world_grid,
            "wall_normals": wall_normals,
            "fitted_normal": fit["normal"],
            "fitted_center": fit["center"],
            "fit_mse": fit["mse"],
            "fit_max_error": fit["max_error"],
        },
    )
    print(f"Saved {len(world_grid)} virtual-wall samples to {output_path}")
    print(f"Plane-fit MSE: {fit['mse']:.6e} m^2")
    print(f"Maximum absolute fit error: {fit['max_error']:.6e} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input MATLAB dataset.")
    parser.add_argument("--output", required=True, help="Output virtual-wall MATLAB file.")
    parser.add_argument("--key", default="camera_points", help="Relay-point variable name.")
    parser.add_argument("--wall-size", type=float, required=True, help="Wall side length in meters.")
    parser.add_argument("--resolution", type=int, default=128, help="Samples per side.")
    arguments = parser.parse_args()
    generate_virtual_wall(
        arguments.input,
        arguments.output,
        arguments.key,
        arguments.wall_size,
        arguments.resolution,
    )


if __name__ == "__main__":
    main()
