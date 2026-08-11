"""Run the publication pruning pipeline on a trained 3D-GTR checkpoint."""

import argparse

import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial import cKDTree

from prune_gaussians import analyze_gaussians, prune_gaussians, save_pruned_model
from save_load_model import load_adaptive_model


def run_pipeline(
    input_model,
    output_model,
    device="cuda:0",
    saturation_threshold=2.0,
    low_albedo_threshold=0.005,
    edge_ratio=0.05,
    albedo_prune_percent=0.0,
    use_albedo_volume_product=False,
    knn_neighbors=3,
    isolated_percent=0.0,
    knn_2d=False,
):
    """Remove boundary, low-contribution, and optionally isolated primitives."""
    model = load_adaptive_model(input_model, device=device)
    checkpoint = torch.load(input_model, map_location="cpu", weights_only=False)
    config = EasyDict(checkpoint["config"])
    stats = analyze_gaussians(model, config)
    total = stats["N_total"]

    saturated = (
        (np.abs(stats["raw_centers"]) >= saturation_threshold).any(axis=1)
        | (np.abs(stats["raw_z"]) >= saturation_threshold)
    )
    low_albedo = stats["albedos"] < low_albedo_threshold

    centers = stats["centers"]
    lower = np.array([config.x_min, config.y_min, config.z_min])
    upper = np.array([config.x_max, config.y_max, config.z_max])
    margin = edge_ratio * (upper - lower)
    near_boundary = ((centers - lower) < margin).any(axis=1) | (
        (upper - centers) < margin
    ).any(axis=1)

    keep_mask = ~(saturated | low_albedo | near_boundary)
    print(f"Initial primitives: {total}")
    print(f"Saturated-position removals: {saturated.sum()}")
    print(f"Low-albedo removals: {low_albedo.sum()}")
    print(f"ROI-boundary removals: {near_boundary.sum()}")

    if albedo_prune_percent > 0:
        if not 0 < albedo_prune_percent < 1:
            raise ValueError("albedo_prune_percent must lie in (0, 1).")
        remaining = np.flatnonzero(keep_mask)
        scores = stats["albedos"][remaining]
        if use_albedo_volume_product:
            scores = scores * np.prod(stats["scales"][remaining], axis=1)
        count = int(len(remaining) * albedo_prune_percent)
        remove = remaining[np.argsort(scores)[:count]]
        keep_mask[remove] = False
        print(f"Lowest-contribution percentile removals: {len(remove)}")

    if isolated_percent > 0:
        if not 0 < isolated_percent < 100:
            raise ValueError("isolated_percent must lie in (0, 100).")
        remaining = np.flatnonzero(keep_mask)
        coordinates = centers[remaining, :2] if knn_2d else centers[remaining]
        if len(coordinates) > 1:
            neighbor_count = min(knn_neighbors + 1, len(coordinates))
            distances, _ = cKDTree(coordinates).query(
                coordinates, k=neighbor_count
            )
            if distances.ndim == 1:
                distances = distances[:, None]
            isolation_score = distances[:, -1]
            threshold = np.percentile(isolation_score, 100 - isolated_percent)
            remove = remaining[isolation_score >= threshold]
            keep_mask[remove] = False
            print(f"Isolated KNN removals: {len(remove)}")

    if not keep_mask.any():
        raise ValueError("The selected pruning parameters remove every primitive.")
    original_count = prune_gaussians(model, keep_mask, config)
    save_pruned_model(model, config, output_model, original_count)
    return keep_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--saturation-threshold", type=float, default=2.0)
    parser.add_argument("--low-albedo-threshold", type=float, default=0.005)
    parser.add_argument("--edge-ratio", type=float, default=0.05)
    parser.add_argument("--albedo-prune-percent", type=float, default=0.0)
    parser.add_argument("--use-albedo-volume-product", action="store_true")
    parser.add_argument("--knn-neighbors", type=int, default=3)
    parser.add_argument("--isolated-percent", type=float, default=0.0)
    parser.add_argument("--knn-2d", action="store_true")
    arguments = parser.parse_args()
    run_pipeline(
        arguments.input_model,
        arguments.output_model,
        device=arguments.device,
        saturation_threshold=arguments.saturation_threshold,
        low_albedo_threshold=arguments.low_albedo_threshold,
        edge_ratio=arguments.edge_ratio,
        albedo_prune_percent=arguments.albedo_prune_percent,
        use_albedo_volume_product=arguments.use_albedo_volume_product,
        knn_neighbors=arguments.knn_neighbors,
        isolated_percent=arguments.isolated_percent,
        knn_2d=arguments.knn_2d,
    )


if __name__ == "__main__":
    main()
