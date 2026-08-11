"""Analyze and prune negligible 3D Gaussian primitives."""

import argparse

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

from save_load_model import load_adaptive_model


def analyze_gaussians(model, config):
    """Return physical and raw Gaussian attributes used by pruning rules."""
    del config
    with torch.no_grad():
        centers = model.get_gs_centers().cpu().numpy()
        scales = model.compute_scale(model.raw_scaling).cpu().numpy()
        albedos = model.compute_albedo(model.albedo).squeeze(-1).cpu().numpy()
        raw_xy = model.xy_centers_raw.cpu().numpy()
        raw_z = model.z_centers_raw.squeeze(-1).cpu().numpy()
    return {
        "centers": centers,
        "scales": scales,
        "albedos": albedos,
        "raw_centers": raw_xy,
        "raw_z": raw_z,
        "N_total": len(albedos),
    }


def define_pruning_criteria(
    stats,
    config,
    *,
    saturation_threshold=2.0,
    albedo_threshold=0.005,
    remove_unoptimized=True,
):
    """Build a Boolean mask that keeps physically relevant primitives."""
    severe_saturation = (
        (np.abs(stats["raw_centers"]) >= saturation_threshold).any(axis=1)
        | (np.abs(stats["raw_z"]) >= saturation_threshold)
    )
    low_albedo = stats["albedos"] < albedo_threshold
    keep_mask = ~(severe_saturation | low_albedo)

    unoptimized = np.zeros_like(keep_mask)
    if remove_unoptimized:
        initial_albedo = float(getattr(config, "albedo_init", 0.1))
        unoptimized = np.isclose(stats["albedos"], initial_albedo, atol=1e-6)
        keep_mask &= ~unoptimized

    total = len(keep_mask)
    print(f"Gaussian primitives: {total}")
    print(f"Removed at saturated position parameters: {severe_saturation.sum()}")
    print(f"Removed below albedo {albedo_threshold}: {low_albedo.sum()}")
    print(f"Removed at initial albedo: {unoptimized.sum()}")
    print(f"Retained: {keep_mask.sum()} ({100.0 * keep_mask.mean():.1f}%)")
    return keep_mask


def prune_gaussians(model, keep_mask, config=None):
    """Apply a Boolean keep mask to every learnable Gaussian attribute."""
    del config
    keep_mask = torch.as_tensor(keep_mask, dtype=torch.bool, device=model.albedo.device)
    if keep_mask.ndim != 1 or keep_mask.numel() != model.albedo.shape[0]:
        raise ValueError("keep_mask must contain one Boolean value per Gaussian.")
    if not keep_mask.any():
        raise ValueError("Pruning would remove every Gaussian primitive.")

    original_count = int(model.albedo.shape[0])
    for name in ("xy_centers_raw", "z_centers_raw", "raw_scaling", "quats", "albedo"):
        value = getattr(model, name).detach()[keep_mask].clone()
        setattr(model, name, nn.Parameter(value))

    new_count = int(keep_mask.sum())
    model.xyz_gradient_accum = torch.zeros(new_count, device=model.albedo.device)
    model.denom = torch.zeros(new_count, device=model.albedo.device)
    model.num_gaussians = new_count
    if hasattr(model, "ones_BN"):
        model.ones_BN = torch.ones(
            model.ones_BN.shape[0],
            new_count,
            device=model.ones_BN.device,
            dtype=model.ones_BN.dtype,
        )
    print(f"Retained {new_count}/{original_count} Gaussian primitives.")
    return original_count


def save_pruned_model(model, config, output_path, original_params=None):
    """Save a compact checkpoint for a pruned model."""
    if isinstance(config, dict):
        config_dict = dict(config)
    else:
        config_dict = dict(vars(config))
    config_dict["num_gaussians_init"] = int(model.num_gaussians)

    if isinstance(original_params, int):
        original_count = original_params
    elif isinstance(original_params, dict) and "xy_centers_raw" in original_params:
        original_count = len(original_params["xy_centers_raw"])
    else:
        original_count = None

    torch.save(
        {
            "num_gaussians": int(model.num_gaussians),
            "config": config_dict,
            "state_dict": model.state_dict(),
            "pruning_info": {
                "original_num_gaussians": original_count,
                "pruned_num_gaussians": int(model.num_gaussians),
            },
        },
        output_path,
    )
    print(f"Saved pruned model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--saturation-threshold", type=float, default=2.0)
    parser.add_argument("--albedo-threshold", type=float, default=0.005)
    parser.add_argument("--keep-unoptimized", action="store_true")
    arguments = parser.parse_args()

    checkpoint = torch.load(
        arguments.input_model, map_location=arguments.device, weights_only=False
    )
    config = EasyDict(checkpoint["config"])
    model = load_adaptive_model(arguments.input_model, device=arguments.device)
    stats = analyze_gaussians(model, config)
    keep_mask = define_pruning_criteria(
        stats,
        config,
        saturation_threshold=arguments.saturation_threshold,
        albedo_threshold=arguments.albedo_threshold,
        remove_unoptimized=not arguments.keep_unoptimized,
    )
    original_count = prune_gaussians(model, keep_mask, config)
    save_pruned_model(model, config, arguments.output_model, original_count)


if __name__ == "__main__":
    main()
