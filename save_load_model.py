"""Checkpoint serialization for 3D-GTR models."""

import os

import torch

from nlos_3dgs import NLOS_3DGS


def clean_state_dict(state_dict):
    """Remove wrappers added by torch.compile or distributed training."""
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    for prefix in ("_orig_mod.", "module."):
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix) :] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
            }
    return state_dict


def save_model_checkpoint(model, config, epoch=None, filename=None):
    """Save an unwrapped model state together with its full configuration."""
    os.makedirs(config.output_dir, exist_ok=True)
    model_to_save = getattr(model, "_orig_mod", model)
    model_to_save = getattr(model_to_save, "module", model_to_save)

    if filename is None:
        filename = f"model_epoch{epoch}.pth" if epoch is not None else "final_model.pth"
    save_path = os.path.join(config.output_dir, filename)
    config_to_save = dict(config)

    torch.save(
        {
            "num_gaussians": model_to_save.xy_centers_raw.shape[0],
            "state_dict": model_to_save.state_dict(),
            "epoch": epoch,
            "config": config_to_save,
        },
        save_path,
    )
    return save_path


def load_adaptive_model(checkpoint_path, device="cuda", fallback_config_path=None):
    """Reconstruct a model from a checkpoint and its embedded configuration."""
    import yaml
    from easydict import EasyDict

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("The checkpoint does not contain a 'config' entry.")
    config = EasyDict(config) if isinstance(config, dict) else config

    if fallback_config_path and os.path.exists(fallback_config_path):
        with open(fallback_config_path, "r", encoding="utf-8") as stream:
            fallback_config = EasyDict(yaml.safe_load(stream))
        for key, value in fallback_config.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        print(f"Loaded missing configuration values from {fallback_config_path}.")

    required_keys = (
        "num_gaussians_init",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "scale_min",
        "scale_max",
        "scale_init",
        "albedo_init",
        "activation_mode",
        "isplanar",
        "isconfocal",
        "isretroreflective",
    )
    missing = [key for key in required_keys if not hasattr(config, key)]
    if missing:
        raise KeyError(f"Checkpoint configuration is missing: {missing}")

    state_dict = clean_state_dict(checkpoint["state_dict"])
    num_gaussians = int(
        checkpoint.get(
            "num_gaussians", state_dict["xy_centers_raw"].shape[0]
        )
    )
    model = NLOS_3DGS(device=device)
    model.initialize(
        num_gaussians=num_gaussians,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        z_min=config.z_min,
        z_max=config.z_max,
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        scale_init=config.scale_init,
        albedo_init=config.albedo_init,
        isplanar=config.isplanar,
        isconfocal=config.isconfocal,
        isretroreflective=config.isretroreflective,
        activation_mode=config.activation_mode,
        scale_activation_mode=getattr(config, "scale_activation_mode", "relu"),
        albedo_activation_mode=getattr(config, "albedo_activation_mode", "relu"),
        anisotropy_threshold=getattr(config, "anisotropy_threshold", 0.0),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model
