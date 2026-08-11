"""Generate an annotated example configuration for 3D-GTR."""

import yaml


default_config = {
    # Input and output paths. Run scripts from the code&data directory.
    "mat_path": "./real_data/real_trans_dataset_Swall3_O_32ps_manualz0_center.mat",
    "output_dir": "./real_output/Swall3_O_example",

    # Data preparation.
    "scale_factor": 0.001,  # Multiplicative scale applied to measured transients.
    "train_split": 0.99,  # Remaining samples form the validation split.
    "batch_size": 1,  # Used when the MATLAB file does not contain normals.
    "norm_cycle_on_batch": 1,  # Used when both laser and camera normals exist.
    "num_workers": 8,

    # Runtime and reproducibility.
    "device": "cuda:0",
    "use_seed": True,
    "seed": 0,

    # Gaussian initialization and reconstruction volume, in meters.
    "num_gaussians_init": 3000,
    "x_min": -0.5,
    "x_max": 0.5,
    "y_min": -0.5,
    "y_max": 0.5,
    "z_min": 0.5,
    "z_max": 0.9,
    "scale_min": 0.005,
    "scale_max": 0.015,
    "scale_init": 0.01,
    "albedo_init": 0.1,
    "activation_mode": "tanh",  # Bounds Gaussian centers to the volume.
    "scale_activation_mode": "tanh",  # Bounds principal-axis scales.
    "albedo_activation_mode": "tanh",  # Bounds albedo to [0, 1].
    "anisotropy_threshold": 15.0,  # Maximum ratio of largest to smallest scale.

    # Forward-model assumptions.
    "isplanar": True,  # Use +z relay normals when measured normals are absent.
    "isconfocal": False,  # Require identical illumination and detection points.
    "isretroreflective": False,  # Valid only for confocal measurements.
    "render_magnification": 5e4,  # Global gain g in the paper.
    "dT": 32e-12,  # Measurement bin width in seconds.
    "t_range": [0, 350],  # Half-open range [start, end) of bins used for fitting.

    # Optimization.
    "num_epochs": 20,
    "save_interval": 20,
    "lr_schedule": "multistep",  # Either "multistep" or "none".
    "warmup_ratio": 0.0,  # Fraction of all optimizer steps used for warmup.
    "milestones": [8, 16],  # Epochs at which the learning rate is multiplied by gamma.
    "gamma": 0.8,
    "weight_decay": 1e-9,

    # Learning rates for inputs without measured relay normals.
    "lr_xy": 5e-3,
    "lr_z": 3e-3,
    "lr_scale": 1e-2,
    "lr_quat": 1e-2,
    "lr_albedo": 5e-2,

    # Learning rates for inputs with measured illumination and detection normals.
    "lr_xy_post_norm": 5e-4,
    "lr_z_post_norm": 1e-4,
    "lr_scale_post_norm": 8e-4,
    "lr_quat_post_norm": 8e-4,
    "lr_albedo_post_norm": 3e-5,

    # Dense transient synthesis on a virtual confocal relay surface.
    "output_wallsize": 1.0,  # Side length in meters for an automatic planar wall.
    "output_res": 128,  # Samples per side.
    "output_t_range": [0, 512],
    "output_dT": 32e-12,
    "predict_batch_size": 256,
    "use_auto_wall": True,
    "virtual_wall_path": "./virtual_wall/example_virtual_wall.mat",
}


if __name__ == "__main__":
    config_path = "./default_config.yaml"
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(default_config, stream, sort_keys=False)
    print(f"Wrote {config_path}")
