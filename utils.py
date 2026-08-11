"""Reproducibility, visualization, and compilation helpers for 3D-GTR."""

import os
import random
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F


def set_random_seed(seed):
    """Seed Python, NumPy, and PyTorch and request deterministic CUDA kernels."""
    print(f"Using random seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _choose_indices(length, count):
    if length <= 0:
        return []
    count = min(length, count)
    return np.random.choice(length, count, replace=False).tolist()


def _render_sample(dataset, index, model, config):
    laser, camera, transient, laser_normal, camera_normal = dataset[index]
    device = next(model.parameters()).device
    laser = laser.unsqueeze(0).to(device)
    camera = camera.unsqueeze(0).to(device)
    transient = transient.squeeze().cpu()
    use_normals = bool(getattr(config, "use_normals", False))
    laser_normal = laser_normal.unsqueeze(0).to(device) if use_normals else None
    camera_normal = camera_normal.unsqueeze(0).to(device) if use_normals else None
    with torch.no_grad():
        prediction = model(
            camera,
            config.dT,
            config.t_range,
            laser_points=laser,
            render_magnification=config.render_magnification,
            laser_normals=laser_normal,
            camera_normals=camera_normal,
        ).squeeze().cpu()
    mse = F.mse_loss(prediction, transient).item()
    return prediction, transient, camera.squeeze(0).cpu(), mse


def visualize_results(
    dataset,
    model,
    config,
    save_path="result.png",
    sample_indices=None,
    train_dataset=None,
    test_dataset=None,
    sample_info=None,
):
    """Plot fixed representative transient fits and save their numerical values."""
    model.eval()
    if sample_info is None:
        if sample_indices is not None:
            sample_info = [(int(index), "full") for index in sample_indices]
        elif train_dataset is not None and test_dataset is not None:
            sample_info = [
                *[(index, "train") for index in _choose_indices(len(train_dataset), 4)],
                *[(index, "validation") for index in _choose_indices(len(test_dataset), 2)],
            ]
            random.shuffle(sample_info)
        else:
            sample_info = [
                (index, "full") for index in _choose_indices(len(dataset), 3)
            ]

    if not sample_info:
        raise ValueError("No samples are available for visualization.")

    figure, axes = plt.subplots(
        len(sample_info), 1, figsize=(12, 3.5 * len(sample_info)), squeeze=False
    )
    saved = {
        "indices": [],
        "splits": [],
        "pred": [],
        "true_vals": [],
        "camera_points": [],
        "mse": [],
    }

    for row, (index, split) in enumerate(sample_info):
        source_dataset = {
            "train": train_dataset,
            "validation": test_dataset,
            "full": dataset,
        }[split]
        prediction, target, camera, mse = _render_sample(
            source_dataset, index, model, config
        )
        axis = axes[row, 0]
        axis.plot(target.numpy(), label="Measured", linewidth=1.8)
        axis.plot(prediction.numpy(), "--", label="Rendered", linewidth=1.4)
        axis.set_title(
            f"{split.capitalize()} sample {index} | "
            f"point={np.round(camera.numpy(), 3).tolist()} | MSE={mse:.2e}"
        )
        axis.set_xlabel("Time bin")
        axis.set_ylabel("Intensity")
        axis.legend()

        saved["indices"].append(index)
        saved["splits"].append(split)
        saved["pred"].append(prediction.numpy())
        saved["true_vals"].append(target.numpy())
        saved["camera_points"].append(camera.numpy())
        saved["mse"].append(mse)

    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    mat_path = os.path.splitext(save_path)[0] + ".mat"
    scipy.io.savemat(
        mat_path,
        {
            "indices": np.asarray(saved["indices"]),
            "splits": np.asarray(saved["splits"], dtype=object),
            "pred": np.asarray(saved["pred"]),
            "true_vals": np.asarray(saved["true_vals"]),
            "camera_points": np.asarray(saved["camera_points"]),
            "mse": np.asarray(saved["mse"]),
        },
    )
    print(f"Saved transient-fit visualization to {save_path}")
    return sample_info


def visualize_multiple_results(
    dataset, model, config, save_dir="results", num_groups=3, dataset_type="train"
):
    """Save several three-sample transient-fit panels for qualitative inspection."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    samples_per_group = min(3, len(dataset))
    if samples_per_group == 0:
        return

    for group_index in range(num_groups):
        indices = _choose_indices(len(dataset), samples_per_group)
        figure, axes = plt.subplots(
            samples_per_group,
            1,
            figsize=(12, 3.5 * samples_per_group),
            squeeze=False,
        )
        for row, index in enumerate(indices):
            prediction, target, camera, mse = _render_sample(
                dataset, index, model, config
            )
            axis = axes[row, 0]
            axis.plot(target.numpy(), label="Measured", linewidth=1.8)
            axis.plot(prediction.numpy(), "--", label="Rendered", linewidth=1.4)
            axis.set_title(
                f"Sample {index} | point={np.round(camera.numpy(), 3).tolist()} "
                f"| MSE={mse:.2e}"
            )
            axis.set_xlabel("Time bin")
            axis.set_ylabel("Intensity")
            axis.legend()
        figure.tight_layout()
        path = os.path.join(save_dir, f"{dataset_type}_group{group_index + 1}.png")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)


def _plot_loss(history, title, label, path):
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.semilogy(history, linewidth=1.6, label=label)
    minimum = min(history)
    minimum_epoch = history.index(minimum)
    axis.scatter([minimum_epoch], [minimum], facecolors="none", edgecolors="red")
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.grid(True, which="both", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def visualize_training(train_mse_history, train_total_loss_history, output_dir):
    """Plot training MSE. The second argument is retained for API compatibility."""
    del train_total_loss_history
    _plot_loss(
        train_mse_history,
        "Training MSE",
        "Training MSE",
        os.path.join(output_dir, "train_mse_curve.png"),
    )


def visualize_test_loss(test_loss_history, save_dir):
    """Plot validation MSE across epochs."""
    _plot_loss(
        test_loss_history,
        "Validation MSE",
        "Validation MSE",
        os.path.join(save_dir, "validation_mse_curve.png"),
    )


def maybe_compile(model, mode="reduce-overhead"):
    """Compile a model when torch.compile is available, otherwise use eager mode."""
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile is unavailable; using eager execution.")
        return model
    try:
        return torch.compile(model, mode=mode)
    except Exception as error:
        warnings.warn(f"torch.compile failed; using eager execution: {error}")
        return model
