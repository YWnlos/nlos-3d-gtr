"""Optimize a 3D Gaussian scene against measured NLOS transients."""

import argparse
import atexit
import datetime
import io
import os
import shutil
import sys
import warnings

import scipy.io
import torch
import yaml
from easydict import EasyDict as EasyDict

from nlos_3dgs import NLOS_3DGS
from nlos_data import build_dataloaders
from nlos_train import train_nlos_model
from save_load_model import save_model_checkpoint
from utils import (
    maybe_compile,
    set_random_seed,
    visualize_multiple_results,
    visualize_results,
    visualize_test_loss,
    visualize_training,
)


class TeeLogger:
    """Mirror console output to one or more log streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        for stream in self.streams:
            if hasattr(stream, "fileno"):
                try:
                    return stream.fileno()
                except (AttributeError, io.UnsupportedOperation):
                    continue
        raise io.UnsupportedOperation("fileno is not available")

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def _build_model(config, device):
    model = NLOS_3DGS(device=device).to(device)
    model.initialize(
        num_gaussians=config.num_gaussians_init,
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
    return maybe_compile(model)


def main(config):
    """Run optimization, diagnostics, and checkpoint export."""
    if not torch.cuda.is_available():
        raise RuntimeError("3D-GTR requires a CUDA-capable GPU and Triton.")

    device = torch.device(config.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    if config.get("use_seed", False):
        set_random_seed(config.get("seed", 0))

    model = _build_model(config, device)
    train_loader, validation_loader, full_dataset = build_dataloaders(config)
    config.use_normals = full_dataset.has_normals
    active_batch_size = train_loader.batch_size
    print(
        f"Dataset: {len(full_dataset)} transients; normals={config.use_normals}; "
        f"batch_size={active_batch_size}."
    )

    model.ensure_timebin_buffer(config.t_range, device=device, dtype=torch.float32)
    model.ensure_aux_buffers(
        active_batch_size,
        config.num_gaussians_init,
        device=device,
        dtype=torch.float32,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    initial_figure = os.path.join(config.output_dir, "initial_transient_fit.png")
    sample_info = visualize_results(
        full_dataset,
        model,
        config,
        save_path=initial_figure,
        train_dataset=train_loader.dataset,
        test_dataset=validation_loader.dataset,
    )

    train_mse, train_total, validation_mse, epoch_times = train_nlos_model(
        model=model,
        train_loader=train_loader,
        test_loader=validation_loader,
        config=config,
    )

    final_figure = os.path.join(config.output_dir, "final_transient_fit.png")
    visualize_results(
        full_dataset,
        model,
        config,
        save_path=final_figure,
        train_dataset=train_loader.dataset,
        test_dataset=validation_loader.dataset,
        sample_info=sample_info,
    )
    visualize_multiple_results(
        train_loader.dataset, model, config, config.output_dir, 5, "train"
    )
    visualize_multiple_results(
        validation_loader.dataset, model, config, config.output_dir, 3, "validation"
    )
    visualize_training(train_mse, train_total, config.output_dir)
    visualize_test_loss(validation_mse, config.output_dir)

    scipy.io.savemat(
        os.path.join(config.output_dir, "train_loss_history.mat"),
        {
            "train_mse_history": train_mse,
            "epochs": list(range(len(train_mse))),
            "epoch_times": epoch_times,
        },
    )
    scipy.io.savemat(
        os.path.join(config.output_dir, "validation_loss_history.mat"),
        {
            "validation_mse_history": validation_mse,
            "epochs": list(range(len(validation_mse))),
            "epoch_times": epoch_times,
        },
    )

    checkpoint_path = save_model_checkpoint(
        model=model, config=config, filename="final_model.pth"
    )
    print(f"Final model saved to {checkpoint_path}")


def _configure_logging(output_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"log_{timestamp}.txt")
    log_stdout = open(log_path, "a", encoding="utf-8", buffering=1)
    log_stderr = open(log_path, "a", encoding="utf-8", buffering=1)
    atexit.register(log_stdout.close)
    atexit.register(log_stderr.close)
    sys.stdout = TeeLogger(sys.stdout, log_stdout)
    sys.stderr = TeeLogger(sys.stderr, log_stderr)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated",
        category=FutureWarning,
        module="torch.cuda",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="default_config.yaml", help="Path to a YAML configuration."
    )
    arguments = parser.parse_args()

    with open(arguments.config, "r", encoding="utf-8") as stream:
        config = EasyDict(yaml.safe_load(stream))

    os.makedirs(config.output_dir, exist_ok=True)
    shutil.copyfile(
        arguments.config, os.path.join(config.output_dir, "used_config.yaml")
    )
    _configure_logging(config.output_dir)
    main(config)
