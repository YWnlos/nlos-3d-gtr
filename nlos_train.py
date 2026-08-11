"""Training loop for 3D Gaussian Transient Rendering."""

import math
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from save_load_model import save_model_checkpoint


def _learning_rates(config):
    """Return learning rates for datasets with or without relay normals."""
    suffix = "_post_norm" if getattr(config, "use_normals", False) else ""
    return {
        "xy": getattr(config, f"lr_xy{suffix}"),
        "z": getattr(config, f"lr_z{suffix}"),
        "scale": getattr(config, f"lr_scale{suffix}"),
        "quat": getattr(config, f"lr_quat{suffix}"),
        "albedo": getattr(config, f"lr_albedo{suffix}"),
    }


def configure_optimizer(model, config):
    """Create the AdamW optimizer with one rate per Gaussian attribute."""
    rates = _learning_rates(config)
    return torch.optim.AdamW(
        [
            {"params": model.xy_centers_raw, "lr": rates["xy"]},
            {"params": model.z_centers_raw, "lr": rates["z"]},
            {"params": model.raw_scaling, "lr": rates["scale"]},
            {"params": model.quats, "lr": rates["quat"]},
            {"params": model.albedo, "lr": rates["albedo"]},
        ],
        fused=torch.cuda.is_available(),
        weight_decay=getattr(config, "weight_decay", 0.0),
    )


def build_multistep_scheduler(
    optimizer,
    *,
    num_epochs,
    steps_per_epoch,
    warmup_ratio=0.0,
    milestones=(),
    gamma=0.3,
):
    """Build a step-wise scheduler with optional warmup and epoch milestones."""
    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = int(warmup_ratio * total_steps)
    milestone_steps = sorted(int(epoch * steps_per_epoch) for epoch in milestones)

    def lr_scale(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        drops = sum(step >= milestone for milestone in milestone_steps)
        return gamma**drops

    return LambdaLR(optimizer, lr_lambda=lr_scale, last_epoch=-1)


def _validate_geometry(model, laser_points, camera_points):
    """Validate configuration assumptions that remain fixed during training."""
    if model.isconfocal and not torch.allclose(
        laser_points, camera_points, rtol=1e-7, atol=1e-7
    ):
        raise ValueError(
            "isconfocal=True requires identical laser_points and camera_points."
        )
    if model.isretroreflective and not model.isconfocal:
        raise ValueError("Retroreflective rendering requires a confocal configuration.")


def _move_batch(batch, device, use_normals):
    laser_points, camera_points, transients, laser_normals, camera_normals = batch
    laser_points = laser_points.to(device, non_blocking=True)
    camera_points = camera_points.to(device, non_blocking=True)
    transients = transients.to(device, non_blocking=True)
    if use_normals:
        laser_normals = laser_normals.to(device, non_blocking=True)
        camera_normals = camera_normals.to(device, non_blocking=True)
    else:
        laser_normals = None
        camera_normals = None
    return laser_points, camera_points, transients, laser_normals, camera_normals


def train_nlos_model(model, train_loader, test_loader, config):
    """Optimize the 3D Gaussian primitives against measured transients."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(config.device)
    use_normals = bool(getattr(config, "use_normals", False))
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, "tensorboard"))
    criterion = nn.MSELoss()
    optimizer = configure_optimizer(model, config)

    schedule = getattr(config, "lr_schedule", "multistep")
    if schedule == "multistep":
        scheduler = build_multistep_scheduler(
            optimizer,
            num_epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            warmup_ratio=getattr(config, "warmup_ratio", 0.0),
            milestones=getattr(config, "milestones", ()),
            gamma=getattr(config, "gamma", 0.3),
        )
        print(
            "Using the MultiStep learning-rate schedule "
            f"(warmup_ratio={getattr(config, 'warmup_ratio', 0.0)})."
        )
    elif schedule == "none":
        scheduler = None
        print("Learning-rate scheduling is disabled.")
    else:
        raise ValueError(
            f"Unsupported lr_schedule '{schedule}'. Use 'multistep' or 'none'."
        )

    sample = next(iter(train_loader))
    sample_laser, sample_camera, sample_trans, _, _ = _move_batch(
        sample, device, use_normals
    )
    _validate_geometry(model, sample_laser, sample_camera)
    model.ensure_timebin_buffer(config.t_range, device=device, dtype=torch.float32)
    model.ensure_aux_buffers(
        sample_camera.shape[0], model.xy_centers_raw.shape[0], device, torch.float32
    )

    train_mse_history = []
    test_loss_history = []
    epoch_times = []
    max_gpu_memory = 0.0
    start_time = time.time()

    with tqdm(
        total=config.num_epochs, desc="Training Progress", unit="epoch"
    ) as progress:
        for epoch in range(config.num_epochs):
            model.train()
            epoch_loss = 0.0
            last_gradients = {}

            for batch_index, batch in enumerate(train_loader):
                laser_points, camera_points, transients, laser_normals, camera_normals = (
                    _move_batch(batch, device, use_normals)
                )
                optimizer.zero_grad(set_to_none=True)
                predictions = model(
                    camera_points,
                    config.dT,
                    config.t_range,
                    laser_points=laser_points,
                    render_magnification=config.render_magnification,
                    laser_normals=laser_normals,
                    camera_normals=camera_normals,
                )
                loss = criterion(predictions, transients.squeeze(1))
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if batch_index == len(train_loader) - 1:
                    for group_index, group in enumerate(optimizer.param_groups):
                        parameter = group["params"][0]
                        if parameter.grad is not None:
                            last_gradients[group_index] = parameter.grad.detach().norm().item()

                epoch_loss += loss.item() * camera_points.shape[0]

            average_train_loss = epoch_loss / len(train_loader.dataset)
            train_mse_history.append(average_train_loss)
            writer.add_scalar("Loss/train_mse", average_train_loss, epoch)

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    laser_points, camera_points, transients, laser_normals, camera_normals = (
                        _move_batch(batch, device, use_normals)
                    )
                    predictions = model(
                        camera_points,
                        config.dT,
                        config.t_range,
                        laser_points=laser_points,
                        render_magnification=config.render_magnification,
                        laser_normals=laser_normals,
                        camera_normals=camera_normals,
                    )
                    test_loss += criterion(
                        predictions, transients.squeeze(1)
                    ).item() * camera_points.shape[0]

            average_test_loss = test_loss / len(test_loader.dataset)
            test_loss_history.append(average_test_loss)
            writer.add_scalar("Loss/test", average_test_loss, epoch)

            group_names = ["xy", "z", "scale", "quat", "albedo"]
            for group_index, group in enumerate(optimizer.param_groups):
                name = group_names[group_index]
                writer.add_scalar(f"LR/{name}", group["lr"], epoch)
                if group_index in last_gradients:
                    writer.add_scalar(
                        f"Grad/{name}", last_gradients[group_index], epoch
                    )

            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated(device) / 1024**3
                max_gpu_memory = max(max_gpu_memory, current_memory)
                writer.add_scalar("Memory/allocated_gb", current_memory, epoch)
            else:
                current_memory = 0.0

            progress.set_postfix(
                {
                    "TrainMSE": f"{average_train_loss:.4e}",
                    "TestMSE": f"{average_test_loss:.4e}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "Mem(GB)": f"{current_memory:.2f}",
                }
            )
            progress.update(1)
            epoch_times.append(time.time() - start_time)

            if (epoch + 1) % config.save_interval == 0:
                save_model_checkpoint(model=model, config=config, epoch=epoch + 1)

    total_time = time.time() - start_time
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_pytorch_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        peak_pytorch_memory = 0.0

    print("\n=== Training Summary ===")
    print(
        f"Total Time: {total_time // 3600:.0f}h "
        f"{total_time % 3600 // 60:.0f}m {total_time % 60:.2f}s"
    )
    print(f"Peak PyTorch GPU Memory: {peak_pytorch_memory:.2f} GB")
    print(f"Peak Allocated GPU Memory: {max_gpu_memory:.2f} GB")
    print(f"Final Train MSE: {train_mse_history[-1]:.4e}")
    print(f"Final Test MSE: {test_loss_history[-1]:.4e}")

    writer.add_scalar("Summary/FinalTrainMSE", train_mse_history[-1], config.num_epochs)
    writer.add_scalar("Summary/FinalTestMSE", test_loss_history[-1], config.num_epochs)
    writer.add_scalar("Summary/MaxGPUMemoryGB", max_gpu_memory, config.num_epochs)
    writer.add_scalar("Summary/TotalTimeSeconds", total_time, config.num_epochs)
    writer.close()

    # Preserve the historical four-value return signature. No extra regularizer is used,
    # so total training loss is identical to MSE.
    return (
        train_mse_history,
        list(train_mse_history),
        test_loss_history,
        epoch_times,
    )
