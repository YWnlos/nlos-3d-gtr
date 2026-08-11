"""Dataset loading and train/validation splitting for measured transients."""

import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class RealTransDataset(Dataset):
    """Load relay points, transient histograms, and optional surface normals."""

    def __init__(self, mat_path, scale_factor=1.0):
        data = scipy.io.loadmat(mat_path)
        required = {"camera_points", "trans_data"}
        missing = required.difference(data)
        if missing:
            raise KeyError(f"Missing required MATLAB variables: {sorted(missing)}")

        self.camera_points = torch.as_tensor(
            data["camera_points"], dtype=torch.float32
        )
        self.trans = (
            torch.as_tensor(data["trans_data"], dtype=torch.float32) * scale_factor
        )
        self.laser_points = torch.as_tensor(
            data.get("laser_points", data["camera_points"]), dtype=torch.float32
        )

        self.has_normals = "laser_normals" in data and "camera_normals" in data
        if self.has_normals:
            self.laser_normals = torch.as_tensor(
                data["laser_normals"], dtype=torch.float32
            )
            self.camera_normals = torch.as_tensor(
                data["camera_normals"], dtype=torch.float32
            )
        else:
            default_normal = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            self.laser_normals = default_normal.repeat(len(self.camera_points), 1)
            self.camera_normals = default_normal.repeat(len(self.camera_points), 1)

        expected_length = self.trans.shape[0]
        arrays = {
            "camera_points": self.camera_points,
            "laser_points": self.laser_points,
            "laser_normals": self.laser_normals,
            "camera_normals": self.camera_normals,
        }
        for name, array in arrays.items():
            if array.shape[0] != expected_length:
                raise ValueError(
                    f"{name} contains {array.shape[0]} samples, but trans_data "
                    f"contains {expected_length}."
                )

    def __len__(self):
        return self.camera_points.shape[0]

    def __getitem__(self, index):
        return (
            self.laser_points[index],
            self.camera_points[index],
            self.trans[index],
            self.laser_normals[index],
            self.camera_normals[index],
        )


def build_dataloaders(config):
    """Build deterministic train and validation loaders from a MATLAB dataset."""
    dataset = RealTransDataset(config.mat_path, config.scale_factor)
    total_length = len(dataset)
    if total_length < 2:
        raise ValueError("At least two transient measurements are required.")

    train_length = min(int(config.train_split * total_length), total_length - 1)
    validation_length = total_length - train_length

    generator = torch.Generator()
    generator.manual_seed(int(getattr(config, "seed", 0)))
    train_dataset, validation_dataset = random_split(
        dataset, [train_length, validation_length], generator=generator
    )

    # Experiments with measured normals use the dedicated normal-aware batch size.
    # Historical used_config.yaml files call this field norm_cycle_on_batch; the
    # released pipeline does not alternate between normal-on and normal-off epochs.
    batch_size = (
        getattr(config, "norm_cycle_on_batch", config.batch_size)
        if dataset.has_normals
        else config.batch_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, config.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader, dataset
