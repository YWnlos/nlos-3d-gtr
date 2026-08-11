# 3D-GTR: 3D Gaussian Transient Rendering for NLOS Imaging

This repository contains the code, measured transients, optimized 3D Gaussian
primitive (3D GP) checkpoints, synthesized virtual-wall transients, and final
reconstruction images associated with:

> Yi Wang, Ziyu Zhan, Yuran Wang, Hao Wang, Qiang Liu, Zuoqiang Shi, Lingyun
> Qiu, and Xing Fu. **Non-line-of-sight imaging with arbitrary relay surface
> geometries via 3D Gaussian Transient Rendering.** SIGGRAPH Conference Papers
> 2026. [https://doi.org/10.1145/3799902.3811137](https://doi.org/10.1145/3799902.3811137)

3D-GTR represents the hidden scene with oriented 3D Gaussian primitives and
optimizes their positions, scales, rotations, and albedos so that differentiably
rendered transients match measured transients.

## Important scope

The code in this directory ends at **dense confocal transient synthesis on a
virtual relay surface**. `output.py` writes a MATLAB file containing the
synthesized transient grid. It does **not** run the downstream NLOS solver.
The final volumetric images shown in the paper were obtained by subsequently
running LCT or RSD, as specified in the paper and in `DATA_AND_RESULTS.md`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `nlos_3dgs.py` | 3D GP representation and differentiable transient forward model. |
| `triton_timekernel.py` | Memory-efficient custom backward pass for temporal Gaussian accumulation. |
| `nlos_3dgs_optimize.py` | Main optimization entry point. |
| `nlos_data.py` | MATLAB input loading and deterministic train/validation splitting. |
| `nlos_train.py` | AdamW optimization, learning-rate scheduling, validation, and checkpointing. |
| `output.py` | Dense confocal transient synthesis on a planar or user-provided virtual relay surface. |
| `generate_config.py` | Annotated example configuration generator. |
| `default_config.yaml` | Ready-to-edit example configuration. |
| `generate_virtual_wall.py` | Plane fitting and uniform virtual-wall generation. |
| `compute_normals.py` | Optional relay-point normal estimation helper. |
| `prune_gaussians.py` | Basic low-contribution Gaussian pruning. |
| `run_prune_pipeline.py` | Publication pruning pipeline, including ROI-boundary removal. |
| `save_load_model.py` | Checkpoint serialization and loading. |
| `utils.py` | Reproducibility and diagnostic plotting helpers. |
| `real_data/` | Measured or simulated transient datasets used by the released experiments. |
| `real_output/` | Experiment configurations, checkpoints, synthesized transients, and reconstructions. |
| `LICENSE` | MIT license for the 3D-GTR software and associated documentation. |
| `DATA_LICENSE` | CC BY 4.0 terms and scope for original 3D-GTR data and experiment artifacts. |
| `THIRD_PARTY_NOTICES.md` | Upstream terms and citation requirements for Teaser, Statue, and Bunny data. |

The misspelled historical entry point `generate_virtal_wall.py` is retained as a
thin compatibility wrapper. New scripts should use `generate_virtual_wall.py`.

The released binary assets occupy approximately 1.21 GiB (`real_data/` plus
`real_output/`). Before publishing, choose an intentional binary-distribution
strategy such as Git LFS or a versioned external data archive; avoid committing
large experiment artifacts to ordinary Git history by accident.

## Requirements

- Linux is recommended.
- A CUDA-capable NVIDIA GPU is required for optimization and rendering.
- Python 3.10 or newer.
- A CUDA-compatible PyTorch installation and Triton.

Install PyTorch using the command recommended for your CUDA version, then install
the remaining packages:

```bash
pip install -r requirements.txt
```

Because PyTorch/CUDA/Triton compatibility is platform-specific, `requirements.txt`
does not pin a CUDA build. Record the exact package versions and GPU model when
reporting new experiments.

## Input data format

Each input `.mat` file must contain:

| Variable | Shape | Meaning |
| --- | --- | --- |
| `camera_points` | `(M, 3)` | Detection points on the relay surface, in meters. |
| `trans_data` | `(M, 1, K)` or `(M, K)` | Measured transient histograms. |

It may additionally contain:

| Variable | Shape | Meaning |
| --- | --- | --- |
| `laser_points` | `(M, 3)` | Illumination points. If absent, `camera_points` are reused. |
| `laser_normals` | `(M, 3)` | Illumination-side relay normals. |
| `camera_normals` | `(M, 3)` | Detection-side relay normals. |

Normals are used only when **both** normal arrays are present. Planar data without
normals use the `+z` relay normal when `isplanar: true`.

## Quick start

Run commands from this directory so that the relative paths in the YAML files
resolve correctly.

1. Create and edit an example configuration:

   ```bash
   python generate_config.py
   ```

2. Optimize the 3D Gaussian representation:

   ```bash
   python nlos_3dgs_optimize.py --config default_config.yaml
   ```

   The output directory receives `used_config.yaml`, diagnostic transient-fit
   plots, loss histories, intermediate checkpoints, and `final_model.pth`.

3. Optionally prune primitives that lie near the reconstruction-volume boundary
   or make negligible contributions:

   ```bash
   python run_prune_pipeline.py \
     --input-model real_output/EXPERIMENT/final_model.pth \
     --output-model real_output/EXPERIMENT/final_model_pruned.pth \
     --device cuda:0 \
     --edge-ratio 0.05 \
     --low-albedo-threshold 0.005
   ```

4. Render a dense virtual-wall transient grid:

   ```bash
   python output.py \
     --config real_output/EXPERIMENT/used_config.yaml \
     --model real_output/EXPERIMENT/final_model_pruned.pth
   ```

   Omit `--model` to use `final_model.pth`. The generated
   `resample_pred_trans_128x128.mat` contains the virtual-wall samples and
   synthesized transients.

5. Run an external LCT or RSD implementation on the synthesized transient grid to
   obtain the final reconstruction. Solver implementations are not included in
   this release.

## Configuration guide

### Paths and data scaling

| Key | Selection guidance |
| --- | --- |
| `mat_path` | Input MATLAB dataset. Use a path relative to this directory. |
| `output_dir` | A new directory for one experiment. Do not reuse a directory unless overwriting is intended. |
| `scale_factor` | Multiplies `trans_data`. Choose it so rendered and measured transients have comparable numerical magnitude. Released values are recorded in each `used_config.yaml`. |
| `train_split` | Fraction used for fitting; the rest is held out for physical-consistency validation. The paper uses 0.98 or 0.99. |
| `batch_size` | Batch size for inputs without measured normals. |
| `norm_cycle_on_batch` | Batch size for inputs that contain both laser and camera normals. Despite its historical name, no on/off cycle is used. |
| `num_workers` | DataLoader worker processes. Reduce to `0` if multiprocessing is unavailable. |

### Reconstruction volume and Gaussian initialization

| Key | Selection guidance |
| --- | --- |
| `num_gaussians_init` | Initial number of 3D GPs. More primitives increase capacity, memory, and runtime. Released experiments use 500 to 50,000. |
| `x_min/x_max`, `y_min/y_max`, `z_min/z_max` | Hidden-scene region of interest in meters. It must contain the expected target with minimal unused volume. |
| `scale_min/scale_max` | Allowed principal-axis standard-deviation range in meters. Choose values relative to target feature size and sampling density. |
| `scale_init` | Initial principal-axis scale; it must lie within the scale bounds. |
| `albedo_init` | Initial Gaussian albedo. Avoid zero, which can suppress useful gradients. |
| `activation_mode` | Center parameterization. `tanh` bounds centers to the reconstruction volume. |
| `scale_activation_mode` | `tanh` strictly bounds scale; historical checkpoints may use `relu`. |
| `albedo_activation_mode` | `tanh` bounds albedo to `[0,1]`; historical checkpoints may use `relu`. |
| `anisotropy_threshold` | Maximum largest-to-smallest scale ratio. Values greater than 1 suppress needle-like primitives by clamping activated scales. |

### Forward-model assumptions

| Key | Selection guidance |
| --- | --- |
| `isplanar` | Set `true` for planar relay data without measured normals. Set `false` for arbitrary relay geometry. |
| `isconfocal` | Set `true` only when every illumination point equals its detection point. |
| `isretroreflective` | Specialized confocal amplitude model. It requires `isconfocal: true`; released experiments use `false`. |
| `render_magnification` | Global gain `g` from the paper. Adjust together with `scale_factor`, not the transient shape. |
| `dT` | Input temporal-bin width in seconds. It must match the acquisition or simulation. Released values include 10 ps, 16 ps, and 32 ps. |
| `t_range` | Half-open input-bin interval `[start, end)`. Include the full nonzero signal support while excluding irrelevant bins when possible. |

### Optimization

| Key | Selection guidance |
| --- | --- |
| `num_epochs` | Number of complete passes through the fitting split. Refer to Fig. S3 for the speed-quality trade-off. |
| `save_interval` | Epoch interval for intermediate checkpoints. |
| `lr_schedule` | `multistep` or `none`. |
| `warmup_ratio` | Fraction of all optimizer steps used for linear warmup. |
| `milestones` | Epoch numbers at which all learning rates are multiplied by `gamma`. |
| `gamma` | Multiplicative learning-rate decay at each milestone. |
| `weight_decay` | AdamW weight decay. The released settings use a very small value. |
| `lr_xy`, `lr_z`, `lr_scale`, `lr_quat`, `lr_albedo` | Attribute-specific rates for inputs without normals. Position and albedo often require different numerical scales. |
| `lr_*_post_norm` | Corresponding rates for datasets containing measured relay normals. |

### Virtual-wall synthesis

| Key | Selection guidance |
| --- | --- |
| `output_wallsize` | Side length in meters of an automatically generated square wall. |
| `output_res` | Samples per side; the released results use 128. |
| `output_t_range` | Half-open output-bin interval. It must cover the synthesized transient support expected by the downstream solver. |
| `output_dT` | Output-bin width in seconds. |
| `predict_batch_size` | Inference batch size; reduce it if GPU memory is insufficient. |
| `use_auto_wall` | If `true`, generate a square plane at `z=0`; otherwise load `virtual_wall_path`. |
| `virtual_wall_path` | MATLAB file containing `(N,3)` arrays `virtual_cam_pts` and `wall_normals`. |

## Historical configuration keys

The released `real_output/*/used_config.yaml` files are preserved as experiment
records. They contain several zero-valued or disabled keys from development:

- adaptive density control: `adaptive_control`, `adaptive_control_interval`,
  `grad_threshold`, `min_opacity`, `prune_upthreshold`, `prune_lowthreshold`, and
  `split_threshold`;
- unused regularization or augmentation: all `lambda_*` keys,
  `aug_noise_std_*`, and `simplified_amplitude_epochs`;
- an unused normal-cycling design: `laser_norm_start_epoch`,
  `norm_cycle_on_len`, `norm_cycle_off_len`, and `norm_cycle_off_batch`.

These fields had no effect on the reported experiments and are ignored by the
released training code. `norm_cycle_on_batch` remains active solely as the batch
size for datasets containing measured normals. For data without normals,
`batch_size` is used.

## Reproducing paper figures

See [`DATA_AND_RESULTS.md`](DATA_AND_RESULTS.md) for the exact mapping from paper
figures to `real_data/` and `real_output/`, artifact naming conventions, the two
corrected historical data paths, and the boundary between 3D-GTR output and the
downstream LCT/RSD reconstruction.

## Citation

Please cite the paper when using this code or data. A machine-readable citation is
provided in `CITATION.cff`.

## License

3D-GTR uses separate licenses for software and data:

| Material | Terms |
| --- | --- |
| Original 3D-GTR software and associated documentation | [MIT License](LICENSE) |
| Original 3D-GTR datasets, checkpoints, synthesized transients, and reconstruction images | [CC BY 4.0](DATA_LICENSE) |
| Processed Teaser and Statue measurements and their derived experiment directories | Stanford nlos-fk non-commercial terms; see [third-party notices](THIRD_PARTY_NOTICES.md) |
| Processed Zaragoza Bunny measurements and their derived experiment directories | Publicly provided for research use; see [third-party notices](THIRD_PARTY_NOTICES.md) |

Copyright (c) 2026 Yi Wang, Ziyu Zhan and the 3D-GTR authors.

The Teaser and Statue materials remain subject to the upstream nlos-fk terms.
The Zaragoza Bunny materials are included for academic research and
reproducibility under the research-use context stated by the dataset provider.
Users must preserve the upstream notices and citations; the repository's MIT
and CC BY 4.0 licenses do not relicense these third-party materials.
