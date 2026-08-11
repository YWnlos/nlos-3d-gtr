"""Memory-efficient autograd kernel for Gaussian transient accumulation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _backward_reduce_kernel(
    amplitude_ptr,
    mean_ptr,
    std_ptr,
    time_ptr,
    gaussian_ptr,
    output_gradient_ptr,
    amplitude_gradient_ptr,
    mean_gradient_ptr,
    std_gradient_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_a_b,
    stride_a_n,
    stride_mean_b,
    stride_mean_n,
    stride_std_b,
    stride_std_n,
    stride_gaussian_b,
    stride_gaussian_n,
    stride_gaussian_k,
    stride_output_b,
    stride_output_k,
    stride_gradient_b,
    stride_gradient_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_index = tl.program_id(0)
    gaussian_block = tl.program_id(1)
    gaussian_offsets = gaussian_block * BLOCK_N + tl.arange(0, BLOCK_N)
    gaussian_mask = gaussian_offsets < N

    amplitude = tl.load(
        amplitude_ptr
        + batch_index * stride_a_b
        + gaussian_offsets * stride_a_n,
        mask=gaussian_mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(
        mean_ptr
        + batch_index * stride_mean_b
        + gaussian_offsets * stride_mean_n,
        mask=gaussian_mask,
        other=0.0,
    ).to(tl.float32)
    std = tl.load(
        std_ptr
        + batch_index * stride_std_b
        + gaussian_offsets * stride_std_n,
        mask=gaussian_mask,
        other=1.0,
    ).to(tl.float32)

    gradient_amplitude = tl.zeros([BLOCK_N], dtype=tl.float32)
    gradient_mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    gradient_std = tl.zeros([BLOCK_N], dtype=tl.float32)
    time_offsets = tl.arange(0, BLOCK_K)

    for time_start in range(0, K, BLOCK_K):
        time_indices = time_start + time_offsets
        time_mask = time_indices < K
        output_gradient = tl.load(
            output_gradient_ptr
            + batch_index * stride_output_b
            + time_indices * stride_output_k,
            mask=time_mask,
            other=0.0,
        ).to(tl.float32)
        time = tl.load(
            time_ptr + time_indices, mask=time_mask, other=0.0
        ).to(tl.float32)

        gaussian_pointers = (
            gaussian_ptr
            + batch_index * stride_gaussian_b
            + gaussian_offsets[:, None] * stride_gaussian_n
            + time_indices[None, :] * stride_gaussian_k
        )
        gaussian = tl.load(
            gaussian_pointers,
            mask=gaussian_mask[:, None] & time_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        normalized_time = (time[None, :] - mean[:, None]) / std[:, None]

        gradient_amplitude += tl.where(
            gaussian_mask,
            tl.sum(output_gradient[None, :] * gaussian, axis=1),
            0.0,
        )
        gradient_mean += tl.where(
            gaussian_mask,
            tl.sum(
                output_gradient[None, :]
                * amplitude[:, None]
                * gaussian
                * (normalized_time / std[:, None]),
                axis=1,
            ),
            0.0,
        )
        gradient_std += tl.where(
            gaussian_mask,
            tl.sum(
                output_gradient[None, :]
                * amplitude[:, None]
                * gaussian
                * (normalized_time * normalized_time / std[:, None]),
                axis=1,
            ),
            0.0,
        )

    gradient_offsets = (
        batch_index * stride_gradient_b + gaussian_offsets * stride_gradient_n
    )
    tl.store(
        amplitude_gradient_ptr + gradient_offsets,
        gradient_amplitude,
        mask=gaussian_mask,
    )
    tl.store(
        mean_gradient_ptr + gradient_offsets, gradient_mean, mask=gaussian_mask
    )
    tl.store(
        std_gradient_ptr + gradient_offsets, gradient_std, mask=gaussian_mask
    )


class TimeKernelReduce(torch.autograd.Function):
    """Autograd function for summing temporal Gaussian responses."""

    @staticmethod
    def forward(ctx, amplitude, mean, std, time_bins):
        if not (amplitude.ndim == mean.ndim == std.ndim == 2):
            raise ValueError("amplitude, mean, and std must have shape (B, N).")
        if time_bins.ndim != 1:
            raise ValueError("time_bins must have shape (K,).")

        gaussian = torch.exp(
            -0.5
            * (
                (time_bins.view(1, 1, -1) - mean.unsqueeze(-1))
                / std.unsqueeze(-1)
            ).square()
        )
        transient = (amplitude.unsqueeze(-1) * gaussian).sum(dim=1)
        ctx.save_for_backward(
            amplitude,
            mean,
            std,
            time_bins,
            gaussian.to(torch.float16).contiguous(),
        )
        return transient

    @staticmethod
    def backward(ctx, output_gradient):
        amplitude, mean, std, time_bins, gaussian = ctx.saved_tensors
        batch_size, num_gaussians = amplitude.shape
        num_bins = time_bins.numel()
        gradient_amplitude = torch.empty_like(amplitude, dtype=torch.float32)
        gradient_mean = torch.empty_like(mean, dtype=torch.float32)
        gradient_std = torch.empty_like(std, dtype=torch.float32)

        block_n = 32
        block_k = 64
        grid = (batch_size, triton.cdiv(num_gaussians, block_n))
        _backward_reduce_kernel[grid](
            amplitude,
            mean,
            std,
            time_bins,
            gaussian,
            output_gradient.contiguous(),
            gradient_amplitude,
            gradient_mean,
            gradient_std,
            batch_size,
            num_gaussians,
            num_bins,
            amplitude.stride(0),
            amplitude.stride(1),
            mean.stride(0),
            mean.stride(1),
            std.stride(0),
            std.stride(1),
            gaussian.stride(0),
            gaussian.stride(1),
            gaussian.stride(2),
            output_gradient.stride(0),
            output_gradient.stride(1),
            gradient_amplitude.stride(0),
            gradient_amplitude.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
        )
        return (
            gradient_amplitude.to(amplitude.dtype),
            gradient_mean.to(mean.dtype),
            gradient_std.to(std.dtype),
            None,
        )


def time_kernel_and_reduce_triton_fullG(amplitude, mean, std, time_bins):
    """Accumulate Gaussian time kernels and expose custom backward gradients."""
    return TimeKernelReduce.apply(amplitude, mean, std, time_bins)
