"""Differentiable 3D Gaussian transient renderer used by 3D-GTR."""

import math

import torch
import torch.nn as nn
from triton_timekernel import time_kernel_and_reduce_triton_fullG
class NLOS_3DGS(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # Learnable parameters are allocated by initialize().
        self.xy_centers_raw = None
        self.z_centers_raw = None
        self.raw_scaling = None
        self.quats = None
        self.albedo = None

        # These buffers are retained for released-checkpoint compatibility.
        self.register_buffer('xyz_gradient_accum', torch.zeros(0))
        self.register_buffer('denom', torch.zeros(0))


    def ensure_timebin_buffer(self, trans_range, device, dtype=torch.float32):
        """Create or update the non-persistent (1,1,K) time-bin buffer."""
        if isinstance(trans_range, int):
            start, end = 0, int(trans_range)
        else:
            start, end = int(trans_range[0]), int(trans_range[1])
        K = end - start

        new_tb = torch.arange(start, end, device=device, dtype=dtype).view(1, 1, K)

        if getattr(self, "timebin", None) is not None:
            self.timebin = new_tb
        else:
            self.register_buffer("timebin", new_tb, persistent=False)

    def ensure_aux_buffers(self, B, N, device, dtype=torch.float32):
        """Prepare a reusable (B,N) all-ones buffer for compiled execution."""
        need_new = (
            (getattr(self, "ones_BN", None) is None) or
            (self.ones_BN.shape != (B, N)) or
            (self.ones_BN.device != device) or
            (self.ones_BN.dtype  != dtype)
        )
        if need_new:
            self.register_buffer("ones_BN", torch.ones(B, N, device=device, dtype=dtype),
                                persistent=False)
        else:
            pass

    def _apply_activation(self, raw, a, b):
        if self.activation_mode == 'tanh':
            return ((torch.tanh(raw) + 1) / 2) * (b - a) + a
        elif self.activation_mode == 'trig':
            return ((torch.sin(raw) + 1) / 2) * (b - a) + a
        elif self.activation_mode == 'relu':
            return self._apply_activation_relu(raw, a, b)
        else:
            raise ValueError(f"Unsupported activation mode: {self.activation_mode}")

    def _inverse_activation(self, val, a, b):
        if self.activation_mode == 'relu':
            return self._inverse_activation_relu(val, a, b)
        eps = 1e-6
        val = torch.clamp(val, a + eps, b - eps)
        normed = (2 * (val - a) / (b - a)) - 1
        if self.activation_mode == 'tanh':
            return 0.5 * torch.log((1 + normed) / (1 - normed))
        elif self.activation_mode == 'trig':
            return torch.asin(normed)
        else:
            raise ValueError(f"Unsupported activation mode: {self.activation_mode}")

    def _apply_activation_relu(self, raw, a, b):
        return torch.nn.functional.relu(raw)

    def _inverse_activation_relu(self, val, a, b):
        return val

    def compute_xy(self, raw_xy):
        return torch.stack([
            self._apply_activation(raw_xy[:, 0], self.x_min, self.x_max),
            self._apply_activation(raw_xy[:, 1], self.y_min, self.y_max)
        ], dim=1)

    def xy_to_raw(self, xy):
        return torch.stack([
            self._inverse_activation(xy[:, 0], self.x_min, self.x_max),
            self._inverse_activation(xy[:, 1], self.y_min, self.y_max)
        ], dim=1)

    def compute_z(self, raw_z):
        return self._apply_activation(raw_z.squeeze(-1), self.z_min, self.z_max).unsqueeze(-1)

    def z_to_raw(self, z):
        return self._inverse_activation(z.squeeze(-1), self.z_min, self.z_max).unsqueeze(-1)

    def compute_scale(self, raw):
        if self.scale_activation_mode == 'relu':
            scale = torch.clamp(torch.nn.functional.relu(raw) + self.scale_min, max=self.scale_max)
        elif self.scale_activation_mode == 'tanh':
            scale = ((torch.tanh(raw) + 1) / 2) * (self.scale_max - self.scale_min) + self.scale_min
        elif self.scale_activation_mode == 'trig':
            scale = ((torch.sin(raw) + 1) / 2) * (self.scale_max - self.scale_min) + self.scale_min
        else:
            raise ValueError(f"Unsupported scale activation: {self.scale_activation_mode}")

        # Enforce anisotropy constraint if configured
        if hasattr(self, 'anisotropy_threshold') and self.anisotropy_threshold > 1.0:
            # We want max(s) / min(s) <= T
            # This implies s_i >= max(s) / T for all i

            max_s = scale.max(dim=1, keepdim=True)[0]
            min_allowed = max_s / self.anisotropy_threshold

            # Clamp all scales to be at least min_allowed
            scale = torch.max(scale, min_allowed)

        return scale

    def scale_to_raw(self, scale):
        if self.scale_activation_mode == 'relu':
            return scale - self.scale_min
        elif self.scale_activation_mode == 'tanh':
            eps = 1e-6
            scale = torch.clamp(scale, self.scale_min + eps, self.scale_max - eps)
            normed = (2 * (scale - self.scale_min) / (self.scale_max - self.scale_min)) - 1
            return 0.5 * torch.log((1 + normed) / (1 - normed))
        elif self.scale_activation_mode == 'trig':
            eps = 1e-6
            scale = torch.clamp(scale, self.scale_min + eps, self.scale_max - eps)
            normed = (2 * (scale - self.scale_min) / (self.scale_max - self.scale_min)) - 1
            return torch.asin(normed)
        else:
            raise ValueError(f"Unsupported scale activation: {self.scale_activation_mode}")

    def compute_albedo(self, raw):
        if self.albedo_activation_mode == 'relu':
            return torch.nn.functional.relu(raw)
        elif self.albedo_activation_mode == 'tanh':
            return (torch.tanh(raw) + 1) / 2
        elif self.albedo_activation_mode == 'trig':
            return (torch.sin(raw) + 1) / 2
        else:
            raise ValueError(f"Unsupported albedo activation: {self.albedo_activation_mode}")

    def albedo_to_raw(self, albedo):
        if self.albedo_activation_mode == 'relu':
            return albedo
        elif self.albedo_activation_mode == 'tanh':
            eps = 1e-6
            albedo = torch.clamp(albedo, eps, 1 - eps)
            normed = 2 * albedo - 1
            return 0.5 * torch.log((1 + normed) / (1 - normed))
        elif self.albedo_activation_mode == 'trig':
            eps = 1e-6
            albedo = torch.clamp(albedo, eps, 1 - eps)
            normed = 2 * albedo - 1
            return torch.asin(normed)
        else:
            raise ValueError(f"Unsupported albedo activation: {self.albedo_activation_mode}")

    def _init_quats(self, N, device):
        u1 = torch.rand(N, device=device)
        u2 = torch.rand(N, device=device) * 2 * math.pi
        u3 = torch.rand(N, device=device) * 2 * math.pi
        a = torch.sqrt(1 - u1)
        b = torch.sqrt(u1)
        return torch.stack([a*torch.sin(u2), a*torch.cos(u2), b*torch.sin(u3), b*torch.cos(u3)], dim=1)

    def initialize(self,
                   num_gaussians,
                   x_min, x_max, y_min, y_max, z_min, z_max,
                   scale_min, scale_max, scale_init,
                   albedo_init,
                   isplanar, isconfocal, isretroreflective,
                   activation_mode='tanh',
                   scale_activation_mode='relu',
                   albedo_activation_mode='relu',
                   anisotropy_threshold=0.0):

        self.num_gaussians = num_gaussians
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        self.scale_min, self.scale_max = scale_min, scale_max
        self.activation_mode = activation_mode  # For xyz positions
        self.scale_activation_mode = scale_activation_mode  # For scale
        self.albedo_activation_mode = albedo_activation_mode  # For albedo
        self.isplanar = bool(isplanar)
        self.isconfocal = bool(isconfocal)
        self.isretroreflective = bool(isretroreflective)
        self.anisotropy_threshold = anisotropy_threshold


        N = num_gaussians
        device = self.device

        xy = torch.empty(N, 2, device=device)
        xy[:, 0] = torch.rand(N, device=device) * (x_max - x_min) + x_min
        xy[:, 1] = torch.rand(N, device=device) * (y_max - y_min) + y_min
        self.xy_centers_raw = nn.Parameter(self.xy_to_raw(xy))

        z = torch.rand(N, 1, device=device) * (z_max - z_min) + z_min
        self.z_centers_raw = nn.Parameter(self.z_to_raw(z))

        scaling = torch.ones(N, 3, device=device) * scale_init
        self.raw_scaling = nn.Parameter(self.scale_to_raw(scaling))

        self.quats = nn.Parameter(self._init_quats(N, device))

        albedo = torch.ones(N, 1, device=device) * albedo_init
        self.albedo = nn.Parameter(self.albedo_to_raw(albedo))

        self.xyz_gradient_accum = torch.zeros(N, device=device)
        self.denom = torch.zeros(N, device=device)

    def get_gs_centers(self):
        xy_centers = self.compute_xy(self.xy_centers_raw)
        z_centers = self.compute_z(self.z_centers_raw)
        return torch.cat([xy_centers, z_centers], dim=1)

    def forward(
        self,
        camera_points,
        dt,
        trans_range,
        laser_points,
        render_magnification=1,
        laser_normals=None,
        camera_normals=None,
    ):
        centers = self.get_gs_centers()
        albedos = self.compute_albedo(self.albedo)
        if isinstance(laser_normals, list) and len(laser_normals) == 0:
            laser_normals = None
        if isinstance(camera_normals, list) and len(camera_normals) == 0:
            camera_normals = None
        result = self.batch_render_nonconfocal(
            centers,
            albedos,
            laser_points,
            camera_points,
            dt,
            trans_range,
            laser_normals,
            camera_normals,
        ) * render_magnification
        return result

    def batch_render_nonconfocal(
        self,
        gs_centers,       # (N,3)
        albedos,          # (N,1)
        laser_points,     # (B,3)
        camera_points,    # (B,3)
        dt,
        trans_range,
        laser_normals,    # (B,3) or None
        camera_normals    # (B,3) or None
    ):
        """Render non-confocal transients from oriented 3D Gaussian primitives.

        The implementation evaluates covariance terms in each Gaussian's principal
        frame and avoids materializing full covariance matrices.
        """
        isplanar  = self.isplanar
        isconfocal = self.isconfocal
        isretroreflective = self.isretroreflective

        device = gs_centers.device
        dtype  = gs_centers.dtype
        eps    = 1e-12

        # Convert path lengths in meters to temporal-bin coordinates.
        c_dt_inv = 1.0 / (3e8 * dt)

        B = laser_points.shape[0]
        N = gs_centers.shape[0]

        # Convert quaternion and principal scales to a local Gaussian frame.
        quats = self.quats / (torch.norm(self.quats, dim=1, keepdim=True) + 1e-6)  # (N,4)
        w, x, y, z = quats.unbind(dim=1)
        rot = torch.stack([
            1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
        ], dim=1).view(-1, 3, 3)                                        # (N,3,3)
        RT = rot.transpose(1, 2).contiguous()                            # (N,3,3)
        RT_b = RT.unsqueeze(0).expand(B, -1, -1, -1)                     # (B,N,3,3)

        s        = self.compute_scale(self.raw_scaling).clamp_min(1e-12).view(N, 3) # (N,3)
        inv_s    = 1.0 / s                                               # (N,3)
        sqrt_det = s.prod(dim=1)                                         # (N,)

        # Compute illumination and detection directions. Confocal data reuse u1.
        xg = gs_centers.unsqueeze(0)       # (B,N,3), broadcast over measurements
        xl = laser_points.unsqueeze(1)     # (B,N,3)
        a  = xg - xl
        d1_sq = (a * a).sum(dim=2).clamp_min(eps)    # (B,N)
        d1    = d1_sq.sqrt()
        u1    = a / d1.unsqueeze(-1)                 # (B,N,3)

        if isconfocal:
            d2_sq = d1_sq
            d2    = d1
            u2    = u1
        else:
            xc   = camera_points.unsqueeze(1)        # (B,N,3)
            b    = xg - xc
            d2_sq = (b * b).sum(dim=2).clamp_min(eps)
            d2    = d2_sq.sqrt()
            u2    = b / d2.unsqueeze(-1)             # (B,N,3)

        # Rotate directions into each Gaussian's principal frame.
        u1p = torch.matmul(u1.unsqueeze(2), RT_b).squeeze(2)   # (B,N,3)
        u2p = torch.matmul(u2.unsqueeze(2), RT_b).squeeze(2)   # (B,N,3)

        # u^T Sigma^-1 u = ||u' / s||^2.
        inv_s_b = inv_s.view(1, N, 3)                          # (1,N,3)
        quad_u1 = (u1p * inv_s_b).pow_(2).sum(dim=-1)          # (B,N)
        quad_u2 = quad_u1 if isconfocal else (u2p * inv_s_b).pow_(2).sum(dim=-1)

        sqrt_det_b = sqrt_det.unsqueeze(0)                     # (B,N)
        sqrt_in  = sqrt_det_b * (quad_u1 + eps).sqrt()         # (B,N)
        sqrt_out = sqrt_in if isconfocal else (sqrt_det_b * (quad_u2 + eps).sqrt())

        # Temporal standard deviation in bins after propagating spatial covariance.
        w   = u1 + u2                                          # (B,N,3)
        wp  = torch.matmul(w.unsqueeze(2), RT_b).squeeze(2)    # (B,N,3)
        std_bins = (((wp * s.unsqueeze(0)).pow_(2)).sum(dim=-1) + eps).sqrt() * c_dt_inv  # (B,N)

        # Relay-surface cosine terms.
        if (laser_normals is not None) and (not (hasattr(laser_normals, "numel") and laser_normals.numel() == 0)):
            nl_eff = laser_normals.unsqueeze(1).expand(-1, N, -1)
            nl_eff = torch.nn.functional.normalize(nl_eff, dim=-1)
            # Clamp min to avoid zero gradients for grazing angles
            # Use Softplus to ensure non-negative intensity while keeping gradients smooth
            cos1 = torch.nn.functional.relu((nl_eff * u1).sum(dim=-1))  # (B,N)
        elif isplanar:
            # A planar relay surface uses the +z normal.
            cos1 = torch.nn.functional.relu(u1[..., 2])                # (B,N)
        else:
            # Arbitrary surfaces without normals omit the cosine term.
            cos1 = torch.ones_like(d1_sq)

        if isconfocal:
            cos2 = cos1
        else:
            if (camera_normals is not None) and (not (hasattr(camera_normals, "numel") and camera_normals.numel() == 0)):
                nc_eff = camera_normals.unsqueeze(1).expand(-1, N, -1)
                nc_eff = torch.nn.functional.normalize(nc_eff, dim=-1)
                cos2 = torch.nn.functional.relu((nc_eff * u2).sum(dim=-1))
            elif isplanar:
                cos2 = torch.nn.functional.relu(u2[..., 2])
            else:
                cos2 = torch.ones_like(d2_sq)

        # Assemble amplitude terms while minimizing intermediate tensors.
        rho = albedos.squeeze(-1).unsqueeze(0).expand(B, -1)
        if isretroreflective:  # Valid only for isconfocal=True.
            A = rho * cos1 * (sqrt_in / d1_sq)
        else:
            A_core = (sqrt_det_b * sqrt_det_b) * (
                quad_u1.mul(quad_u2).add_(eps).sqrt()
            )
            A = rho * (cos1 * cos2) * (A_core / (d1_sq * d2_sq))
        tb = self.timebin                               # (1,1,K)
        mu  = (d1 + d2).mul_(c_dt_inv).unsqueeze(-1)    # (B,N,1)
        std = (std_bins + eps).unsqueeze(-1)            # (B,N,1)

        # Use Triton kernel for memory-efficient backward pass
        trans = time_kernel_and_reduce_triton_fullG(A, mu.squeeze(-1), std.squeeze(-1), tb.view(-1))
        return trans
