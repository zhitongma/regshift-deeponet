"""
Fourier Neural Operator (2D) for parametric PDE surrogate modeling.

Operates on the fixed (z, t) grid and maps input parameters to (h, c) fields.
The forward signature is compatible with the DeepONet/FNN evaluation interface:
    forward(branch_input, trunk_input) -> (N, M, 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution via truncated FFT."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes_z: int, modes_t: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_t = modes_t

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_z, modes_t, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_z, modes_t, dtype=torch.cfloat))

    def _compl_mul2d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        mz, mt = self.modes_z, self.modes_t
        out_ft = torch.zeros(
            batch, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :mz, :mt] = self._compl_mul2d(x_ft[:, :, :mz, :mt], self.weights1)
        out_ft[:, :, -mz:, :mt] = self._compl_mul2d(x_ft[:, :, -mz:, :mt], self.weights2)

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """2D Fourier Neural Operator for the Richards-ADE surrogate problem.

    Input parameters (N, n_params) are broadcast to the (n_z, n_t) grid and
    concatenated with normalized coordinates, then processed through spectral
    convolution layers.
    """

    def __init__(
        self,
        n_params: int = 10,
        n_z: int = 101,
        n_t: int = 49,
        width: int = 64,
        modes_z: int = 20,
        modes_t: int = 12,
        n_layers: int = 4,
        num_outputs: int = 2,
        activation: str = "gelu",
        padding_z: int = 0,
        padding_t: int = 0,
    ):
        super().__init__()
        self.n_z = n_z
        self.n_t = n_t
        self.width = width
        self.n_layers = n_layers
        self.num_outputs = num_outputs
        # Domain padding for the non-periodic (z, t) domain: zero-pad before
        # the spectral layers and crop afterwards (Li et al., 2021 practice).
        # Defaults (0, 0) reproduce the original behaviour exactly.
        self.padding_z = padding_z
        self.padding_t = padding_t

        in_dim = n_params + 2

        self.lifting = nn.Linear(in_dim, width)

        self.spectral_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_convs.append(SpectralConv2d(width, width, modes_z, modes_t))
            self.pointwise_convs.append(nn.Conv2d(width, width, 1))
            self.norms.append(nn.InstanceNorm2d(width))

        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
        self.act = act_map[activation]()

        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            self.act,
            nn.Linear(128, num_outputs),
        )

        self._register_grid()
        self._init_weights()

    def _register_grid(self):
        gz = torch.linspace(0, 1, self.n_z)
        gt = torch.linspace(0, 1, self.n_t)
        grid_z, grid_t = torch.meshgrid(gz, gt, indexing="ij")
        grid = torch.stack([grid_z, grid_t], dim=-1)  # (n_z, n_t, 2)
        self.register_buffer("grid", grid)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        """Compatible interface with DeepONet / FNN.

        Parameters
        ----------
        branch_input : (N, n_params)
        trunk_input  : (M, 2)  — unused internally; grid is fixed

        Returns
        -------
        output : (N, M, num_outputs)
        """
        N = branch_input.shape[0]

        params = branch_input.unsqueeze(1).unsqueeze(1).expand(N, self.n_z, self.n_t, -1)
        grid = self.grid.unsqueeze(0).expand(N, -1, -1, -1)
        x = torch.cat([params, grid], dim=-1)  # (N, n_z, n_t, n_params+2)

        x = self.lifting(x)  # (N, n_z, n_t, width)
        x = x.permute(0, 3, 1, 2)  # (N, width, n_z, n_t)

        if self.padding_z > 0 or self.padding_t > 0:
            # F.pad pads the last dims first: (t_left, t_right, z_left, z_right).
            x = F.pad(x, (0, self.padding_t, 0, self.padding_z))

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.pointwise_convs[i](x)
            x = self.norms[i](x1 + x2)
            if i < self.n_layers - 1:
                x = self.act(x)

        if self.padding_z > 0 or self.padding_t > 0:
            x = x[..., : self.n_z, : self.n_t]

        x = x.permute(0, 2, 3, 1)  # (N, n_z, n_t, width)
        x = self.projection(x)  # (N, n_z, n_t, num_outputs)

        M = self.n_z * self.n_t
        return x.reshape(N, M, self.num_outputs)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_fno_model(cfg: dict) -> FNO2d:
    """Build FNO from the model config dict."""
    fno_cfg = cfg.get("fno", {})
    physics = cfg.get("physics", {})
    return FNO2d(
        n_params=cfg.get("branch_input_dim", 10),
        n_z=physics.get("n_spatial_nodes", 101),
        n_t=physics.get("n_time_steps", 49),
        width=fno_cfg.get("width", 64),
        modes_z=fno_cfg.get("modes_z", 20),
        modes_t=fno_cfg.get("modes_t", 12),
        n_layers=fno_cfg.get("n_layers", 4),
        num_outputs=cfg.get("num_outputs", 2),
        activation=cfg.get("activation", "gelu"),
        padding_z=fno_cfg.get("padding_z", 0),
        padding_t=fno_cfg.get("padding_t", 0),
    )
