"""
Fixed-grid FNN baseline.

This baseline maps the parameter vector directly to the native (z, t) grid
used during training. Off-grid time queries must therefore be obtained via
post-hoc interpolation rather than direct coordinate evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GridFNNBaseline(nn.Module):
    def __init__(
        self,
        branch_input_dim: int,
        n_points: int,
        hidden_dims: list[int] | None = None,
        num_outputs: int = 2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024, 1024]

        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers: list[nn.Module] = []
        in_dim = branch_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_points * num_outputs))

        self.net = nn.Sequential(*layers)
        self.n_points = n_points
        self.num_outputs = num_outputs
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        out = self.net(branch_input)
        return out.view(branch_input.shape[0], self.n_points, self.num_outputs)

    @property
    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters())


def build_grid_fnn_model(cfg: dict) -> GridFNNBaseline:
    grid_cfg = cfg.get("grid_fnn", {})
    physics = cfg.get("physics", {})
    n_z = int(physics.get("n_spatial_nodes", 101))
    n_t = int(physics.get("n_time_steps", 49))
    return GridFNNBaseline(
        branch_input_dim=int(cfg.get("branch_input_dim", 10)),
        n_points=n_z * n_t,
        hidden_dims=grid_cfg.get("hidden_dims", [1024, 1024, 1024]),
        num_outputs=int(cfg.get("num_outputs", 2)),
        activation=cfg.get("activation", "gelu"),
        use_layer_norm=bool(grid_cfg.get("use_layer_norm", True)),
    )
