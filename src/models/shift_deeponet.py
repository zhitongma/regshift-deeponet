"""
Shift-DeepONet: 参数依赖仿射坐标变换增强的 DeepONet。

核心思想: 不同参数下, 湿润锋/浓度锋位于不同时空位置。
标准 DeepONet 的 trunk 对所有样本使用相同坐标, 无法表达这种平移。
Shift-DeepONet 学习 params -> affine_transform, 用变换后的坐标计算 trunk。
"""

import math
import torch
import torch.nn as nn

from src.models.deeponet import (
    ResMLP, FourierFeatureEncoding, MultiScaleFourierEncoding, OutputFeatureHead,
)


class ShiftDeepONet(nn.Module):
    """DeepONet with parameter-dependent coordinate transformation.

    forward flow:
        1. TransformNet(params) -> affine parameters (scale, shift)
        2. trunk_shifted = scale * trunk + shift  (per-sample)
        3. trunk_features = Trunk(FourierEncode(trunk_shifted))  (batched)
        4. branch_coeffs = Branch(params)
        5. output = inner_product(branch_coeffs, trunk_features) + bias
    """

    def __init__(
        self,
        branch_input_dim: int = 10,
        branch_hidden: list[int] | None = None,
        trunk_input_dim: int = 2,
        trunk_hidden: list[int] | None = None,
        trunk_output_dim: int = 256,
        num_outputs: int = 2,
        activation: str = "gelu",
        fourier_features: int = 32,
        fourier_scale: float = 4.0,
        fourier_scales: list[float] | None = None,
        transform_hidden: list[int] | None = None,
        use_output_heads: bool = True,
        output_head_hidden: list[int] | None = None,
        transform_bound_scale: float = 0.0,
        transform_bound_shift: float = 0.0,
    ):
        super().__init__()
        if branch_hidden is None:
            branch_hidden = [512, 512, 512, 512]
        if trunk_hidden is None:
            trunk_hidden = [256, 256, 256, 256]
        if transform_hidden is None:
            transform_hidden = [64, 64]
        output_head_hidden = output_head_hidden or [256]

        self.num_outputs = num_outputs
        self.p = trunk_output_dim
        self.transform_bound_scale = transform_bound_scale
        self.transform_bound_shift = transform_bound_shift

        self.branch = ResMLP(
            [branch_input_dim] + branch_hidden + [self.p * num_outputs],
            activation, use_layer_norm=True,
        )

        if fourier_scales:
            self.trunk_encoder = MultiScaleFourierEncoding(
                trunk_input_dim, num_features=fourier_features, scales=fourier_scales,
            )
        else:
            self.trunk_encoder = FourierFeatureEncoding(
                trunk_input_dim, num_features=fourier_features, scale=fourier_scale,
            )
        trunk_in = self.trunk_encoder.output_dim
        self.trunk = ResMLP(
            [trunk_in] + trunk_hidden + [self.p],
            activation, use_layer_norm=True,
        )

        transform_layers = [branch_input_dim] + transform_hidden
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        t_layers = []
        for i in range(len(transform_layers) - 1):
            t_layers.append(nn.Linear(transform_layers[i], transform_layers[i + 1]))
            t_layers.append(act_cls())
        t_layers.append(nn.Linear(transform_layers[-1], 2 * trunk_input_dim))
        self.transform_net = nn.Sequential(*t_layers)
        with torch.no_grad():
            self.transform_net[-1].weight.zero_()
            self.transform_net[-1].bias.copy_(
                torch.tensor([1.0, 1.0, 0.0, 0.0])
            )

        self.output_heads = nn.ModuleList([
            OutputFeatureHead(self.p, hidden_dims=output_head_hidden,
                              activation=activation, use_layer_norm=True)
            for _ in range(num_outputs)
        ]) if use_output_heads else None

        self.bias = nn.Parameter(torch.zeros(num_outputs))
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if "transform_net" in name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        N = branch_input.shape[0]
        M = trunk_input.shape[0]
        D = trunk_input.shape[1]

        transform_params = self.transform_net(branch_input)
        if self.transform_bound_scale > 0:
            scale = 1.0 + self.transform_bound_scale * torch.tanh(transform_params[:, :D])
        else:
            scale = transform_params[:, :D]
        if self.transform_bound_shift > 0:
            shift = self.transform_bound_shift * torch.tanh(transform_params[:, D:])
        else:
            shift = transform_params[:, D:]

        trunk_expanded = trunk_input.unsqueeze(0).expand(N, M, D)
        trunk_shifted = trunk_expanded * scale.unsqueeze(1) + shift.unsqueeze(1)

        trunk_flat = trunk_shifted.reshape(N * M, D)
        trunk_encoded = self.trunk_encoder(trunk_flat)
        trunk_feats = self.trunk(trunk_encoded)
        trunk_feats = trunk_feats.reshape(N, M, self.p)

        b = self.branch(branch_input)
        b = b.view(N, self.num_outputs, self.p)

        outputs = []
        for k in range(self.num_outputs):
            t_k = trunk_feats
            if self.output_heads is not None:
                t_k = trunk_feats + self.output_heads[k](trunk_feats)
            out_k = torch.sum(b[:, k, :].unsqueeze(1) * t_k, dim=-1) + self.bias[k]
            outputs.append(out_k.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_shift_deeponet(cfg: dict) -> ShiftDeepONet:
    shift_cfg = cfg.get("shift_deeponet", {})
    return ShiftDeepONet(
        branch_input_dim=cfg.get("branch_input_dim", 10),
        branch_hidden=cfg.get("branch_hidden", [512, 512, 512, 512]),
        trunk_input_dim=cfg.get("trunk_input_dim", 2),
        trunk_hidden=cfg.get("trunk_hidden", [256, 256, 256, 256]),
        trunk_output_dim=cfg.get("trunk_output_dim", 256),
        num_outputs=cfg.get("num_outputs", 2),
        activation=cfg.get("activation", "gelu"),
        fourier_features=cfg.get("fourier_features", 32),
        fourier_scale=cfg.get("fourier_scale", 4.0),
        fourier_scales=cfg.get("fourier_scales", None),
        transform_hidden=shift_cfg.get("transform_hidden", [64, 64]),
        use_output_heads=cfg.get("use_output_heads", True),
        output_head_hidden=cfg.get("output_head_hidden", [256]),
        transform_bound_scale=shift_cfg.get("transform_bound_scale", 0.0),
        transform_bound_shift=shift_cfg.get("transform_bound_shift", 0.0),
    )
