"""
多输出 DeepONet (纯 PyTorch 实现)

支持两种 trunk 模式:
  - shared trunk (默认): h 和 c 共享同一组 basis functions
  - dual trunk: h 和 c 各有独立的 trunk 网络和 Fourier 编码器,
    学习各自适合的时空基函数 (h 偏扩散, c 偏对流)
"""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机, 最后一层无激活函数。"""

    def __init__(self, layer_sizes: list[int], activation: str = "tanh"):
        super().__init__()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act_fn())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """残差块: Linear -> LayerNorm(可选) -> Activation -> Linear -> LayerNorm(可选) + skip。"""

    def __init__(self, dim: int, activation: str = "gelu",
                 use_layer_norm: bool = True):
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        layers = []
        layers.append(nn.Linear(dim, dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dim))
        layers.append(act_cls())
        layers.append(nn.Linear(dim, dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dim))
        self.block = nn.Sequential(*layers)
        self.act = act_cls()

    def forward(self, x):
        return self.act(x + self.block(x))


class ResMLP(nn.Module):
    """带残差连接的 MLP。

    结构: Linear(in -> hidden) -> [ResBlock] * (n_hidden - 1) -> Linear(hidden -> out)
    当相邻层宽度相同时使用 ResBlock, 首尾层做普通投影。
    """

    def __init__(self, layer_sizes: list[int], activation: str = "gelu",
                 use_layer_norm: bool = True):
        super().__init__()
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        hidden_dims = layer_sizes[1:-1]
        assert len(hidden_dims) >= 1, "ResMLP needs at least one hidden layer"
        hidden_dim = hidden_dims[0]

        modules = []
        # input projection
        modules.append(nn.Linear(layer_sizes[0], hidden_dim))
        if use_layer_norm:
            modules.append(nn.LayerNorm(hidden_dim))
        modules.append(act_cls())

        # residual blocks for consecutive same-width layers
        for i in range(1, len(hidden_dims)):
            if hidden_dims[i] == hidden_dims[i - 1]:
                modules.append(ResBlock(hidden_dims[i], activation, use_layer_norm))
            else:
                modules.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                if use_layer_norm:
                    modules.append(nn.LayerNorm(hidden_dims[i]))
                modules.append(act_cls())

        # output projection (no activation)
        modules.append(nn.Linear(hidden_dims[-1], layer_sizes[-1]))

        self.net = nn.Sequential(*modules)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class FourierFeatureEncoding(nn.Module):
    """对 trunk 坐标做随机 Fourier 特征编码。"""

    def __init__(self, input_dim: int, num_features: int = 32, scale: float = 4.0):
        super().__init__()
        b_matrix = torch.randn(input_dim, num_features) * scale
        self.register_buffer("b_matrix", b_matrix)
        self.output_dim = input_dim + 2 * num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.b_matrix
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class MultiScaleFourierEncoding(nn.Module):
    """多尺度随机 Fourier 特征编码，同时捕获低频（h平滑场）和高频（c锋面）。"""

    def __init__(self, input_dim: int, num_features: int = 16,
                 scales: list[float] | None = None):
        super().__init__()
        if scales is None:
            scales = [1.0, 4.0, 16.0]
        parts = [torch.randn(input_dim, num_features) * s for s in scales]
        self.register_buffer("b_matrix", torch.cat(parts, dim=1))
        self.output_dim = input_dim + 2 * num_features * len(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.b_matrix
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class OutputFeatureHead(nn.Module):
    """对共享 trunk 特征做轻量输出解耦。"""

    def __init__(self, dim: int, hidden_dims: list[int], activation: str = "gelu",
                 use_layer_norm: bool = True):
        super().__init__()
        self.use_residual = bool(hidden_dims)
        if not hidden_dims:
            self.net = nn.Identity()
            return

        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        layers = []
        in_dim = dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net, nn.Identity):
            return x
        return x + self.net(x)


class MultiOutputDeepONet(nn.Module):
    """split_branch DeepONet, 支持 shared trunk 或 dual trunk 模式。

    dual_trunk=False (默认): h 和 c 共享一个 trunk, 与原始行为完全一致。
    dual_trunk=True: h 和 c 各有独立的 trunk 网络和 Fourier 编码器,
        允许两个物理场学习各自最优的时空基函数。
    """

    def __init__(
        self,
        branch_input_dim: int = 10,
        branch_hidden: list[int] | None = None,
        trunk_input_dim: int = 2,
        trunk_hidden: list[int] | None = None,
        trunk_output_dim: int = 128,
        num_outputs: int = 2,
        activation: str = "tanh",
        use_residual: bool = False,
        use_layer_norm: bool = False,
        trunk_encoding: str = "none",
        fourier_features: int = 32,
        fourier_scale: float = 4.0,
        fourier_scales: list[float] | None = None,
        use_output_heads: bool = False,
        output_head_hidden: list[int] | None = None,
        dual_trunk: bool = False,
    ):
        super().__init__()

        if branch_hidden is None:
            branch_hidden = [256, 256, 256]
        if trunk_hidden is None:
            trunk_hidden = [128, 128, 128]

        self.num_outputs = num_outputs
        self.p = trunk_output_dim
        self.dual_trunk = dual_trunk
        output_head_hidden = output_head_hidden or []

        if use_residual:
            mlp_cls = lambda sizes: ResMLP(sizes, activation, use_layer_norm)
        else:
            mlp_cls = lambda sizes: MLP(sizes, activation)

        branch_layers = [branch_input_dim] + branch_hidden + [self.p * num_outputs]
        self.branch = mlp_cls(branch_layers)

        def _make_encoder():
            if trunk_encoding == "multiscale_fourier" and fourier_scales:
                return MultiScaleFourierEncoding(
                    trunk_input_dim, num_features=fourier_features, scales=fourier_scales
                )
            elif trunk_encoding == "fourier":
                return FourierFeatureEncoding(
                    trunk_input_dim, num_features=fourier_features, scale=fourier_scale
                )
            else:
                return nn.Identity()

        if dual_trunk:
            self.trunk_encoders = nn.ModuleList()
            self.trunks = nn.ModuleList()
            for _ in range(num_outputs):
                enc = _make_encoder()
                trunk_input_dim_eff = enc.output_dim if hasattr(enc, 'output_dim') else trunk_input_dim
                self.trunk_encoders.append(enc)
                trunk_layers = [trunk_input_dim_eff] + trunk_hidden + [self.p]
                self.trunks.append(mlp_cls(trunk_layers))
            self.trunk_encoder = None
            self.trunk = None
        else:
            self.trunk_encoder = _make_encoder()
            trunk_input_dim_eff = (self.trunk_encoder.output_dim
                                   if hasattr(self.trunk_encoder, 'output_dim')
                                   else trunk_input_dim)
            trunk_layers = [trunk_input_dim_eff] + trunk_hidden + [self.p]
            self.trunk = mlp_cls(trunk_layers)
            self.trunk_encoders = None
            self.trunks = None

        self.output_heads = (
            nn.ModuleList(
                [
                    OutputFeatureHead(
                        self.p,
                        hidden_dims=output_head_hidden,
                        activation=activation,
                        use_layer_norm=use_layer_norm,
                    )
                    for _ in range(num_outputs)
                ]
            )
            if use_output_heads else None
        )
        self.bias = nn.Parameter(torch.zeros(num_outputs))

    def _get_trunk_features(self, trunk_input: torch.Tensor) -> list[torch.Tensor]:
        """返回每个输出通道的 trunk 特征。"""
        if self.dual_trunk:
            return [
                self.trunks[k](self.trunk_encoders[k](trunk_input))
                for k in range(self.num_outputs)
            ]
        trunk_encoded = self.trunk_encoder(trunk_input)
        shared = self.trunk(trunk_encoded)
        return [shared] * self.num_outputs

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch_input)
        b = b.view(-1, self.num_outputs, self.p)

        trunk_feats = self._get_trunk_features(trunk_input)

        if self.output_heads is None:
            outputs = []
            for k in range(self.num_outputs):
                out_k = torch.einsum("np,mp->nm", b[:, k, :], trunk_feats[k]) + self.bias[k]
                outputs.append(out_k.unsqueeze(-1))
            return torch.cat(outputs, dim=-1)

        outputs = []
        for k in range(self.num_outputs):
            t_k = self.output_heads[k](trunk_feats[k])
            out_k = torch.einsum("np,mp->nm", b[:, k, :], t_k) + self.bias[k]
            outputs.append(out_k.unsqueeze(-1))

        return torch.cat(outputs, dim=-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict) -> MultiOutputDeepONet:
    """从配置字典创建模型。"""
    return MultiOutputDeepONet(
        branch_input_dim=cfg["branch_input_dim"],
        branch_hidden=cfg["branch_hidden"],
        trunk_input_dim=cfg["trunk_input_dim"],
        trunk_hidden=cfg["trunk_hidden"],
        trunk_output_dim=cfg["trunk_output_dim"],
        num_outputs=cfg["num_outputs"],
        activation=cfg["activation"],
        use_residual=cfg.get("use_residual", False),
        use_layer_norm=cfg.get("use_layer_norm", False),
        trunk_encoding=cfg.get("trunk_encoding", "none"),
        fourier_features=cfg.get("fourier_features", 32),
        fourier_scale=cfg.get("fourier_scale", 4.0),
        fourier_scales=cfg.get("fourier_scales", None),
        use_output_heads=cfg.get("use_output_heads", False),
        output_head_hidden=cfg.get("output_head_hidden", []),
        dual_trunk=cfg.get("dual_trunk", False),
    )
