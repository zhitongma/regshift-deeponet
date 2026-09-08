"""
FNN Baseline: 标准全连接网络，直接从 [params, z, t] 映射到 [h, c]。

与 DeepONet 的区别: 没有 branch/trunk 分离，不学习算子结构，
而是把参数和坐标拼接成单一输入向量进行回归。
"""

import torch
import torch.nn as nn


class FNNBaseline(nn.Module):

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: list[int] | None = None,
        num_outputs: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512, 512]

        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_cls())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_outputs))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        """保持与 DeepONet 相同的调用接口。

        Parameters
        ----------
        branch_input : (N, branch_dim)  参数向量
        trunk_input  : (M, 2)           坐标 (z, t)

        Returns
        -------
        output : (N, M, 2)
        """
        N = branch_input.shape[0]
        M = trunk_input.shape[0]

        params_expanded = branch_input.unsqueeze(1).expand(N, M, -1)
        coords_expanded = trunk_input.unsqueeze(0).expand(N, M, -1)
        x = torch.cat([params_expanded, coords_expanded], dim=-1)

        out = self.net(x.reshape(N * M, -1))
        return out.reshape(N, M, -1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_fnn_model(cfg: dict) -> FNNBaseline:
    branch_dim = cfg.get("branch_input_dim", 10)
    trunk_dim = cfg.get("trunk_input_dim", 2)
    fnn_cfg = cfg.get("fnn", {})
    return FNNBaseline(
        input_dim=branch_dim + trunk_dim,
        hidden_dims=fnn_cfg.get("hidden_dims", [512, 512, 512, 512]),
        num_outputs=cfg.get("num_outputs", 2),
        activation=cfg.get("activation", "gelu"),
    )
