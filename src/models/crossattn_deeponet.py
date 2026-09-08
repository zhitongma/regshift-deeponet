"""
CrossAttn-DeepONet: 用多头交叉注意力替换 DeepONet 的双线性内积。

标准 DeepONet: output = sum(b_k * phi_k)  (固定双线性交互)
CrossAttn:     output = CrossAttention(Q=trunk, K=V=branch_tokens)

让每个时空位置自适应地加权参数信息, 增强 branch-trunk 交互的表达力。
"""

import math
import torch
import torch.nn as nn

from src.models.deeponet import ResMLP, FourierFeatureEncoding


class CrossAttnDeepONet(nn.Module):
    """DeepONet with cross-attention branch-trunk interaction.

    Branch output 被投影为 n_tokens 个 token, 作为 K/V;
    Trunk output 作为 Q; 多头交叉注意力产生输出。
    """

    def __init__(
        self,
        branch_input_dim: int = 10,
        branch_hidden: list[int] | None = None,
        trunk_input_dim: int = 2,
        trunk_hidden: list[int] | None = None,
        d_model: int = 256,
        n_heads: int = 4,
        n_tokens: int = 16,
        n_attn_layers: int = 2,
        num_outputs: int = 2,
        activation: str = "gelu",
        fourier_features: int = 32,
        fourier_scale: float = 4.0,
    ):
        super().__init__()
        if branch_hidden is None:
            branch_hidden = [512, 512, 512]
        if trunk_hidden is None:
            trunk_hidden = [256, 256, 256]

        self.d_model = d_model
        self.n_tokens = n_tokens
        self.num_outputs = num_outputs

        self.branch = ResMLP(
            [branch_input_dim] + branch_hidden + [n_tokens * d_model],
            activation, use_layer_norm=True,
        )

        self.trunk_encoder = FourierFeatureEncoding(
            trunk_input_dim, num_features=fourier_features, scale=fourier_scale,
        )
        self.trunk = ResMLP(
            [self.trunk_encoder.output_dim] + trunk_hidden + [d_model],
            activation, use_layer_norm=True,
        )

        self.attn_layers = nn.ModuleList()
        self.attn_norms_q = nn.ModuleList()
        self.attn_norms_kv = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            )
            self.attn_norms_q.append(nn.LayerNorm(d_model))
            self.attn_norms_kv.append(nn.LayerNorm(d_model))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            ))
            self.ffn_norms.append(nn.LayerNorm(d_model))

        self.output_head = nn.Linear(d_model, num_outputs)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch_input: torch.Tensor,
                trunk_input: torch.Tensor) -> torch.Tensor:
        N = branch_input.shape[0]
        M = trunk_input.shape[0]

        branch_tokens = self.branch(branch_input)
        branch_tokens = branch_tokens.view(N, self.n_tokens, self.d_model)

        trunk_encoded = self.trunk_encoder(trunk_input)
        trunk_feats = self.trunk(trunk_encoded)
        trunk_feats = trunk_feats.unsqueeze(0).expand(N, M, self.d_model)

        x = trunk_feats
        for attn, norm_q, norm_kv, ffn, ffn_norm in zip(
            self.attn_layers, self.attn_norms_q, self.attn_norms_kv,
            self.ffn_layers, self.ffn_norms,
        ):
            q = norm_q(x)
            kv = norm_kv(branch_tokens)
            attn_out, _ = attn(q, kv, kv)
            x = x + attn_out
            x = x + ffn(ffn_norm(x))

        output = self.output_head(x)
        return output

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_crossattn_deeponet(cfg: dict) -> CrossAttnDeepONet:
    ca_cfg = cfg.get("crossattn_deeponet", {})
    return CrossAttnDeepONet(
        branch_input_dim=cfg.get("branch_input_dim", 10),
        branch_hidden=ca_cfg.get("branch_hidden", [512, 512, 512]),
        trunk_input_dim=cfg.get("trunk_input_dim", 2),
        trunk_hidden=ca_cfg.get("trunk_hidden", [256, 256, 256]),
        d_model=ca_cfg.get("d_model", 256),
        n_heads=ca_cfg.get("n_heads", 4),
        n_tokens=ca_cfg.get("n_tokens", 16),
        n_attn_layers=ca_cfg.get("n_attn_layers", 2),
        num_outputs=cfg.get("num_outputs", 2),
        activation=cfg.get("activation", "gelu"),
        fourier_features=cfg.get("fourier_features", 32),
        fourier_scale=cfg.get("fourier_scale", 4.0),
    )
