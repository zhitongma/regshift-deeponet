"""
归一化工具函数 (与 postprocess.DataScaler 配合)
"""

import numpy as np


def vg_theta(h: np.ndarray, theta_r: float, theta_s: float,
             alpha: float, n: float) -> np.ndarray:
    """van Genuchten 水分特征曲线: h -> theta (向量化)。"""
    m = 1.0 - 1.0 / n
    Se = np.where(h >= 0, 1.0, 1.0 / (1.0 + np.abs(alpha * h) ** n) ** m)
    return theta_r + (theta_s - theta_r) * Se


def vg_theta_torch(h, theta_r, theta_s, alpha, n):
    """PyTorch 版 VG 水分特征曲线。"""
    import torch
    m = 1.0 - 1.0 / n
    Se = torch.where(h >= 0, torch.ones_like(h),
                     1.0 / (1.0 + torch.abs(alpha * h) ** n) ** m)
    return theta_r + (theta_s - theta_r) * Se


def vg_Se_torch(h, alpha, n):
    """PyTorch 版有效饱和度 Se(h), 保留计算图用于自动微分。"""
    import torch
    m = 1.0 - 1.0 / n
    abs_ah = torch.abs(alpha * h)
    Se = torch.where(h >= 0, torch.ones_like(h),
                     1.0 / (1.0 + abs_ah ** n) ** m)
    return Se


def vg_K_torch(h, K_s, alpha, n):
    """Mualem-van Genuchten 非饱和导水率 K(h)。"""
    import torch
    m = 1.0 - 1.0 / n
    Se = vg_Se_torch(h, alpha, n)
    Se_safe = torch.clamp(Se, 1e-8, 1.0)
    inner = 1.0 - (1.0 - Se_safe ** (1.0 / m)) ** m
    return K_s * Se_safe.sqrt() * inner ** 2
