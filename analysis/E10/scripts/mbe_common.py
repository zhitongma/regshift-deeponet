#!/usr/bin/env python3
"""
E10 公共工具（numpy-only，无 torch / pandas / scipy 依赖）。

- MBE 公式与 van Genuchten 映射均逐行照抄主仓库实现（出处见各函数 docstring），
  保证与论文 B2 指标定义完全一致。
- iCloud dataless 防护：主仓库位于 iCloud 同步盘，未物化文件读内容会永久挂起；
  所有读取前先用 `is_dataless()`（只 stat 元数据，安全）检查。
"""

from __future__ import annotations

import json
import os

import numpy as np

# numpy 1.x/2.x 兼容：np.trapz 在 numpy 2.0 更名为 np.trapezoid。
# 主仓库 metrics.py 用的是 np.trapz（集群 numpy 1.x）；此 shim 保证两端数值一致。
trapz = getattr(np, "trapezoid", None) or np.trapz

# macOS SF_DATALESS 标志位（iCloud 未物化占位文件）
SF_DATALESS = 0x40000000


def is_dataless(path) -> bool:
    """True = iCloud dataless 占位文件，读取内容会挂起。只 stat，不触发下载。"""
    try:
        st = os.stat(path)
    except OSError:
        return False  # 不存在等情况由调用方另行处理
    return bool(getattr(st, "st_flags", 0) & SF_DATALESS)


def safe_load_json(path):
    """带 dataless 防护的 json 读取。返回 (dict|None, err_msg|None)。"""
    if not os.path.exists(path):
        return None, "missing"
    if is_dataless(path):
        return None, "dataless (先 brctl download 物化，或在集群上跑)"
    with open(path) as f:
        return json.load(f), None


def safe_load_npz(path):
    """带 dataless 防护的 npz 读取。返回 (NpzFile|None, err_msg|None)。"""
    if not os.path.exists(path):
        return None, "missing"
    if is_dataless(path):
        return None, "dataless (先 brctl download 物化，或在集群上跑)"
    return np.load(path), None


# ---------------------------------------------------------------------------
# van Genuchten 映射 —— 照抄 src/utils/normalization.py:8-13 (vg_theta)
# ---------------------------------------------------------------------------

def vg_theta(h, theta_r, theta_s, alpha, n):
    """van Genuchten 水分特征曲线: h -> theta (向量化)。

    与主仓库 `src/utils/normalization.py::vg_theta` 逐行一致
    （B2 评估中 theta_pred 即由该函数从 h_pred 计算，06_evaluate.py:327-329）。
    """
    m = 1.0 - 1.0 / n
    Se = np.where(h >= 0, 1.0, 1.0 / (1.0 + np.abs(alpha * h) ** n) ** m)
    return theta_r + (theta_s - theta_r) * Se


# ---------------------------------------------------------------------------
# B2 的 MBE 定义 —— 照抄 src/evaluation/metrics.py:82-109
# ---------------------------------------------------------------------------

def _mass_balance_error(delta_storage, cum_target):
    """照抄 metrics.py:82-86：分母 max(|累计通量|, 1)，百分数，t0 置 0。"""
    denom = np.maximum(np.abs(cum_target), 1.0)
    mbe = np.abs(delta_storage - cum_target) / denom * 100.0
    mbe[0] = 0.0
    return mbe


def mass_balance_error_water(theta, water_cum_target, dz=1.0):
    """水量平衡误差 [%]（照抄 metrics.py:89-97）。

    theta: (n_z, n_t)；water_cum_target: (n_t,) 累计净边界通量（HYDRUS 参考）。
    返回 (n_t,) 曲线；论文表 5 用终点值 mbe[-1] 的样本均值。
    """
    storage = trapz(theta, dx=dz, axis=0)
    delta_s = storage - storage[0]
    return _mass_balance_error(delta_s, water_cum_target)


def mass_balance_error_solute(theta, c, solute_cum_target, dz=1.0):
    """溶质质量平衡误差 [%]（照抄 metrics.py:100-109）。"""
    storage = trapz(theta * c, dx=dz, axis=0)
    delta_sc = storage - storage[0]
    return _mass_balance_error(delta_sc, solute_cum_target)


# ---------------------------------------------------------------------------
# 相对 L2（照抄 metrics.py:18-20）与 branch 反归一化（照抄 06_evaluate.py:268-273）
# ---------------------------------------------------------------------------

def relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


BRANCH_KEYS_LEGACY = ["theta_r", "theta_s", "alpha", "n_vg", "K_s",
                      "D_L", "q_top", "c_top", "h_init", "c_init"]


def get_raw_params(test_data, scaler_npz):
    """反归一化测试集参数（照抄 06_evaluate.py::get_raw_params 的 min-max 逆变换）。"""
    branch_norm = test_data["branch_inputs"]
    input_min = scaler_npz["input_min"]
    input_max = scaler_npz["input_max"]
    branch_raw = branch_norm * (input_max - input_min + 1e-8) + input_min
    if "branch_keys" in test_data.files:
        keys = [str(x) for x in test_data["branch_keys"].tolist()]
    else:
        keys = list(BRANCH_KEYS_LEGACY)
    return {k: branch_raw[:, i] for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# 手写 Spearman 相关（置换检验 p 值，双侧）—— 本机无 scipy
# ---------------------------------------------------------------------------

def _rank_average_ties(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    vals = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return ranks


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman(x, y, n_perm=20000, seed=42):
    """Spearman rho + 双侧置换检验 p 值。返回 dict(n, rho, p_perm)。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 4:
        return {"n": n, "rho": float("nan"), "p_perm": float("nan")}
    rx = _rank_average_ties(x)
    ry = _rank_average_ties(y)
    rho = _pearson(rx, ry)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        r = _pearson(rx, rng.permutation(ry))
        if np.isfinite(r) and abs(r) >= abs(rho) - 1e-12:
            hits += 1
    p = (hits + 1.0) / (n_perm + 1.0)
    return {"n": n, "rho": rho, "p_perm": p}


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")
