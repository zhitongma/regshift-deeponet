"""
数据后处理与质量控制

功能:
  1. 从 raw/*.npz 中加载原始 HYDRUS 输出
  2. 质量控制: 过滤不合格的模拟
  3. 归一化: 输入 min-max, 输出零均值单位方差
  4. 按 train/val/test 划分并保存
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# 质量控制
# -----------------------------------------------------------------------

def check_quality(
    h: np.ndarray,
    c: np.ndarray,
    theta: np.ndarray,
    params: dict,
    threshold: float = 0.01,
    water_balance_rel: np.ndarray | None = None,
    solute_balance_rel: np.ndarray | None = None,
) -> tuple[bool, str]:
    """对单个模拟执行质量控制检查。

    检查项:
      1. h/c 无 NaN（数据完整性）
      2. theta 无 NaN 且在物理范围 [theta_r, theta_s] 内
      3. 浓度非负
      4. HYDRUS 自带质量平衡误差不过大

    Returns
    -------
    (passed, reason)
    """
    # 1. NaN 检查 — h 和 c 必须完整
    if np.any(np.isnan(h)) or np.any(np.isnan(c)):
        n_nan_h = np.isnan(h).sum()
        n_nan_c = np.isnan(c).sum()
        return False, f"contains NaN (h:{n_nan_h}, c:{n_nan_c})"

    tr = params["theta_r"]
    ts = params["theta_s"]

    # 2. 含水率: 无 NaN + 物理范围
    if theta is not None and not np.all(np.isnan(theta)):
        if np.any(np.isnan(theta)):
            return False, f"theta contains NaN ({np.isnan(theta).sum()} values)"
        theta_min, theta_max = np.nanmin(theta), np.nanmax(theta)
        if theta_min < tr - 0.01 or theta_max > ts + 0.01:
            return False, f"theta out of range [{theta_min:.4f}, {theta_max:.4f}]"

    # 3. 浓度非负
    if np.any(c < -1e-6):
        return False, f"negative concentration (min={c.min():.6f})"

    # 4. 使用 HYDRUS 自身输出的平衡误差做 QC
    balance_limit = threshold * 100.0  # 配置是小数, HYDRUS 输出是百分比
    if water_balance_rel is not None and len(water_balance_rel) > 1:
        water_max = float(np.nanmax(np.abs(water_balance_rel[1:])))
        if water_max > balance_limit:
            return False, f"water balance error too large ({water_max:.3f}% > {balance_limit:.3f}%)"

    if solute_balance_rel is not None and len(solute_balance_rel) > 1:
        solute_max = float(np.nanmax(np.abs(solute_balance_rel[1:])))
        if solute_max > balance_limit:
            return False, f"solute balance error too large ({solute_max:.3f}% > {balance_limit:.3f}%)"

    return True, "ok"


# -----------------------------------------------------------------------
# 数据加载与整合
# -----------------------------------------------------------------------

def load_raw_data(raw_dir: str | Path) -> list[dict]:
    """加载所有 raw/*.npz 文件。"""
    raw_dir = Path(raw_dir)
    samples = []

    for npz_path in sorted(raw_dir.glob("sample_*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        params = json.loads(str(data["params"]))
        samples.append(dict(
            path=str(npz_path),
            sample_id=int(npz_path.stem.split("_")[1]),
            h=data["h"],
            c=data["c"],
            theta=data["theta"],
            z=data["z"],
            t=data["t"],
            water_flux_top=data["water_flux_top"] if "water_flux_top" in data.files else np.zeros_like(data["t"]),
            water_flux_bottom=data["water_flux_bottom"] if "water_flux_bottom" in data.files else np.zeros_like(data["t"]),
            water_cum_top=data["water_cum_top"] if "water_cum_top" in data.files else np.zeros_like(data["t"]),
            water_cum_bottom=data["water_cum_bottom"] if "water_cum_bottom" in data.files else np.zeros_like(data["t"]),
            water_cum_net=data["water_cum_net"] if "water_cum_net" in data.files else np.zeros_like(data["t"]),
            solute_flux_top=data["solute_flux_top"] if "solute_flux_top" in data.files else np.zeros_like(data["t"]),
            solute_flux_bottom=data["solute_flux_bottom"] if "solute_flux_bottom" in data.files else np.zeros_like(data["t"]),
            solute_cum_top=data["solute_cum_top"] if "solute_cum_top" in data.files else np.zeros_like(data["t"]),
            solute_cum_bottom=data["solute_cum_bottom"] if "solute_cum_bottom" in data.files else np.zeros_like(data["t"]),
            solute_cum_net=data["solute_cum_net"] if "solute_cum_net" in data.files else np.zeros_like(data["t"]),
            water_balance_rel=data["water_balance_rel"] if "water_balance_rel" in data.files else np.zeros_like(data["t"]),
            solute_balance_rel=data["solute_balance_rel"] if "solute_balance_rel" in data.files else np.zeros_like(data["t"]),
            params=params,
        ))

    logger.info("加载了 %d 个原始样本", len(samples))
    return samples


def run_quality_control(samples: list[dict],
                        threshold: float = 0.01,
                        balance_stride: int = 1) -> tuple[list[dict], list[dict]]:
    """对所有样本执行 QC，返回 (合格, 不合格)。"""
    passed, failed = [], []
    balance_stride = max(1, int(balance_stride))
    for s in samples:
        water_balance_rel = s.get("water_balance_rel")
        solute_balance_rel = s.get("solute_balance_rel")
        if water_balance_rel is not None and balance_stride > 1:
            water_balance_rel = np.asarray(water_balance_rel)[::balance_stride]
        if solute_balance_rel is not None and balance_stride > 1:
            solute_balance_rel = np.asarray(solute_balance_rel)[::balance_stride]
        ok, reason = check_quality(
            s["h"],
            s["c"],
            s["theta"],
            s["params"],
            threshold,
            water_balance_rel=water_balance_rel,
            solute_balance_rel=solute_balance_rel,
        )
        if ok:
            passed.append(s)
        else:
            failed.append({**s, "fail_reason": reason})
            logger.warning("样本 %d 不合格: %s", s["sample_id"], reason)

    logger.info("QC 结果: %d 合格, %d 不合格 (有效率 %.1f%%)",
                len(passed), len(failed),
                100 * len(passed) / max(len(passed) + len(failed), 1))
    return passed, failed


# -----------------------------------------------------------------------
# 归一化
# -----------------------------------------------------------------------

class DataScaler:
    """输入 min-max 归一化, 输出零均值单位方差。"""

    def __init__(self):
        self.input_min = None
        self.input_max = None
        self.h_mean = None
        self.h_std = None
        self.c_mean = None
        self.c_std = None

    def fit(self, params_array: np.ndarray, h_all: np.ndarray, c_all: np.ndarray):
        self.input_min = params_array.min(axis=0)
        self.input_max = params_array.max(axis=0)

        self.h_mean = h_all.mean()
        self.h_std = h_all.std() + 1e-8
        self.c_mean = c_all.mean()
        self.c_std = c_all.std() + 1e-8

    def transform_input(self, params_array: np.ndarray) -> np.ndarray:
        return (params_array - self.input_min) / (self.input_max - self.input_min + 1e-8)

    def transform_h(self, h: np.ndarray) -> np.ndarray:
        return (h - self.h_mean) / self.h_std

    def transform_c(self, c: np.ndarray) -> np.ndarray:
        return (c - self.c_mean) / self.c_std

    def inverse_h(self, h_norm: np.ndarray) -> np.ndarray:
        return h_norm * self.h_std + self.h_mean

    def inverse_c(self, c_norm: np.ndarray) -> np.ndarray:
        return c_norm * self.c_std + self.c_mean

    def save(self, path: str | Path):
        np.savez(
            path,
            input_min=self.input_min, input_max=self.input_max,
            h_mean=self.h_mean, h_std=self.h_std,
            c_mean=self.c_mean, c_std=self.c_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DataScaler":
        data = np.load(path)
        scaler = cls()
        scaler.input_min = data["input_min"]
        scaler.input_max = data["input_max"]
        scaler.h_mean = float(data["h_mean"])
        scaler.h_std = float(data["h_std"])
        scaler.c_mean = float(data["c_mean"])
        scaler.c_std = float(data["c_std"])
        return scaler


# -----------------------------------------------------------------------
# 组装最终数据集
# -----------------------------------------------------------------------

LEGACY_PARAM_KEYS = ["theta_r", "theta_s", "alpha", "n_vg", "K_s",
                     "D_L", "q_top", "c_top"]

# 报告定义: h_init 和 c_init 作为固定常数输入 branch
BRANCH_KEYS = LEGACY_PARAM_KEYS + ["h_init", "c_init"]


def get_param_keys(params_df: pd.DataFrame) -> list[str]:
    """根据参数表自动推断 branch 中的物理输入顺序。

    Handles: scalar params, q_top(t) series, c_top(t) series.
    Excludes: derived summary columns, h_init/c_init (handled by get_branch_keys).
    """
    q_top_cols = sorted(
        [
            str(col)
            for col in params_df.columns
            if str(col).startswith("q_top_") and str(col).split("_")[-1].isdigit()
        ],
        key=lambda name: int(name.split("_")[-1]),
    )
    c_top_cols = sorted(
        [
            str(col)
            for col in params_df.columns
            if str(col).startswith("c_top_") and str(col).split("_")[-1].isdigit()
        ],
        key=lambda name: int(name.split("_")[-1]),
    )

    keys = ["theta_r", "theta_s", "alpha", "n_vg", "K_s", "D_L"]
    if q_top_cols:
        keys.extend(q_top_cols)
    else:
        keys.append("q_top")
    if c_top_cols:
        keys.extend(c_top_cols)
    elif "c_top" in params_df.columns:
        keys.append("c_top")

    return keys


def get_branch_keys(params_df: pd.DataFrame) -> list[str]:
    return get_param_keys(params_df) + ["h_init", "c_init"]


def build_dataset(samples: list[dict], params_df: pd.DataFrame,
                  scaler: DataScaler | None = None, fit_scaler: bool = False):
    """将样本列表转为 DeepONet 训练所需的数组格式。

    Returns
    -------
    dict with keys:
        branch_inputs: (N, branch_dim) -- 归一化后的参数
        trunk_inputs:  (n_z * n_t, 2) -- (z, t) 网格点
        h_targets:     (N, n_z * n_t) -- h 值 (展平)
        c_targets:     (N, n_z * n_t) -- c 值 (展平)
        scaler:        DataScaler
        z, t:          坐标数组
    """
    n = len(samples)
    n_z = samples[0]["h"].shape[0]
    n_t = samples[0]["h"].shape[1]
    z = samples[0]["z"]
    t = samples[0]["t"]

    param_keys = get_param_keys(params_df)
    branch_keys = param_keys + ["h_init", "c_init"]

    # Branch 输入: 动态参数向量 + h_init + c_init
    branch = np.zeros((n, len(branch_keys)))
    h_all = np.zeros((n, n_z, n_t))
    c_all = np.zeros((n, n_z, n_t))
    water_flux_top = np.zeros((n, n_t))
    water_flux_bottom = np.zeros((n, n_t))
    water_cum_top = np.zeros((n, n_t))
    water_cum_bottom = np.zeros((n, n_t))
    water_cum_net = np.zeros((n, n_t))
    solute_flux_top = np.zeros((n, n_t))
    solute_flux_bottom = np.zeros((n, n_t))
    solute_cum_top = np.zeros((n, n_t))
    solute_cum_bottom = np.zeros((n, n_t))
    solute_cum_net = np.zeros((n, n_t))
    water_balance_rel = np.zeros((n, n_t))
    solute_balance_rel = np.zeros((n, n_t))

    for i, s in enumerate(samples):
        param_row = params_df.loc[s["sample_id"]]
        for j, key in enumerate(param_keys):
            branch[i, j] = float(param_row[key])
        branch[i, len(param_keys)] = s["h"][:, 0].mean()       # h_init
        branch[i, len(param_keys) + 1] = s["c"][:, 0].mean()   # c_init
        h_all[i] = s["h"]
        c_all[i] = s["c"]
        water_flux_top[i] = s["water_flux_top"]
        water_flux_bottom[i] = s["water_flux_bottom"]
        water_cum_top[i] = s["water_cum_top"]
        water_cum_bottom[i] = s["water_cum_bottom"]
        water_cum_net[i] = s["water_cum_net"]
        solute_flux_top[i] = s["solute_flux_top"]
        solute_flux_bottom[i] = s["solute_flux_bottom"]
        solute_cum_top[i] = s["solute_cum_top"]
        solute_cum_bottom[i] = s["solute_cum_bottom"]
        solute_cum_net[i] = s["solute_cum_net"]
        water_balance_rel[i] = s["water_balance_rel"]
        solute_balance_rel[i] = s["solute_balance_rel"]

    # Trunk 输入: (z, t) 笛卡尔积
    zz, tt = np.meshgrid(z, t, indexing="ij")  # (n_z, n_t)
    trunk = np.column_stack([zz.ravel(), tt.ravel()])  # (n_z*n_t, 2)

    # 归一化
    if scaler is None:
        scaler = DataScaler()
    if fit_scaler:
        scaler.fit(branch, h_all, c_all)

    branch_norm = scaler.transform_input(branch)
    h_flat = scaler.transform_h(h_all.reshape(n, -1))
    c_flat = scaler.transform_c(c_all.reshape(n, -1))

    # trunk 坐标也归一化到 [0,1]
    trunk_norm = trunk.copy()
    trunk_norm[:, 0] /= z.max()  # z / L
    trunk_norm[:, 1] /= t.max()  # t / T

    return dict(
        branch_inputs=branch_norm.astype(np.float32),
        trunk_inputs=trunk_norm.astype(np.float32),
        h_targets=h_flat.astype(np.float32),
        c_targets=c_flat.astype(np.float32),
        h_raw=h_all.reshape(n, -1).astype(np.float32),
        c_raw=c_all.reshape(n, -1).astype(np.float32),
        water_flux_top=water_flux_top.astype(np.float32),
        water_flux_bottom=water_flux_bottom.astype(np.float32),
        water_cum_top=water_cum_top.astype(np.float32),
        water_cum_bottom=water_cum_bottom.astype(np.float32),
        water_cum_net=water_cum_net.astype(np.float32),
        solute_flux_top=solute_flux_top.astype(np.float32),
        solute_flux_bottom=solute_flux_bottom.astype(np.float32),
        solute_cum_top=solute_cum_top.astype(np.float32),
        solute_cum_bottom=solute_cum_bottom.astype(np.float32),
        solute_cum_net=solute_cum_net.astype(np.float32),
        water_balance_rel=water_balance_rel.astype(np.float32),
        solute_balance_rel=solute_balance_rel.astype(np.float32),
        param_keys=np.asarray(param_keys, dtype=str),
        branch_keys=np.asarray(branch_keys, dtype=str),
        scaler=scaler,
        z=z, t=t,
    )
