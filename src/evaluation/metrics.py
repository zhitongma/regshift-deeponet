"""
评估指标体系

B1: ML 精度指标 (相对 L2, R^2, 最大绝对误差)
B2: 守恒性指标 (水量/溶质质量平衡误差)
B3: 水文学过程指标 (锋面位置, 穿透时间)
B4: 计算效率指标
"""

import numpy as np
from scipy.interpolate import interp1d


# -----------------------------------------------------------------------
# B1: ML 精度指标
# -----------------------------------------------------------------------

def relative_l2_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """相对 L2 误差: ||pred - ref||_2 / ||ref||_2。"""
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12)


def r_squared(pred: np.ndarray, ref: np.ndarray) -> float:
    """决定系数 R^2。"""
    ss_res = np.sum((pred - ref) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def max_abs_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """逐点最大绝对误差。"""
    return float(np.max(np.abs(pred - ref)))


def compute_ml_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """计算一组样本的全部 ML 指标。

    Parameters
    ----------
    pred, ref : (n_z, n_t) or (N, n_z*n_t)
    """
    return {
        "rel_l2": relative_l2_error(pred, ref),
        "r2": r_squared(pred.ravel(), ref.ravel()),
        "max_abs": max_abs_error(pred, ref),
    }


def compute_ml_metrics_batch(preds: np.ndarray, refs: np.ndarray) -> dict:
    """在测试集上批量计算指标, 返回统计分布。

    Parameters
    ----------
    preds, refs : (N, n_z*n_t)
    """
    N = preds.shape[0]
    metrics = {"rel_l2": [], "r2": [], "max_abs": []}

    for i in range(N):
        m = compute_ml_metrics(preds[i], refs[i])
        for k in metrics:
            metrics[k].append(m[k])

    stats = {}
    for k, vals in metrics.items():
        vals = np.array(vals)
        stats[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "p5": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
            "values": vals.tolist(),
        }
    return stats


# -----------------------------------------------------------------------
# B2: 守恒性指标
# -----------------------------------------------------------------------

def _mass_balance_error(delta_storage: np.ndarray, cum_target: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(cum_target), 1.0)
    mbe = np.abs(delta_storage - cum_target) / denom * 100.0
    mbe[0] = 0.0
    return mbe


def mass_balance_error_water(
    theta: np.ndarray,
    water_cum_target: np.ndarray,
    dz: float = 1.0,
) -> np.ndarray:
    """水量平衡误差: 预测储量变化 vs 真实累计边界通量。"""
    storage = np.trapz(theta, dx=dz, axis=0)
    delta_s = storage - storage[0]
    return _mass_balance_error(delta_s, water_cum_target)


def mass_balance_error_solute(
    theta: np.ndarray,
    c: np.ndarray,
    solute_cum_target: np.ndarray,
    dz: float = 1.0,
) -> np.ndarray:
    """溶质质量平衡误差: 预测储量变化 vs 真实累计边界通量。"""
    storage = np.trapz(theta * c, dx=dz, axis=0)
    delta_sc = storage - storage[0]
    return _mass_balance_error(delta_sc, solute_cum_target)


# -----------------------------------------------------------------------
# B3: 水文学过程指标
# -----------------------------------------------------------------------

def find_front_position(
    field: np.ndarray, z: np.ndarray, threshold: float,
) -> float | None:
    """在深度方向查找场值等于阈值的位置 (线性插值)。

    Parameters
    ----------
    field : (n_z,)  某时刻的 theta 或 c 剖面
    z : (n_z,)
    threshold : 阈值

    Returns
    -------
    z_front : 锋面位置 [cm], 或 None (未到达)
    """
    above = field >= threshold
    if np.all(above) or not np.any(above):
        return None

    for i in range(len(field) - 1):
        if (field[i] >= threshold) != (field[i + 1] >= threshold):
            frac = (threshold - field[i]) / (field[i + 1] - field[i] + 1e-12)
            return z[i] + frac * (z[i + 1] - z[i])
    return None


def wetting_front_position(
    theta: np.ndarray, z: np.ndarray, t: np.ndarray,
    theta_init: float, theta_s: float,
) -> np.ndarray:
    """入渗锋位置: z_f(t) where theta = theta_init + 0.5*(theta_s - theta_init)。

    Returns
    -------
    z_front : (n_t,)  锋面深度, NaN 表示未到达
    """
    threshold = theta_init + 0.5 * (theta_s - theta_init)
    n_t = theta.shape[1]
    z_front = np.full(n_t, np.nan)

    for j in range(n_t):
        pos = find_front_position(theta[:, j], z, threshold)
        if pos is not None:
            z_front[j] = pos

    return z_front


def concentration_front_position(
    c: np.ndarray, z: np.ndarray, c_top: float,
) -> np.ndarray:
    """浓度锋位置: z_c(t) where c = 0.5 * c_top。"""
    threshold = 0.5 * c_top
    n_t = c.shape[1]
    z_front = np.full(n_t, np.nan)

    for j in range(n_t):
        pos = find_front_position(c[:, j], z, threshold)
        if pos is not None:
            z_front[j] = pos

    return z_front


def breakthrough_time(
    c: np.ndarray, t: np.ndarray, c_top: float, c_th_frac: float = 0.05,
) -> float:
    """穿透时间: t_bt = min{t : c(L, t) >= c_th}。

    Parameters
    ----------
    c : (n_z, n_t)
    t : (n_t,)
    c_top : 上边界浓度
    c_th_frac : 穿透阈值 (c_top 的比例)

    Returns
    -------
    t_bt : 穿透时间 [h], 或 np.inf (未穿透)
    """
    c_bottom = c[-1, :]  # 土柱底部浓度
    c_th = c_th_frac * c_top
    idx = np.where(c_bottom >= c_th)[0]
    if len(idx) == 0:
        return np.inf
    return float(t[idx[0]])


# -----------------------------------------------------------------------
# B4: 计算效率指标
# -----------------------------------------------------------------------

def cost_crossover(
    t_data_gen: float, t_train: float, t_infer: float, t_hydrus: float,
) -> float:
    """计算成本交叉点 N_cross。

    N_cross = (t_data_gen + t_train) / t_hydrus

    超过这个场景数, DeepONet 更划算。
    """
    return (t_data_gen + t_train) / (t_hydrus + 1e-12)


def total_cost_curves(
    t_data_gen: float, t_train: float, t_infer: float, t_hydrus: float,
    n_range: np.ndarray = None,
) -> dict:
    """生成总计算时间 vs 场景数量的曲线数据。"""
    if n_range is None:
        n_range = np.logspace(0, 4, 100).astype(int)
        n_range = np.unique(n_range)

    cost_hydrus = t_hydrus * n_range
    cost_deeponet = t_data_gen + t_train + t_infer * n_range

    n_cross = cost_crossover(t_data_gen, t_train, t_infer, t_hydrus)

    return {
        "n_range": n_range.tolist(),
        "cost_hydrus": cost_hydrus.tolist(),
        "cost_deeponet": cost_deeponet.tolist(),
        "n_cross": float(n_cross),
    }
