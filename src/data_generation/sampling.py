"""
LHS 参数采样模块
基于 Carsel & Parrish (1988) 的 USDA 标准土壤参数范围
"""

import os

# Some login nodes have a tight process limit; cap BLAS threads before SciPy import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import qmc


SCALAR_PARAM_NAMES = [
    "theta_r", "theta_s", "alpha", "n_vg", "K_s",
    "D_L", "q_top", "c_top",
]

PARAM_NAMES = [
    "theta_r", "theta_s", "alpha", "n_vg", "K_s",
    "D_L", "q_top", "c_top",
]

PARAM_BOUNDS = {
    "theta_r": (0.034, 0.098, "uniform"),
    "theta_s": (0.36, 0.51, "uniform"),
    "alpha":   (0.005, 0.145, "log_uniform"),
    "n_vg":    (1.09, 2.68, "uniform"),
    "K_s":     (0.25, 29.7, "log_uniform"),
    "D_L":     (1.0, 20.0, "uniform"),
    "q_top":   (0.1, 5.0, "uniform"),
    "c_top":   (0.1, 1.0, "uniform"),
}


def q_top_function_enabled(cfg: dict | None) -> bool:
    if not cfg:
        return False
    return bool(cfg.get("q_top_function", {}).get("enabled", False))


def c_top_function_enabled(cfg: dict | None) -> bool:
    if not cfg:
        return False
    return bool(cfg.get("c_top_function", {}).get("enabled", False))


def variable_ic_enabled(cfg: dict | None) -> bool:
    if not cfg:
        return False
    return bool(cfg.get("variable_ic", {}).get("enabled", False))


def get_q_top_n_steps(cfg: dict) -> int:
    qcfg = cfg.get("q_top_function", {})
    default_steps = int(round(float(cfg["physics"]["simulation_time"])))
    return int(qcfg.get("n_steps", default_steps))


def get_c_top_n_steps(cfg: dict) -> int:
    ccfg = cfg.get("c_top_function", {})
    default_steps = int(round(float(cfg["physics"]["simulation_time"])))
    return int(ccfg.get("n_steps", default_steps))


def get_q_top_series_column_names(cfg: dict | None = None, n_steps: int | None = None) -> list[str]:
    if n_steps is None:
        if cfg is None:
            raise ValueError("Either cfg or n_steps must be provided")
        n_steps = get_q_top_n_steps(cfg)
    return [f"q_top_{i:02d}" for i in range(int(n_steps))]


def get_c_top_series_column_names(cfg: dict | None = None, n_steps: int | None = None) -> list[str]:
    if n_steps is None:
        if cfg is None:
            raise ValueError("Either cfg or n_steps must be provided")
        n_steps = get_c_top_n_steps(cfg)
    return [f"c_top_{i:02d}" for i in range(int(n_steps))]


def infer_q_top_series_column_names(columns) -> list[str]:
    q_cols = [
        str(col)
        for col in columns
        if str(col).startswith("q_top_") and str(col).split("_")[-1].isdigit()
    ]
    return sorted(q_cols, key=lambda name: int(name.split("_")[-1]))


def infer_c_top_series_column_names(columns) -> list[str]:
    c_cols = [
        str(col)
        for col in columns
        if str(col).startswith("c_top_") and str(col).split("_")[-1].isdigit()
    ]
    return sorted(c_cols, key=lambda name: int(name.split("_")[-1]))


def get_sampled_param_names(cfg: dict | None = None) -> list[str]:
    if q_top_function_enabled(cfg):
        names = ["theta_r", "theta_s", "alpha", "n_vg", "K_s", "D_L"]
        if not c_top_function_enabled(cfg):
            names.append("c_top")
        return names
    return list(SCALAR_PARAM_NAMES)


def rediscretize_forcing(
    series: np.ndarray,
    n_new_control_points: int,
    n_steps: int | None = None,
) -> np.ndarray:
    """Re-discretize a forcing time series through a different number of control points.

    Takes an existing n_steps-length series, resamples it at
    n_new_control_points evenly spaced locations, then re-interpolates back
    to n_steps.  This simulates "what if the forcing had been represented
    with a different resolution" while keeping the same branch dimension.

    Parameters
    ----------
    series : (n_steps,) or (N, n_steps) array
    n_new_control_points : int
    n_steps : int, optional (inferred from series if omitted)

    Returns
    -------
    Rediscretized series with same shape as input.
    """
    series = np.asarray(series, dtype=np.float64)
    single = series.ndim == 1
    if single:
        series = series[np.newaxis, :]
    if n_steps is None:
        n_steps = series.shape[1]

    old_idx = np.arange(n_steps, dtype=np.float64)
    ctrl_idx = np.linspace(0.0, float(n_steps - 1), n_new_control_points)
    ctrl_values = np.array([
        np.interp(ctrl_idx, old_idx, row) for row in series
    ])
    result = _interpolate_q_top_series(ctrl_values, n_steps=n_steps)
    return result[0] if single else result


def _sample_from_bounds(unit_values: np.ndarray, name: str) -> np.ndarray:
    lo, hi, dist = PARAM_BOUNDS[name]
    if dist == "log_uniform":
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        return 10 ** (log_lo + unit_values * (log_hi - log_lo))
    return lo + unit_values * (hi - lo)


def _interpolate_q_top_series(control_values: np.ndarray, n_steps: int) -> np.ndarray:
    ctrl_idx = np.linspace(0.0, float(n_steps - 1), control_values.shape[1])
    full_idx = np.arange(n_steps, dtype=float)
    return np.vstack([
        np.interp(full_idx, ctrl_idx, row).astype(np.float64)
        for row in control_values
    ])


def _count_extra_lhs_dims(cfg: dict | None) -> tuple[int, list[str]]:
    """Count additional LHS dimensions beyond scalar params and q_top control points."""
    extra_dims = 0
    extra_names: list[str] = []
    if c_top_function_enabled(cfg):
        ccfg = cfg["c_top_function"]
        n_ctrl = int(ccfg.get("n_control_points", 6))
        extra_dims += n_ctrl
        extra_names.extend([f"_c_top_ctrl_{i}" for i in range(n_ctrl)])
    if variable_ic_enabled(cfg):
        extra_dims += 2
        extra_names.extend(["_h_init_unit", "_c_init_unit"])
    return extra_dims, extra_names


def generate_lhs_samples(n_samples: int, seed: int = 42, cfg: dict | None = None) -> pd.DataFrame:
    """用 Latin Hypercube Sampling 在参数空间中均匀采样。

    对数均匀参数先在 log 空间均匀采样，再变换回原始空间。
    支持 q_top(t), c_top(t) 函数输入和可变初始条件。
    """
    sampled_names = get_sampled_param_names(cfg)

    if not q_top_function_enabled(cfg) and not c_top_function_enabled(cfg) and not variable_ic_enabled(cfg):
        sampler = qmc.LatinHypercube(d=len(sampled_names), seed=seed)
        unit_samples = sampler.random(n=n_samples)
        samples = np.empty_like(unit_samples)
        for i, name in enumerate(sampled_names):
            samples[:, i] = _sample_from_bounds(unit_samples[:, i], name)
        return pd.DataFrame(samples, columns=sampled_names)

    n_q_ctrl = 0
    if q_top_function_enabled(cfg):
        qcfg = cfg["q_top_function"]
        n_q_ctrl = int(qcfg.get("n_control_points", 6))

    extra_dims, _ = _count_extra_lhs_dims(cfg)
    total_lhs_dims = len(sampled_names) + n_q_ctrl + extra_dims

    sampler = qmc.LatinHypercube(d=total_lhs_dims, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    data: dict[str, np.ndarray] = {}
    col_idx = 0
    for name in sampled_names:
        data[name] = _sample_from_bounds(unit_samples[:, col_idx], name)
        col_idx += 1

    forcing_dt = float(cfg["physics"]["simulation_time"]) / float(
        get_q_top_n_steps(cfg) if q_top_function_enabled(cfg) else 1
    )

    if q_top_function_enabled(cfg):
        qcfg = cfg["q_top_function"]
        n_steps = get_q_top_n_steps(cfg)
        q_top_min = float(qcfg.get("min_flux", PARAM_BOUNDS["q_top"][0]))
        q_top_max = float(qcfg.get("max_flux", PARAM_BOUNDS["q_top"][1]))

        ctrl_unit = unit_samples[:, col_idx:col_idx + n_q_ctrl]
        col_idx += n_q_ctrl
        ctrl_values = q_top_min + ctrl_unit * (q_top_max - q_top_min)
        q_top_series = _interpolate_q_top_series(ctrl_values, n_steps=n_steps)

        for i, col in enumerate(get_q_top_series_column_names(n_steps=n_steps)):
            data[col] = q_top_series[:, i]

        data["q_top_mean"] = q_top_series.mean(axis=1)
        data["q_top_peak"] = q_top_series.max(axis=1)
        data["q_top_std"] = q_top_series.std(axis=1)
        data["q_top_total"] = q_top_series.sum(axis=1) * forcing_dt
        data["q_top_peak_time"] = q_top_series.argmax(axis=1) * forcing_dt

    if c_top_function_enabled(cfg):
        ccfg = cfg["c_top_function"]
        n_c_steps = get_c_top_n_steps(cfg)
        n_c_ctrl = int(ccfg.get("n_control_points", 6))
        c_min = float(ccfg.get("min_conc", PARAM_BOUNDS["c_top"][0]))
        c_max = float(ccfg.get("max_conc", PARAM_BOUNDS["c_top"][1]))

        c_ctrl_unit = unit_samples[:, col_idx:col_idx + n_c_ctrl]
        col_idx += n_c_ctrl
        c_ctrl_values = c_min + c_ctrl_unit * (c_max - c_min)
        c_top_series = _interpolate_q_top_series(c_ctrl_values, n_steps=n_c_steps)

        for i, col in enumerate(get_c_top_series_column_names(n_steps=n_c_steps)):
            data[col] = c_top_series[:, i]

        data["c_top_mean"] = c_top_series.mean(axis=1)
        data["c_top_peak"] = c_top_series.max(axis=1)
        data["c_top_std"] = c_top_series.std(axis=1)
        c_forcing_dt = float(cfg["physics"]["simulation_time"]) / float(n_c_steps)
        data["c_top_total"] = c_top_series.sum(axis=1) * c_forcing_dt
        data["c_top_peak_time"] = c_top_series.argmax(axis=1) * c_forcing_dt

    if variable_ic_enabled(cfg):
        vic = cfg["variable_ic"]
        h_min = float(vic.get("h_init_min", -200.0))
        h_max = float(vic.get("h_init_max", -50.0))
        c_min_ic = float(vic.get("c_init_min", 0.0))
        c_max_ic = float(vic.get("c_init_max", 0.3))

        h_unit = unit_samples[:, col_idx]
        col_idx += 1
        c_unit = unit_samples[:, col_idx]
        col_idx += 1

        data["h_init"] = h_min + h_unit * (h_max - h_min)
        data["c_init"] = c_min_ic + c_unit * (c_max_ic - c_min_ic)

    ordered_cols = ["theta_r", "theta_s", "alpha", "n_vg", "K_s", "D_L"]
    if q_top_function_enabled(cfg):
        ordered_cols.extend(get_q_top_series_column_names(cfg=cfg))
    if c_top_function_enabled(cfg):
        ordered_cols.extend(get_c_top_series_column_names(cfg=cfg))
    if not c_top_function_enabled(cfg):
        ordered_cols.append("c_top")
    if variable_ic_enabled(cfg):
        ordered_cols.extend(["h_init", "c_init"])
    for summary in [
        "q_top_mean", "q_top_peak", "q_top_std", "q_top_total", "q_top_peak_time",
        "c_top_mean", "c_top_peak", "c_top_std", "c_top_total", "c_top_peak_time",
    ]:
        if summary in data:
            ordered_cols.append(summary)

    return pd.DataFrame({col: data[col] for col in ordered_cols})


def build_sample_params(row: pd.Series, cfg: dict | None = None) -> dict:
    """从参数表的一行恢复单个样本的物理参数字典。"""
    params = {
        "theta_r": float(row["theta_r"]),
        "theta_s": float(row["theta_s"]),
        "alpha": float(row["alpha"]),
        "n_vg": float(row["n_vg"]),
        "K_s": float(row["K_s"]),
        "D_L": float(row["D_L"]),
    }

    q_cols = infer_q_top_series_column_names(row.index)
    if q_cols:
        q_top_series = np.asarray([float(row[col]) for col in q_cols], dtype=float)
        params["q_top_series"] = q_top_series.tolist()
        params["q_top"] = float(np.mean(q_top_series))
    else:
        params["q_top"] = float(row["q_top"])

    c_cols = infer_c_top_series_column_names(row.index)
    if c_cols:
        c_top_series = np.asarray([float(row[col]) for col in c_cols], dtype=float)
        params["c_top_series"] = c_top_series.tolist()
        params["c_top"] = float(np.mean(c_top_series))
    else:
        params["c_top"] = float(row["c_top"])

    if "h_init" in row.index:
        params["h_init"] = float(row["h_init"])
    if "c_init" in row.index:
        params["c_init"] = float(row["c_init"])

    return params


def split_dataset(
    df: pd.DataFrame,
    n_train: int = 600,
    n_val: int = 100,
    n_test: int = 100,
    seed: int = 42,
    split_mode: str = "iid",
    ood_parameter: str = "q_top",
    ood_quantile: float = 0.9,
    ood_tail: str = "high",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """划分训练/验证/测试集，支持 IID 与单参数 OOD。"""
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["split"] = ""
    df["split_mode"] = split_mode
    df["is_ood"] = False

    if split_mode == "iid":
        indices = rng.permutation(len(df))
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val : n_train + n_val + n_test]
    elif split_mode == "ood":
        if ood_parameter not in df.columns:
            raise KeyError(f"OOD parameter '{ood_parameter}' not in dataframe columns")
        if ood_tail not in {"high", "low"}:
            raise ValueError("ood_tail must be 'high' or 'low'")

        threshold = df[ood_parameter].quantile(ood_quantile)
        if ood_tail == "high":
            ood_mask = df[ood_parameter] >= threshold
        else:
            ood_mask = df[ood_parameter] <= threshold

        ood_candidates = df.index[ood_mask].to_numpy()
        id_candidates = df.index[~ood_mask].to_numpy()

        if len(ood_candidates) < n_test:
            raise ValueError(
                f"Not enough OOD candidates for test split: need {n_test}, got {len(ood_candidates)}"
            )
        if len(id_candidates) < n_train + n_val:
            raise ValueError(
                f"Not enough in-distribution candidates: need {n_train + n_val}, got {len(id_candidates)}"
            )

        test_idx = rng.choice(ood_candidates, size=n_test, replace=False)
        id_perm = rng.permutation(id_candidates)
        train_idx = id_perm[:n_train]
        val_idx = id_perm[n_train : n_train + n_val]

        df["ood_parameter"] = ood_parameter
        df["ood_quantile"] = float(ood_quantile)
        df["ood_tail"] = ood_tail
        df.loc[test_idx, "is_ood"] = True
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]
