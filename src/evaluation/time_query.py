"""
Helpers for dense-time query experiments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator

from src.models.deeponet import build_model
from src.models.fnn_baseline import build_fnn_model
from src.models.grid_fnn import build_grid_fnn_model
from src.models.fno import build_fno_model


def _build_deeponet_dispatch(model_cfg: dict):
    arch = model_cfg.get("arch", "deeponet")
    if arch == "shift_deeponet":
        from src.models.shift_deeponet import build_shift_deeponet
        return build_shift_deeponet(model_cfg)
    return build_model(model_cfg)


def load_checkpoint_name(model_name: str, ckpt_dir: Path) -> tuple[str, str]:
    if model_name == "m1":
        return "m1_best.pt", "direct"
    if model_name == "m2_data":
        return "m2_best_data.pt", "direct"
    if model_name == "m2_score":
        return "m2_best_score.pt", "direct"
    if model_name == "m2_phase2_score":
        return "m2_phase2_best_score.pt", "direct"
    if model_name == "m2":
        if (ckpt_dir / "m2_best_score.pt").exists():
            return "m2_best_score.pt", "direct"
        return "m2_best.pt", "direct"
    if model_name == "fnn":
        return "fnn_best.pt", "direct"
    if model_name == "fnn_mass":
        return "fnn_mass_best.pt", "direct"
    if model_name == "fnn_mass_score":
        return "fnn_mass_best_score.pt", "direct"
    if model_name == "grid_fnn":
        return "grid_fnn_best.pt", "grid_interp"
    if model_name == "fno":
        return "fno_best.pt", "grid_interp"
    raise ValueError(f"Unsupported model: {model_name}")


def build_query_model(model_name: str, cfg: dict, ckpt_dir: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    ckpt_name, mode = load_checkpoint_name(model_name, ckpt_dir)
    ckpt_path = ckpt_dir / ckpt_name
    if mode == "direct":
        if model_name in ("fnn", "fnn_mass", "fnn_mass_score"):
            model = build_fnn_model(cfg["model"]).to(device)
        else:
            model = _build_deeponet_dispatch(cfg["model"]).to(device)
    elif mode == "grid_interp":
        if model_name == "fno":
            model_cfg = {**cfg["model"], "fno": cfg.get("fno", {}), "physics": cfg.get("physics", {})}
            model = build_fno_model(model_cfg).to(device)
        else:
            model_cfg = {**cfg["model"], "grid_fnn": cfg.get("grid_fnn", {}), "physics": cfg.get("physics", {})}
            model = build_grid_fnn_model(model_cfg).to(device)
    else:
        raise ValueError(f"Unsupported query mode: {mode}")

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, mode


def build_trunk_query_zt(
    z_query: np.ndarray,
    t_query: np.ndarray,
    z_max: float,
    t_max: float,
) -> torch.Tensor:
    """Build trunk input from arrays of z and t coordinates.

    Both z_query and t_query are in physical units (cm and hours).
    Returns normalized (z/z_max, t/t_max) pairs.
    """
    z_query = np.asarray(z_query, dtype=np.float32)
    t_query = np.asarray(t_query, dtype=np.float32)
    if z_query.shape != t_query.shape:
        raise ValueError(
            f"z_query and t_query must have same shape, got {z_query.shape} vs {t_query.shape}"
        )
    trunk = np.column_stack([
        z_query / float(z_max),
        t_query / float(t_max),
    ]).astype(np.float32)
    return torch.from_numpy(trunk)


def build_eval_mask_zt(
    z_query: np.ndarray,
    t_query: np.ndarray,
    z_native: np.ndarray,
    t_native: np.ndarray,
    off_grid_only: bool,
) -> np.ndarray:
    """Mask for arbitrary (z,t) pairs: a point is on-grid if BOTH z and t
    are close to some native coordinate."""
    if not off_grid_only:
        return np.ones(len(z_query), dtype=bool)
    z_on = np.any(np.isclose(z_query[:, None], z_native[None, :], atol=1e-4), axis=1)
    t_on = np.any(np.isclose(t_query[:, None], t_native[None, :], atol=1e-6), axis=1)
    return ~(z_on & t_on)


@torch.no_grad()
def predict_zt_points(
    model: torch.nn.Module,
    branch: torch.Tensor,
    scaler,
    z_query: np.ndarray,
    t_query: np.ndarray,
    z_max: float,
    t_max: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Predict h and c at arbitrary (z,t) pairs via direct query.

    Works for DeepONet and Coord-FNN (both accept arbitrary trunk points).
    Does NOT support grid_interp mode.
    """
    trunk = build_trunk_query_zt(z_query, t_query, z_max, t_max).to(device)
    pred = model(branch, trunk)
    h_pred = scaler.inverse_h(pred[..., 0].cpu().numpy())[0].astype(np.float32)
    c_pred = np.clip(
        scaler.inverse_c(pred[..., 1].cpu().numpy())[0], 0.0, None
    ).astype(np.float32)
    return {"h_pred": h_pred, "c_pred": c_pred}


def build_trunk_query(z_cm: float, z_max: float, t_query: np.ndarray) -> torch.Tensor:
    trunk = np.column_stack([
        np.full_like(t_query, z_cm / z_max, dtype=np.float32),
        t_query / float(t_query.max()),
    ]).astype(np.float32)
    return torch.from_numpy(trunk)


def build_eval_mask(t_query: np.ndarray, t_native: np.ndarray, off_grid_only: bool) -> np.ndarray:
    if not off_grid_only:
        return np.ones_like(t_query, dtype=bool)
    is_native = np.any(np.isclose(t_query[:, None], t_native[None, :], atol=1e-6), axis=1)
    return ~is_native


def nearest_time_distance(t_query: np.ndarray, t_native: np.ndarray) -> np.ndarray:
    t_query = np.asarray(t_query, dtype=np.float64)
    t_native = np.asarray(t_native, dtype=np.float64)
    return np.min(np.abs(t_query[:, None] - t_native[None, :]), axis=1)


def interpolate_time_series(
    t_native: np.ndarray,
    values: np.ndarray,
    t_query: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    t_native = np.asarray(t_native, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    if method == "linear":
        return np.interp(t_query, t_native, values).astype(np.float32)
    if method == "pchip":
        interp = PchipInterpolator(t_native, values, extrapolate=False)
        return interp(t_query).astype(np.float32)
    if method == "cubic":
        interp = CubicSpline(t_native, values, extrapolate=False)
        return interp(t_query).astype(np.float32)
    raise ValueError(f"Unsupported grid interpolation method: {method}")


def resolve_query_depth(
    z_native: np.ndarray,
    z_query: float,
    depth_query_mode: str = "nearest",
) -> tuple[float, int]:
    z_native = np.asarray(z_native, dtype=np.float64)
    z_query = float(np.clip(z_query, z_native.min(), z_native.max()))
    if depth_query_mode == "nearest":
        z_idx = int(np.argmin(np.abs(z_native - z_query)))
        return float(z_native[z_idx]), z_idx
    if depth_query_mode == "linear":
        z_idx = int(np.argmin(np.abs(z_native - z_query)))
        return z_query, z_idx
    raise ValueError(f"Unsupported depth query mode: {depth_query_mode}")


def extract_series_at_depth(
    field: np.ndarray,
    z_native: np.ndarray,
    z_query: float,
    depth_query_mode: str = "nearest",
) -> tuple[np.ndarray, float, int]:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"Expected field with shape (n_z, n_t), got {field.shape}")
    z_eff, z_idx = resolve_query_depth(z_native, z_query, depth_query_mode=depth_query_mode)
    if depth_query_mode == "nearest":
        return field[z_idx].astype(np.float32), z_eff, z_idx
    series = np.asarray(
        [np.interp(z_eff, z_native, field[:, j]) for j in range(field.shape[1])],
        dtype=np.float32,
    )
    return series, z_eff, z_idx


def compute_curve_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = pred - ref
    return {
        "rel_l2": float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12)),
        "r2": float(1.0 - np.sum(diff ** 2) / (np.sum((ref - ref.mean()) ** 2) + 1e-12)),
        "max_abs": float(np.max(np.abs(diff))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


@torch.no_grad()
def predict_time_series(
    model_name: str,
    model: torch.nn.Module,
    mode: str,
    branch: torch.Tensor,
    scaler,
    z_native: np.ndarray,
    t_native: np.ndarray,
    z_query: float,
    t_query: np.ndarray,
    device: torch.device,
    grid_interp_method: str = "linear",
    depth_query_mode: str = "nearest",
) -> dict[str, np.ndarray | float | int]:
    z_native = np.asarray(z_native, dtype=np.float32)
    t_native = np.asarray(t_native, dtype=np.float32)
    t_query = np.asarray(t_query, dtype=np.float32)
    z_query, z_idx = resolve_query_depth(z_native, z_query, depth_query_mode=depth_query_mode)

    if mode == "direct":
        trunk_query = build_trunk_query(z_query, float(z_native.max()), t_query).to(device)
        pred = model(branch, trunk_query)
        h_pred = scaler.inverse_h(pred[..., 0].cpu().numpy())[0].astype(np.float32)
        c_pred = np.clip(scaler.inverse_c(pred[..., 1].cpu().numpy())[0], 0.0, None).astype(np.float32)
        return {
            "z_idx": z_idx,
            "z_query": z_query,
            "t_query": t_query,
            "h_pred": h_pred,
            "c_pred": c_pred,
            "depth_query_mode": depth_query_mode,
        }

    if mode == "grid_interp":
        native_trunk = build_trunk_query(z_query, float(z_native.max()), t_native).to(device)
        pred = model(branch, native_trunk)
        h_native = scaler.inverse_h(pred[..., 0].cpu().numpy())[0].reshape(len(z_native), len(t_native))
        c_native = np.clip(
            scaler.inverse_c(pred[..., 1].cpu().numpy())[0].reshape(len(z_native), len(t_native)),
            0.0,
            None,
        )
        h_series_native, z_query_eff, z_idx = extract_series_at_depth(
            h_native, z_native, z_query, depth_query_mode=depth_query_mode
        )
        c_series_native, _, _ = extract_series_at_depth(
            c_native, z_native, z_query, depth_query_mode=depth_query_mode
        )
        h_pred = interpolate_time_series(t_native, h_series_native, t_query, method=grid_interp_method)
        c_pred = interpolate_time_series(t_native, c_series_native, t_query, method=grid_interp_method)
        return {
            "z_idx": z_idx,
            "z_query": z_query_eff,
            "t_query": t_query,
            "h_pred": h_pred,
            "c_pred": c_pred,
            "t_native_pred": t_native,
            "h_native_pred": h_series_native,
            "c_native_pred": c_series_native,
            "grid_interp_method": grid_interp_method,
            "depth_query_mode": depth_query_mode,
        }

    raise ValueError(f"Unsupported query mode: {mode}")
