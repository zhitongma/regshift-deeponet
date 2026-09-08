"""
损失函数模块。

- data_loss: 逐点拟合误差
- mass_conservation_terms: 基于真实累计边界通量的守恒误差
- mass_conservation_loss: 守恒总损失
- pde_residual_loss: Richards 方程 PDE 残差 (通过自动微分)
"""

import torch
import torch.nn.functional as F

from src.utils.normalization import vg_theta_torch, vg_K_torch


def data_loss(
    pred: torch.Tensor,
    h_true: torch.Tensor,
    c_true: torch.Tensor,
    w_h: float = 1.0,
    w_c: float = 1.0,
) -> torch.Tensor:
    """数据拟合损失: w_h * MSE(h) + w_c * MSE(c)。"""
    loss_h = F.mse_loss(pred[..., 0], h_true)
    loss_c = F.mse_loss(pred[..., 1], c_true)
    return w_h * loss_h + w_c * loss_c


def physical_fields_from_prediction(
    pred: torch.Tensor,
    branch_input_raw: torch.Tensor,
    scaler,
    n_z: int,
    n_t: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把模型输出反归一化到物理场并计算 theta。"""
    n_samples = pred.shape[0]

    h_pred = pred[..., 0].reshape(n_samples, n_z, n_t)
    c_pred = pred[..., 1].reshape(n_samples, n_z, n_t)

    h_phys = h_pred * scaler.h_std + scaler.h_mean
    c_phys = torch.clamp(c_pred * scaler.c_std + scaler.c_mean, min=0.0)

    theta_r = branch_input_raw[:, 0].unsqueeze(-1).unsqueeze(-1)
    theta_s = branch_input_raw[:, 1].unsqueeze(-1).unsqueeze(-1)
    alpha = branch_input_raw[:, 2].unsqueeze(-1).unsqueeze(-1)
    n_vg = branch_input_raw[:, 3].unsqueeze(-1).unsqueeze(-1)

    theta_pred = vg_theta_torch(h_phys, theta_r, theta_s, alpha, n_vg)
    return h_phys, c_phys, theta_pred


def _relative_balance_error(
    delta_storage: torch.Tensor,
    cum_target: torch.Tensor,
) -> torch.Tensor:
    """返回与 B2 同口径的相对质量平衡误差(无量纲, 非百分数)。"""
    denom = torch.clamp(cum_target.abs(), min=1.0)
    rel_error = torch.abs(delta_storage - cum_target) / denom
    rel_error = rel_error.clone()
    rel_error[:, 0] = 0.0
    return rel_error


def _reduce_balance_error(
    rel_error: torch.Tensor,
    objective_mode: str,
) -> torch.Tensor:
    """把逐时刻守恒误差约简为训练目标。"""
    if objective_mode == "relative_mse":
        return torch.mean(rel_error ** 2)
    if objective_mode == "relative_mae":
        return torch.mean(rel_error)
    if objective_mode == "final_mae":
        return torch.mean(rel_error[:, -1])
    if objective_mode == "hybrid_mae":
        return 0.5 * torch.mean(rel_error) + 0.5 * torch.mean(rel_error[:, -1])
    raise ValueError(f"Unsupported objective_mode: {objective_mode}")


def mass_conservation_terms(
    pred: torch.Tensor,
    branch_input_raw: torch.Tensor,
    water_cum_target: torch.Tensor,
    solute_cum_target: torch.Tensor,
    scaler,
    n_z: int,
    n_t: int,
    dz: float,
    target_mode: str = "true_flux",
    time_axis: torch.Tensor | None = None,
    objective_mode: str = "relative_mse",
) -> dict[str, torch.Tensor]:
    """计算水量 / 溶质守恒误差项。"""
    _, c_phys, theta_pred = physical_fields_from_prediction(
        pred,
        branch_input_raw,
        scaler,
        n_z,
        n_t,
    )

    storage_w = torch.trapezoid(theta_pred, dx=float(dz), dim=1)
    delta_sw = storage_w - storage_w[:, 0:1]

    theta_c = theta_pred * c_phys
    storage_c = torch.trapezoid(theta_c, dx=float(dz), dim=1)
    delta_sc = storage_c - storage_c[:, 0:1]

    if target_mode == "simplified":
        if time_axis is None:
            raise ValueError("time_axis is required when target_mode='simplified'")
        time_axis = time_axis.view(1, -1).to(pred.device)
        q_top = branch_input_raw[:, 6:7]
        c_top = branch_input_raw[:, 7:8]
        water_cum_target = q_top * time_axis
        solute_cum_target = q_top * c_top * time_axis
    elif target_mode != "true_flux":
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    water_rel_error = _relative_balance_error(delta_sw, water_cum_target)
    solute_rel_error = _relative_balance_error(delta_sc, solute_cum_target)

    loss_w = _reduce_balance_error(water_rel_error, objective_mode)
    loss_c = _reduce_balance_error(solute_rel_error, objective_mode)

    return {
        "water": loss_w,
        "solute": loss_c,
        "water_rel_error": water_rel_error,
        "solute_rel_error": solute_rel_error,
        "water_time_mean": torch.mean(water_rel_error),
        "solute_time_mean": torch.mean(solute_rel_error),
        "water_final_mean": torch.mean(water_rel_error[:, -1]),
        "solute_final_mean": torch.mean(solute_rel_error[:, -1]),
        "water_final_median": torch.median(water_rel_error[:, -1]),
        "solute_final_median": torch.median(solute_rel_error[:, -1]),
        "delta_storage_water": delta_sw,
        "delta_storage_solute": delta_sc,
        "theta": theta_pred,
        "c_phys": c_phys,
    }


def pde_residual_loss(
    model,
    branch_input: torch.Tensor,
    trunk_input: torch.Tensor,
    branch_input_raw: torch.Tensor,
    scaler,
    n_z: int,
    n_t: int,
    domain_length: float = 100.0,
    simulation_time: float = 48.0,
) -> torch.Tensor:
    """Richards 方程 PDE 残差: dθ/dt - d/dz[K(h)(dh/dz + 1)] = 0.

    使用交错网格有限差分: trunk 是共享的规则网格，autograd 无法逐样本求导，
    改用有限差分对物理场求 z/t 导数，梯度仍通过计算图回传到模型参数。
    """
    pred = model(branch_input, trunk_input)
    n_samples = pred.shape[0]

    h_norm = pred[..., 0]
    h_phys = (h_norm * scaler.h_std + scaler.h_mean).reshape(n_samples, n_z, n_t)

    theta_r = branch_input_raw[:, 0].view(-1, 1, 1)
    theta_s = branch_input_raw[:, 1].view(-1, 1, 1)
    alpha   = branch_input_raw[:, 2].view(-1, 1, 1)
    n_vg    = branch_input_raw[:, 3].view(-1, 1, 1)
    K_s     = branch_input_raw[:, 4].view(-1, 1, 1)

    theta = vg_theta_torch(h_phys, theta_r, theta_s, alpha, n_vg)
    K_h   = vg_K_torch(h_phys, K_s, alpha, n_vg)

    dz = domain_length / (n_z - 1)
    dt = simulation_time / (n_t - 1)

    dh_dz_half = (h_phys[:, 1:, :] - h_phys[:, :-1, :]) / dz
    K_half = 0.5 * (K_h[:, 1:, :] + K_h[:, :-1, :])
    flux_half = K_half * (dh_dz_half + 1.0)

    dflux_dz = (flux_half[:, 1:, :] - flux_half[:, :-1, :]) / dz

    theta_int = theta[:, 1:-1, :]
    dtheta_dt = (theta_int[:, :, 1:] - theta_int[:, :, :-1]) / dt

    dflux_dz_t = 0.5 * (dflux_dz[:, :, 1:] + dflux_dz[:, :, :-1])

    residual = dtheta_dt - dflux_dz_t
    return torch.mean(residual ** 2)


def mass_conservation_loss(
    pred: torch.Tensor,
    branch_input_raw: torch.Tensor,
    water_cum_target: torch.Tensor,
    solute_cum_target: torch.Tensor,
    scaler,
    n_z: int,
    n_t: int,
    dz: float,
    w_water: float = 1.0,
    w_solute: float = 1.0,
    target_mode: str = "true_flux",
    time_axis: torch.Tensor | None = None,
    objective_mode: str = "relative_mse",
) -> torch.Tensor:
    """基于真实累计边界通量的守恒总损失。"""
    terms = mass_conservation_terms(
        pred,
        branch_input_raw,
        water_cum_target,
        solute_cum_target,
        scaler,
        n_z,
        n_t,
        dz,
        target_mode=target_mode,
        time_axis=time_axis,
        objective_mode=objective_mode,
    )
    return w_water * terms["water"] + w_solute * terms["solute"]
