#!/usr/bin/env python3
"""
Step 5: 训练 M2 (DeepONet + 质量守恒约束)

核心区别: L_total = L_data + w_mass * L_mass
w_mass 分两阶段: 前 50% epochs = phase1, 后 50% = phase2

用法:
  python scripts/05_train_m2.py
  python scripts/05_train_m2.py --epochs 10000 --device cpu
"""

import os
import argparse
import json
import math
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deeponet import build_model
from src.models.shift_deeponet import build_shift_deeponet
from src.models.crossattn_deeponet import build_crossattn_deeponet
from src.models.losses import data_loss, mass_conservation_terms, pde_residual_loss


def build_model_dispatch(model_cfg: dict):
    arch = model_cfg.get("arch", "deeponet")
    if arch == "shift_deeponet":
        return build_shift_deeponet(model_cfg)
    if arch == "crossattn_deeponet":
        return build_crossattn_deeponet(model_cfg)
    return build_model(model_cfg)
from src.data_generation.dataset import MassConservationDataset
from src.data_generation.postprocess import DataScaler
from src.utils.experiment import (
    configure_logger,
    ensure_dirs,
    load_config,
    resolve_artifact_paths,
    save_config_snapshot,
    save_manifest,
)
from src.utils.training import (
    epochs_without_improvement,
    patience_epochs_to_checks,
    should_validate,
)


def load_data(proc_dir: Path):
    """加载预处理数据 (返回 numpy dict)。"""
    datasets = {}
    for split in ["train", "val", "test"]:
        d = np.load(proc_dir / f"{split}.npz")
        datasets[split] = {k: d[k] for k in d.files}
    return datasets


def get_raw_params(branch_norm: torch.Tensor, scaler: DataScaler,
                   device: torch.device) -> torch.Tensor:
    """反归一化 branch 输入, 得到物理空间参数。"""
    input_min = torch.from_numpy(scaler.input_min).float().to(device)
    input_max = torch.from_numpy(scaler.input_max).float().to(device)
    return branch_norm * (input_max - input_min + 1e-8) + input_min


def get_w_mass(epoch: int, total_epochs: int, cfg: dict) -> float:
    """渐进式质量守恒权重调度。"""
    schedule = cfg.get("schedule", "delayed")
    if schedule == "linear_ramp":
        return cfg["w_mass_phase2"] * (epoch / max(total_epochs, 1))
    if schedule == "exponential_ramp":
        return cfg["w_mass_phase2"] * (1.0 - math.exp(-3.0 * epoch / max(total_epochs, 1)))
    frac = cfg["switch_epoch_fraction"]
    if epoch < total_epochs * frac:
        return cfg["w_mass_phase1"]
    return cfg["w_mass_phase2"]


def get_w_pde(epoch: int, total_epochs: int, cfg: dict) -> float:
    """PDE 残差权重调度，复用 mass schedule 逻辑。"""
    w1 = float(cfg.get("w_pde_phase1", 0.0))
    w2 = float(cfg.get("w_pde_phase2", 0.0))
    if w1 == 0.0 and w2 == 0.0:
        return 0.0
    schedule = cfg.get("schedule", "delayed")
    if schedule == "linear_ramp":
        return w2 * (epoch / max(total_epochs, 1))
    if schedule == "exponential_ramp":
        return w2 * (1.0 - math.exp(-3.0 * epoch / max(total_epochs, 1)))
    frac = cfg["switch_epoch_fraction"]
    if epoch < total_epochs * frac:
        return w1
    return w2


def get_phase2_start_epoch(total_epochs: int, cfg: dict) -> int:
    """返回 phase2 首个生效 epoch。"""
    frac = float(cfg.get("switch_epoch_fraction", 0.0))
    return max(1, int(math.ceil(total_epochs * frac)))


def get_mass_component_weights(epoch: int, total_epochs: int, cfg: dict) -> tuple[float, float]:
    """返回当前阶段 water / solute 守恒分量权重。"""
    frac = cfg["switch_epoch_fraction"]
    phase = "phase1" if epoch < total_epochs * frac else "phase2"
    return (
        float(cfg.get(f"w_mass_water_{phase}", cfg.get("w_mass_water", 1.0))),
        float(cfg.get(f"w_mass_solute_{phase}", cfg.get("w_mass_solute", 1.0))),
    )


def build_scheduler(optimizer, lr_factor: float, scheduler_patience_checks: int, lr_min: float):
    """构造 ReduceLROnPlateau 调度器，便于 phase2 重新初始化。"""
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=scheduler_patience_checks,
        min_lr=lr_min,
    )


def get_selection_mass_value(mass_terms: dict, component: str, metric_name: str) -> float:
    """把守恒误差转换为 checkpoint 选择/监控口径。"""
    metric_map = {
        "objective": component,
        "time_mean": f"{component}_time_mean",
        "final_mean": f"{component}_final_mean",
        "final_median": f"{component}_final_median",
    }
    if metric_name not in metric_map:
        raise KeyError(f"未知守恒选模口径: {metric_name}")
    return float(mass_terms[metric_map[metric_name]].item())


def get_monitor_value(metrics: dict, metric_name: str) -> float:
    """统一读取训练监控指标。"""
    if metric_name not in metrics:
        raise KeyError(f"未知监控指标: {metric_name}")
    return float(metrics[metric_name])


def evaluate_checkpoint(model, ckpt_path: Path, data, trunk, device, scaler,
                        n_z, n_t, dz, w_h, w_c, mass_cfg, time_axis):
    """加载指定 checkpoint 并返回测试集指标。"""
    if not ckpt_path.exists():
        return None
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return evaluate(
        model,
        data,
        trunk,
        device,
        scaler,
        n_z,
        n_t,
        dz,
        w_h,
        w_c,
        mass_cfg,
        time_axis,
    )


def train_epoch(model, loader, trunk, optimizer, scaler, w_mass, mass_component_weights,
                n_z, n_t, device, dz, w_h, w_c, mass_cfg, time_axis,
                w_pde=0.0, alm_state=None, gradnorm_mode=False,
                domain_length=100.0, simulation_time=48.0,
                n_coords_train=None):
    """Mini-batch 训练一个 epoch: L_data + w_mass * L_mass [+ w_pde * L_pde]."""
    model.train()
    total_loss = 0.0
    total_data = 0.0
    total_mass = 0.0
    total_mass_w = 0.0
    total_mass_c = 0.0
    total_pde = 0.0
    n_batches = 0
    mass_water_weight, mass_solute_weight = mass_component_weights
    use_alm = alm_state is not None

    for branch_b, h_b, c_b, water_cum_b, solute_cum_b in loader:
        branch_b = branch_b.to(device, non_blocking=True)
        h_b = h_b.to(device, non_blocking=True)
        c_b = c_b.to(device, non_blocking=True)
        water_cum_b = water_cum_b.to(device, non_blocking=True)
        solute_cum_b = solute_cum_b.to(device, non_blocking=True)

        optimizer.zero_grad()

        M = trunk.shape[0]
        if n_coords_train and 0 < n_coords_train < M:
            idx = torch.randperm(M, device=device)[:n_coords_train]
            pred_sub = model(branch_b, trunk[idx])
            l_data = data_loss(pred_sub, h_b[:, idx], c_b[:, idx], w_h=w_h, w_c=w_c)
            pred = model(branch_b, trunk) if (w_mass > 0 or use_alm or w_pde > 0) else pred_sub
        else:
            pred = model(branch_b, trunk)
            l_data = data_loss(pred, h_b, c_b, w_h=w_h, w_c=w_c)

        effective_w_mass = w_mass
        if w_mass > 0 or use_alm:
            raw_params = get_raw_params(branch_b, scaler, device)
            mass_terms = mass_conservation_terms(
                pred, raw_params, water_cum_b, solute_cum_b,
                scaler, n_z, n_t, dz,
                target_mode=mass_cfg.get("target_mode", "true_flux"),
                time_axis=time_axis,
                objective_mode=mass_cfg.get("objective_mode", "relative_mse"),
            )
            l_mass = (
                mass_water_weight * mass_terms["water"]
                + mass_solute_weight * mass_terms["solute"]
            )
            if use_alm:
                constraint_w = mass_terms["water"]
                constraint_c = mass_terms["solute"]
                alm_loss = (
                    alm_state["lambda_w"] * constraint_w
                    + alm_state["lambda_c"] * constraint_c
                    + alm_state["rho"] / 2.0 * (constraint_w ** 2 + constraint_c ** 2)
                )
                loss = l_data + alm_loss
            elif gradnorm_mode and w_mass > 0:
                shared_param = next(model.branch.parameters())
                g_data = torch.autograd.grad(l_data, shared_param, retain_graph=True, allow_unused=True)[0]
                g_mass = torch.autograd.grad(l_mass, shared_param, retain_graph=True, allow_unused=True)[0]
                if g_data is not None and g_mass is not None:
                    ratio = (torch.norm(g_data) / (torch.norm(g_mass) + 1e-8)).detach()
                    effective_w_mass = float(ratio.item())
                loss = l_data + effective_w_mass * l_mass
            else:
                loss = l_data + w_mass * l_mass
        else:
            mass_terms = {
                "water": torch.tensor(0.0, device=device),
                "solute": torch.tensor(0.0, device=device),
            }
            l_mass = torch.tensor(0.0, device=device)
            raw_params = None
            loss = l_data

        l_pde_val = 0.0
        if w_pde > 0:
            if raw_params is None:
                raw_params = get_raw_params(branch_b, scaler, device)
            l_pde = pde_residual_loss(
                model, branch_b, trunk, raw_params, scaler,
                n_z, n_t, domain_length, simulation_time,
            )
            loss = loss + w_pde * l_pde
            l_pde_val = l_pde.item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_data += l_data.item()
        total_mass += l_mass.item()
        total_mass_w += mass_terms["water"].item()
        total_mass_c += mass_terms["solute"].item()
        total_pde += l_pde_val
        n_batches += 1

    if use_alm:
        with torch.no_grad():
            alm_state["lambda_w"] = alm_state["lambda_w"] + alm_state["rho"] * total_mass_w / n_batches
            alm_state["lambda_c"] = alm_state["lambda_c"] + alm_state["rho"] * total_mass_c / n_batches

    return (
        total_loss / n_batches,
        total_data / n_batches,
        total_mass / n_batches,
        total_mass_w / n_batches,
        total_mass_c / n_batches,
    )


@torch.no_grad()
def evaluate(model, data, trunk, device, scaler, n_z, n_t, dz, w_h, w_c, mass_cfg, time_axis):
    """全量验证/测试评估。"""
    model.eval()
    branch = torch.from_numpy(data["branch_inputs"]).to(device)
    h_true = torch.from_numpy(data["h_targets"]).to(device)
    c_true = torch.from_numpy(data["c_targets"]).to(device)
    water_cum = torch.from_numpy(data["water_cum_net"]).to(device)
    solute_cum = torch.from_numpy(data["solute_cum_net"]).to(device)

    pred = model(branch, trunk)
    loss = data_loss(pred, h_true, c_true, w_h=w_h, w_c=w_c).item()
    h_err = torch.norm(pred[..., 0] - h_true) / torch.norm(h_true)
    c_err = torch.norm(pred[..., 1] - c_true) / torch.norm(c_true)
    raw_params = get_raw_params(branch, scaler, device)
    mass_terms = mass_conservation_terms(
        pred,
        raw_params,
        water_cum,
        solute_cum,
        scaler,
        n_z,
        n_t,
        dz,
        target_mode=mass_cfg.get("target_mode", "true_flux"),
        time_axis=time_axis,
        objective_mode=mass_cfg.get("objective_mode", "relative_mse"),
    )
    selection_mass_metric = mass_cfg.get("selection_mass_metric", "objective")
    mass_water = get_selection_mass_value(mass_terms, "water", selection_mass_metric)
    mass_solute = get_selection_mass_value(mass_terms, "solute", selection_mass_metric)
    score = (
        mass_cfg.get("selection_data_weight", 1.0) * loss
        + mass_cfg.get("selection_mass_water_weight", 0.0) * mass_water
        + mass_cfg.get("selection_mass_solute_weight", 0.0) * mass_solute
    )
    return {
        "loss": loss,
        "h_rel_l2": h_err.item(),
        "c_rel_l2": c_err.item(),
        "mass_water": mass_water,
        "mass_solute": mass_solute,
        "mass_water_objective": mass_terms["water"].item(),
        "mass_solute_objective": mass_terms["solute"].item(),
        "b2_water_time_mean_pct": mass_terms["water_time_mean"].item() * 100.0,
        "b2_solute_time_mean_pct": mass_terms["solute_time_mean"].item() * 100.0,
        "b2_water_final_mean_pct": mass_terms["water_final_mean"].item() * 100.0,
        "b2_solute_final_mean_pct": mass_terms["solute_final_mean"].item() * 100.0,
        "b2_water_final_median_pct": mass_terms["water_final_median"].item() * 100.0,
        "b2_solute_final_median_pct": mass_terms["solute_final_median"].item() * 100.0,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser(description="训练 M2 DeepONet+Mass")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--w-mass-phase1", type=float, default=None)
    parser.add_argument("--w-mass-phase2", type=float, default=None)
    parser.add_argument("--switch-epoch-fraction", type=float, default=None)
    parser.add_argument("--w-mass-water", type=float, default=None)
    parser.add_argument("--w-mass-solute", type=float, default=None)
    parser.add_argument("--w-mass-water-phase1", type=float, default=None)
    parser.add_argument("--w-mass-water-phase2", type=float, default=None)
    parser.add_argument("--w-mass-solute-phase1", type=float, default=None)
    parser.add_argument("--w-mass-solute-phase2", type=float, default=None)
    parser.add_argument("--selection-data-weight", type=float, default=None)
    parser.add_argument("--selection-mass-water-weight", type=float, default=None)
    parser.add_argument("--selection-mass-solute-weight", type=float, default=None)
    parser.add_argument(
        "--mass-objective-mode",
        type=str,
        choices=["relative_mse", "relative_mae", "final_mae", "hybrid_mae"],
        default=None,
    )
    parser.add_argument(
        "--selection-mass-metric",
        type=str,
        choices=["objective", "time_mean", "final_mean", "final_median"],
        default=None,
    )
    parser.add_argument("--target-mode", type=str, choices=["true_flux", "simplified"], default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--scheduler-metric", type=str, choices=["loss", "score"], default=None)
    parser.add_argument("--early-stop-metric", type=str, choices=["loss", "score"], default=None)
    parser.add_argument("--alias-metric", type=str, choices=["loss", "score"], default=None)
    parser.add_argument("--warm-start", dest="warm_start", action="store_true")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    parser.set_defaults(warm_start=None)
    args = parser.parse_args()

    cfg, cfg_path = load_config(PROJECT_ROOT, args.config)
    paths = resolve_artifact_paths(PROJECT_ROOT, cfg, args.run_dir)
    ensure_dirs(paths, "processed_data", "checkpoints", "logs", "experiments", "metadata")
    logger = configure_logger(__name__, paths.logs / "train_m2.log")

    tcfg = cfg["training"]
    mass_cfg = dict(cfg["mass_loss"])
    if args.w_mass_phase1 is not None:
        mass_cfg["w_mass_phase1"] = args.w_mass_phase1
    if args.w_mass_phase2 is not None:
        mass_cfg["w_mass_phase2"] = args.w_mass_phase2
    if args.switch_epoch_fraction is not None:
        mass_cfg["switch_epoch_fraction"] = args.switch_epoch_fraction
    if args.w_mass_water is not None:
        mass_cfg["w_mass_water"] = args.w_mass_water
    if args.w_mass_solute is not None:
        mass_cfg["w_mass_solute"] = args.w_mass_solute
    if args.w_mass_water_phase1 is not None:
        mass_cfg["w_mass_water_phase1"] = args.w_mass_water_phase1
    if args.w_mass_water_phase2 is not None:
        mass_cfg["w_mass_water_phase2"] = args.w_mass_water_phase2
    if args.w_mass_solute_phase1 is not None:
        mass_cfg["w_mass_solute_phase1"] = args.w_mass_solute_phase1
    if args.w_mass_solute_phase2 is not None:
        mass_cfg["w_mass_solute_phase2"] = args.w_mass_solute_phase2
    if args.selection_data_weight is not None:
        mass_cfg["selection_data_weight"] = args.selection_data_weight
    if args.selection_mass_water_weight is not None:
        mass_cfg["selection_mass_water_weight"] = args.selection_mass_water_weight
    if args.selection_mass_solute_weight is not None:
        mass_cfg["selection_mass_solute_weight"] = args.selection_mass_solute_weight
    if args.mass_objective_mode is not None:
        mass_cfg["objective_mode"] = args.mass_objective_mode
    if args.selection_mass_metric is not None:
        mass_cfg["selection_mass_metric"] = args.selection_mass_metric
    if args.target_mode is not None:
        mass_cfg["target_mode"] = args.target_mode

    epochs = args.epochs or tcfg["epochs"]
    warm_start = tcfg.get("m2_warm_start", True) if args.warm_start is None else args.warm_start
    base_lr = tcfg.get("learning_rate", tcfg.get("lr"))
    default_m2_lr = tcfg.get("m2_learning_rate", base_lr)
    lr = args.lr or (default_m2_lr if warm_start else base_lr)
    lr_min = tcfg["lr_min"]
    patience_epochs = tcfg["early_stopping_patience"]
    seed = args.seed or tcfg["seed"]
    batch_size = tcfg.get("batch_size") or None
    val_interval = args.val_interval or tcfg.get("val_interval", 100)
    lr_patience_epochs = tcfg.get("lr_patience", 1000)
    lr_factor = tcfg.get("lr_factor", 0.5)
    physics = cfg["physics"]
    n_z = physics["n_spatial_nodes"]
    n_t = physics["n_time_steps"]
    w_h = cfg.get("data_loss", {}).get("w_h", 1.0)
    w_c = cfg.get("data_loss", {}).get("w_c", 1.0)
    warm_ckpt_name = args.warm_start_checkpoint or tcfg.get("m2_warm_start_checkpoint", "m1_best.pt")
    scheduler_metric_name = args.scheduler_metric or tcfg.get("m2_scheduler_metric", "loss")
    early_stop_metric_name = args.early_stop_metric or tcfg.get("m2_early_stopping_metric", scheduler_metric_name)
    alias_metric_name = args.alias_metric or tcfg.get("m2_best_alias_metric", early_stop_metric_name)
    scheduler_patience_checks = patience_epochs_to_checks(lr_patience_epochs, val_interval)
    phase2_selection_only = bool(tcfg.get("m2_phase2_selection_only", False))
    phase2_reset_scheduler = bool(tcfg.get("m2_phase2_reset_scheduler", phase2_selection_only))
    phase2_reset_early_stop = bool(tcfg.get("m2_phase2_reset_early_stopping", phase2_selection_only))
    phase2_require_entry_before_early_stop = bool(
        tcfg.get("m2_phase2_require_entry_before_early_stop", phase2_selection_only)
    )
    phase2_scheduler_metric_name = tcfg.get("m2_phase2_scheduler_metric", scheduler_metric_name)
    phase2_early_stop_metric_name = tcfg.get("m2_phase2_early_stopping_metric", early_stop_metric_name)
    phase2_alias_metric_name = tcfg.get("m2_phase2_alias_metric", alias_metric_name)
    phase2_start_epoch = get_phase2_start_epoch(epochs, mass_cfg)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    save_config_snapshot(
        cfg,
        cfg_path,
        paths,
        "train_m2",
        extra={
            "epochs": epochs,
            "lr": lr,
            "val_interval": val_interval,
            "lr_patience_epochs": lr_patience_epochs,
            "scheduler_patience_checks": scheduler_patience_checks,
            "warm_start": warm_start,
            "warm_start_checkpoint": warm_ckpt_name,
            "target_mode": mass_cfg.get("target_mode", "true_flux"),
            "scheduler_metric": scheduler_metric_name,
            "early_stop_metric": early_stop_metric_name,
            "alias_metric": alias_metric_name,
            "phase2_selection_only": phase2_selection_only,
            "phase2_start_epoch": phase2_start_epoch,
            "phase2_reset_scheduler": phase2_reset_scheduler,
            "phase2_reset_early_stopping": phase2_reset_early_stop,
            "phase2_require_entry_before_early_stop": phase2_require_entry_before_early_stop,
            "phase2_scheduler_metric": phase2_scheduler_metric_name,
            "phase2_early_stop_metric": phase2_early_stop_metric_name,
            "phase2_alias_metric": phase2_alias_metric_name,
            "w_mass_phase1": mass_cfg["w_mass_phase1"],
            "w_mass_phase2": mass_cfg["w_mass_phase2"],
            "switch_epoch_fraction": mass_cfg["switch_epoch_fraction"],
            "w_mass_water": mass_cfg.get("w_mass_water", 1.0),
            "w_mass_solute": mass_cfg.get("w_mass_solute", 1.0),
            "w_mass_water_phase1": mass_cfg.get("w_mass_water_phase1", mass_cfg.get("w_mass_water", 1.0)),
            "w_mass_water_phase2": mass_cfg.get("w_mass_water_phase2", mass_cfg.get("w_mass_water", 1.0)),
            "w_mass_solute_phase1": mass_cfg.get("w_mass_solute_phase1", mass_cfg.get("w_mass_solute", 1.0)),
            "w_mass_solute_phase2": mass_cfg.get("w_mass_solute_phase2", mass_cfg.get("w_mass_solute", 1.0)),
            "objective_mode": mass_cfg.get("objective_mode", "relative_mse"),
            "selection_mass_metric": mass_cfg.get("selection_mass_metric", "objective"),
            "selection_data_weight": mass_cfg.get("selection_data_weight", 1.0),
            "selection_mass_water_weight": mass_cfg.get("selection_mass_water_weight", 0.0),
            "selection_mass_solute_weight": mass_cfg.get("selection_mass_solute_weight", 0.0),
        },
    )

    logger.info("=" * 60)
    logger.info("M2 DeepONet + Mass Conservation 训练")
    logger.info("  设备: %s", device)
    logger.info("  run_root: %s", paths.root)
    logger.info("  Epochs: %d, LR: %.1e -> %.1e", epochs, lr, lr_min)
    logger.info("  Batch size: %s", batch_size or "full")
    logger.info(
        "  val_interval=%d, lr_patience=%d epochs -> %d checks, early_stop=%d epochs",
        val_interval,
        lr_patience_epochs,
        scheduler_patience_checks,
        patience_epochs,
    )
    logger.info("  w_mass: %.2f (前%.0f%%) -> %.2f (后%.0f%%)",
                mass_cfg["w_mass_phase1"],
                mass_cfg["switch_epoch_fraction"] * 100,
                mass_cfg["w_mass_phase2"],
                (1 - mass_cfg["switch_epoch_fraction"]) * 100)
    logger.info("  target_mode: %s", mass_cfg.get("target_mode", "true_flux"))
    logger.info(
        "  mass objective: %s, selection_mass_metric: %s",
        mass_cfg.get("objective_mode", "relative_mse"),
        mass_cfg.get("selection_mass_metric", "objective"),
    )
    logger.info("  warm_start: %s (%s)", warm_start, warm_ckpt_name)
    logger.info("  seed: %d", seed)
    logger.info(
        "  monitor: scheduler=%s, early_stop=%s, alias=%s",
        scheduler_metric_name,
        early_stop_metric_name,
        alias_metric_name,
    )
    logger.info(
        "  phase2 policy: start_epoch=%d, selection_only=%s, reset_scheduler=%s, reset_early_stop=%s",
        phase2_start_epoch,
        phase2_selection_only,
        phase2_reset_scheduler,
        phase2_reset_early_stop,
    )
    logger.info(
        "  phase2 monitor: scheduler=%s, early_stop=%s, alias=%s, require_entry_before_early_stop=%s",
        phase2_scheduler_metric_name,
        phase2_early_stop_metric_name,
        phase2_alias_metric_name,
        phase2_require_entry_before_early_stop,
    )
    logger.info(
        "  selection weights: data=%.3f, water=%.3f, solute=%.3f",
        mass_cfg.get("selection_data_weight", 1.0),
        mass_cfg.get("selection_mass_water_weight", 0.0),
        mass_cfg.get("selection_mass_solute_weight", 0.0),
    )
    logger.info(
        "  mass component weights(legacy): water=%.3f, solute=%.3f",
        mass_cfg.get("w_mass_water", 1.0),
        mass_cfg.get("w_mass_solute", 1.0),
    )
    logger.info(
        "  phase components: water=(%.3f -> %.3f), solute=(%.3f -> %.3f)",
        mass_cfg.get("w_mass_water_phase1", mass_cfg.get("w_mass_water", 1.0)),
        mass_cfg.get("w_mass_water_phase2", mass_cfg.get("w_mass_water", 1.0)),
        mass_cfg.get("w_mass_solute_phase1", mass_cfg.get("w_mass_solute", 1.0)),
        mass_cfg.get("w_mass_solute_phase2", mass_cfg.get("w_mass_solute", 1.0)),
    )
    logger.info("=" * 60)

    # 数据
    datasets = load_data(paths.processed_data)
    scaler = DataScaler.load(paths.processed_data / "scaler.npz")

    train_ds = MassConservationDataset(datasets["train"])
    bs = batch_size if batch_size else len(train_ds)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    trunk = train_ds.trunk_inputs.to(device)

    dz = float(datasets["train"]["z"][1] - datasets["train"]["z"][0])
    time_axis = torch.from_numpy(datasets["train"]["t"].astype(np.float32)).to(device)

    logger.info("数据加载完成: train=%d, val=%d, test=%d (batch_size=%d, steps/epoch=%d)",
                len(train_ds),
                len(datasets["val"]["branch_inputs"]),
                len(datasets["test"]["branch_inputs"]),
                bs, len(train_loader))

    # 模型
    model = build_model_dispatch(cfg["model"]).to(device)
    logger.info("模型参数量: %d (%.1f K)", model.num_params, model.num_params / 1000)

    ckpt_dir = paths.checkpoints
    if warm_start:
        warm_ckpt = ckpt_dir / warm_ckpt_name
        if warm_ckpt.exists():
            model.load_state_dict(torch.load(warm_ckpt, map_location=device, weights_only=True))
            logger.info("已加载 warm-start 权重: %s", warm_ckpt)
        else:
            logger.warning("warm-start checkpoint 不存在，改为从头训练: %s", warm_ckpt)
            warm_start = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, lr_factor, scheduler_patience_checks, lr_min)

    # 训练
    best_val_data = float("inf")
    best_val_score = float("inf")
    best_data_epoch = None
    best_score_epoch = None
    best_early_metric = float("inf")
    best_early_epoch = None
    best_alias_metric = float("inf")
    best_alias_epoch = None
    best_pre_phase2_score = float("inf")
    best_pre_phase2_score_epoch = None
    best_pre_phase2_alias_metric = float("inf")
    best_pre_phase2_alias_epoch = None
    best_phase2_score = float("inf")
    best_phase2_score_epoch = None
    best_phase2_alias_metric = float("inf")
    best_phase2_alias_epoch = None
    phase2_validation_started = False
    phase2_start_val_epoch = None
    active_scheduler_metric_name = scheduler_metric_name
    active_early_stop_metric_name = early_stop_metric_name
    active_alias_metric_name = alias_metric_name
    history = {
        "train_loss": [], "data_loss": [], "mass_loss": [],
        "mass_water": [], "mass_solute": [],
        "val_loss": [], "val_mass_water": [], "val_mass_solute": [],
        "val_mass_water_objective": [], "val_mass_solute_objective": [],
        "val_b2_water_time_mean_pct": [], "val_b2_solute_time_mean_pct": [],
        "val_b2_water_final_mean_pct": [], "val_b2_solute_final_mean_pct": [],
        "val_b2_water_final_median_pct": [], "val_b2_solute_final_median_pct": [],
        "val_score": [], "h_rel_l2": [], "c_rel_l2": [], "w_mass": [],
        "w_mass_water": [], "w_mass_solute": [],
        "val_epoch": [], "lr": [], "scheduler_metric": [], "early_stop_metric": [], "alias_metric": [],
        "phase2_active": [],
    }

    t_start = time.time()

    use_alm = bool(mass_cfg.get("alm_mode", False))
    alm_state = None
    if use_alm:
        alm_state = {
            "lambda_w": 0.0,
            "lambda_c": 0.0,
            "rho": float(mass_cfg.get("alm_rho", 1.0)),
        }
    gradnorm_mode = bool(mass_cfg.get("gradnorm_mode", False))

    for epoch in range(1, epochs + 1):
        w_mass = get_w_mass(epoch, epochs, mass_cfg)
        w_pde = get_w_pde(epoch, epochs, mass_cfg)
        mass_component_weights = get_mass_component_weights(epoch, epochs, mass_cfg)

        total, l_d, l_m, l_m_w, l_m_c = train_epoch(
            model, train_loader, trunk, optimizer, scaler,
            w_mass, mass_component_weights, n_z, n_t, device, dz, w_h, w_c, mass_cfg, time_axis,
            w_pde=w_pde, alm_state=alm_state, gradnorm_mode=gradnorm_mode,
            domain_length=float(physics.get("domain_length", 100.0)),
            simulation_time=float(physics.get("simulation_time", 48.0)),
            n_coords_train=tcfg.get("n_coords_train"),
        )

        history["train_loss"].append(total)
        history["data_loss"].append(l_d)
        history["mass_loss"].append(l_m)
        history["mass_water"].append(l_m_w)
        history["mass_solute"].append(l_m_c)
        history["w_mass"].append(w_mass)
        history["w_mass_water"].append(mass_component_weights[0])
        history["w_mass_solute"].append(mass_component_weights[1])

        if should_validate(epoch, val_interval):
            val_metrics = evaluate(
                model,
                datasets["val"],
                trunk,
                device,
                scaler,
                n_z,
                n_t,
                dz,
                w_h,
                w_c,
                mass_cfg,
                time_axis,
            )
            history["val_loss"].append(val_metrics["loss"])
            history["val_mass_water"].append(val_metrics["mass_water"])
            history["val_mass_solute"].append(val_metrics["mass_solute"])
            history["val_mass_water_objective"].append(val_metrics["mass_water_objective"])
            history["val_mass_solute_objective"].append(val_metrics["mass_solute_objective"])
            history["val_b2_water_time_mean_pct"].append(val_metrics["b2_water_time_mean_pct"])
            history["val_b2_solute_time_mean_pct"].append(val_metrics["b2_solute_time_mean_pct"])
            history["val_b2_water_final_mean_pct"].append(val_metrics["b2_water_final_mean_pct"])
            history["val_b2_solute_final_mean_pct"].append(val_metrics["b2_solute_final_mean_pct"])
            history["val_b2_water_final_median_pct"].append(val_metrics["b2_water_final_median_pct"])
            history["val_b2_solute_final_median_pct"].append(val_metrics["b2_solute_final_median_pct"])
            history["val_score"].append(val_metrics["score"])
            history["h_rel_l2"].append(val_metrics["h_rel_l2"])
            history["c_rel_l2"].append(val_metrics["c_rel_l2"])
            history["val_epoch"].append(epoch)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            is_phase2 = epoch >= phase2_start_epoch
            if is_phase2 and not phase2_validation_started:
                phase2_validation_started = True
                phase2_start_val_epoch = epoch
                active_scheduler_metric_name = phase2_scheduler_metric_name
                active_early_stop_metric_name = phase2_early_stop_metric_name
                active_alias_metric_name = phase2_alias_metric_name
                if phase2_selection_only:
                    best_val_score = float("inf")
                    best_score_epoch = None
                    best_alias_metric = float("inf")
                    best_alias_epoch = None
                if phase2_reset_early_stop:
                    best_early_metric = float("inf")
                    best_early_epoch = None
                if phase2_reset_scheduler:
                    scheduler = build_scheduler(optimizer, lr_factor, scheduler_patience_checks, lr_min)
                logger.info(
                    "进入 phase2: epoch=%d | scheduler=%s | early_stop=%s | alias=%s",
                    epoch,
                    active_scheduler_metric_name,
                    active_early_stop_metric_name,
                    active_alias_metric_name,
                )

            scheduler_value = get_monitor_value(val_metrics, active_scheduler_metric_name)
            early_stop_value = get_monitor_value(val_metrics, active_early_stop_metric_name)
            alias_value = get_monitor_value(val_metrics, active_alias_metric_name)
            history["scheduler_metric"].append(scheduler_value)
            history["early_stop_metric"].append(early_stop_value)
            history["alias_metric"].append(alias_value)
            history["phase2_active"].append(1.0 if is_phase2 else 0.0)

            if is_phase2:
                if val_metrics["score"] < best_phase2_score:
                    best_phase2_score = val_metrics["score"]
                    best_phase2_score_epoch = epoch
                if alias_value < best_phase2_alias_metric:
                    best_phase2_alias_metric = alias_value
                    best_phase2_alias_epoch = epoch
            else:
                if val_metrics["score"] < best_pre_phase2_score:
                    best_pre_phase2_score = val_metrics["score"]
                    best_pre_phase2_score_epoch = epoch
                if alias_value < best_pre_phase2_alias_metric:
                    best_pre_phase2_alias_metric = alias_value
                    best_pre_phase2_alias_epoch = epoch

            scheduler.step(scheduler_value)

            if val_metrics["loss"] < best_val_data:
                best_val_data = val_metrics["loss"]
                best_data_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "m2_best_data.pt")

            allow_phase2_checkpoint = (not phase2_selection_only) or is_phase2
            if allow_phase2_checkpoint and val_metrics["score"] < best_val_score:
                best_val_score = val_metrics["score"]
                best_score_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "m2_best_score.pt")
                if is_phase2:
                    torch.save(model.state_dict(), ckpt_dir / "m2_phase2_best_score.pt")

            if early_stop_value < best_early_metric:
                best_early_metric = early_stop_value
                best_early_epoch = epoch

            if allow_phase2_checkpoint and alias_value < best_alias_metric:
                best_alias_metric = alias_value
                best_alias_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "m2_best.pt")
                if is_phase2:
                    torch.save(model.state_dict(), ckpt_dir / "m2_phase2_best.pt")

            if phase2_require_entry_before_early_stop and not phase2_validation_started:
                no_improve_epochs = 0
            else:
                no_improve_epochs = epochs_without_improvement(epoch, best_early_epoch)
            if epoch == 1 or epoch % (val_interval * 10) == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %5d | total=%.4e | data=%.4e | mass=%.4e "
                    "(w=%.3e,c=%.3e) | val=%.4e | mw=%.3e | mc=%.3e | "
                    "mw_obj=%.3e | mc_obj=%.3e | B2w=%.2f%% | B2c=%.2f%% | "
                    "score=%.4e | sched[%s]=%.4e | stop[%s]=%.4e | alias[%s]=%.4e | "
                    "h_L2=%.4f | c_L2=%.4f | w=%.2f | ww=%.2f | wc=%.2f | phase2=%s | lr=%.1e | no_improve=%d",
                    epoch, total, l_d, l_m, l_m_w, l_m_c,
                    val_metrics["loss"], val_metrics["mass_water"], val_metrics["mass_solute"],
                    val_metrics["mass_water_objective"], val_metrics["mass_solute_objective"],
                    val_metrics["b2_water_final_mean_pct"], val_metrics["b2_solute_final_mean_pct"],
                    val_metrics["score"],
                    active_scheduler_metric_name,
                    scheduler_value,
                    active_early_stop_metric_name,
                    early_stop_value,
                    active_alias_metric_name,
                    alias_value,
                    val_metrics["h_rel_l2"], val_metrics["c_rel_l2"],
                    w_mass, mass_component_weights[0], mass_component_weights[1],
                    is_phase2,
                    lr_now,
                    no_improve_epochs,
                )

            if no_improve_epochs >= patience_epochs:
                logger.info(
                    "早停触发 (patience=%d epochs, metric=%s)",
                    patience_epochs,
                    early_stop_metric_name,
                )
                break

    elapsed = time.time() - t_start
    logger.info("训练完成! 耗时: %.1f min", elapsed / 60)

    torch.save(model.state_dict(), ckpt_dir / "m2_final.pt")
    checkpoint_paths = {
        "best_data": ckpt_dir / "m2_best_data.pt",
        "best_score": ckpt_dir / "m2_best_score.pt",
        "best_alias": ckpt_dir / "m2_best.pt",
        "final": ckpt_dir / "m2_final.pt",
    }
    if (ckpt_dir / "m2_phase2_best_score.pt").exists():
        checkpoint_paths["phase2_best_score"] = ckpt_dir / "m2_phase2_best_score.pt"
    if (ckpt_dir / "m2_phase2_best.pt").exists():
        checkpoint_paths["phase2_best_alias"] = ckpt_dir / "m2_phase2_best.pt"
    checkpoint_test_metrics = {}
    for name, ckpt_path in checkpoint_paths.items():
        metrics = evaluate_checkpoint(
            model,
            ckpt_path,
            datasets["test"],
            trunk,
            device,
            scaler,
            n_z,
            n_t,
            dz,
            w_h,
            w_c,
            mass_cfg,
            time_axis,
        )
        if metrics is None:
            continue
        checkpoint_test_metrics[name] = metrics
        logger.info(
            "测试集[%s]: data=%.4e | mw=%.3e | mc=%.3e | B2w=%.2f%% | B2c=%.2f%% | score=%.4e | h_L2=%.4f | c_L2=%.4f",
            name,
            metrics["loss"],
            metrics["mass_water"],
            metrics["mass_solute"],
            metrics["b2_water_final_mean_pct"],
            metrics["b2_solute_final_mean_pct"],
            metrics["score"],
            metrics["h_rel_l2"],
            metrics["c_rel_l2"],
        )
    test_metrics = checkpoint_test_metrics.get("best_alias", {})
    phase2_checkpoint_test_metrics = {
        name: metrics
        for name, metrics in checkpoint_test_metrics.items()
        if name.startswith("phase2_")
    }
    phase2_alias_available = best_phase2_alias_epoch is not None

    history_path = paths.experiments / "m2_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "test_metrics": test_metrics,
            "checkpoint_test_metrics": checkpoint_test_metrics,
            "training_time_min": elapsed / 60,
            "best_val_loss": best_val_data,
            "best_val_score": best_val_score,
            "best_data_epoch": best_data_epoch,
            "best_score_epoch": best_score_epoch,
            "best_early_metric": best_early_metric,
            "best_early_epoch": best_early_epoch,
            "best_alias_metric": best_alias_metric,
            "best_alias_epoch": best_alias_epoch,
            "best_pre_phase2_score": best_pre_phase2_score,
            "best_pre_phase2_score_epoch": best_pre_phase2_score_epoch,
            "best_pre_phase2_alias_metric": best_pre_phase2_alias_metric,
            "best_pre_phase2_alias_epoch": best_pre_phase2_alias_epoch,
            "best_phase2_score": best_phase2_score,
            "best_phase2_score_epoch": best_phase2_score_epoch,
            "best_phase2_alias_metric": best_phase2_alias_metric,
            "best_phase2_alias_epoch": best_phase2_alias_epoch,
            "phase2_checkpoint_test_metrics": phase2_checkpoint_test_metrics,
            "phase2_selection_only": phase2_selection_only,
            "phase2_start_epoch": phase2_start_epoch,
            "phase2_start_val_epoch": phase2_start_val_epoch,
            "phase2_validation_started": phase2_validation_started,
            "phase2_alias_available": phase2_alias_available,
            "total_epochs": epoch,
            "warm_start": warm_start,
            "seed": seed,
            "warm_start_checkpoint": warm_ckpt_name,
            "target_mode": mass_cfg.get("target_mode", "true_flux"),
            "objective_mode": mass_cfg.get("objective_mode", "relative_mse"),
            "selection_mass_metric": mass_cfg.get("selection_mass_metric", "objective"),
            "scheduler_metric": scheduler_metric_name,
            "early_stop_metric": early_stop_metric_name,
            "alias_metric": alias_metric_name,
            "phase2_scheduler_metric": phase2_scheduler_metric_name,
            "phase2_early_stop_metric": phase2_early_stop_metric_name,
            "phase2_alias_metric": phase2_alias_metric_name,
        }, f, indent=2)
    logger.info("历史记录保存: %s", history_path)

    save_manifest(
        paths,
        "train_m2",
        {
            "history_path": str(history_path),
            "checkpoints": {
                "best_data": str(ckpt_dir / "m2_best_data.pt"),
                "best_score": str(ckpt_dir / "m2_best_score.pt"),
                "best_alias": str(ckpt_dir / "m2_best.pt"),
                "final": str(ckpt_dir / "m2_final.pt"),
                "phase2_best_score": str(ckpt_dir / "m2_phase2_best_score.pt"),
                "phase2_best_alias": str(ckpt_dir / "m2_phase2_best.pt"),
            },
            "total_epochs": epoch,
            "best_data_epoch": best_data_epoch,
            "best_score_epoch": best_score_epoch,
            "best_early_epoch": best_early_epoch,
            "best_alias_epoch": best_alias_epoch,
            "best_pre_phase2_score_epoch": best_pre_phase2_score_epoch,
            "best_pre_phase2_alias_epoch": best_pre_phase2_alias_epoch,
            "best_phase2_score_epoch": best_phase2_score_epoch,
            "best_phase2_alias_epoch": best_phase2_alias_epoch,
            "best_val_loss": best_val_data,
            "best_val_score": best_val_score,
            "best_early_metric": best_early_metric,
            "best_alias_metric": best_alias_metric,
            "best_pre_phase2_score": best_pre_phase2_score,
            "best_pre_phase2_alias_metric": best_pre_phase2_alias_metric,
            "best_phase2_score": best_phase2_score,
            "best_phase2_alias_metric": best_phase2_alias_metric,
            "training_time_seconds": elapsed,
            "test_metrics": test_metrics,
            "checkpoint_test_metrics": checkpoint_test_metrics,
            "phase2_checkpoint_test_metrics": phase2_checkpoint_test_metrics,
            "seed": seed,
            "warm_start": warm_start,
            "warm_start_checkpoint": warm_ckpt_name,
            "target_mode": mass_cfg.get("target_mode", "true_flux"),
            "objective_mode": mass_cfg.get("objective_mode", "relative_mse"),
            "selection_mass_metric": mass_cfg.get("selection_mass_metric", "objective"),
            "phase2_selection_only": phase2_selection_only,
            "phase2_start_epoch": phase2_start_epoch,
            "phase2_start_val_epoch": phase2_start_val_epoch,
            "phase2_validation_started": phase2_validation_started,
            "phase2_alias_available": phase2_alias_available,
            "val_interval": val_interval,
            "lr_patience_epochs": lr_patience_epochs,
            "scheduler_patience_checks": scheduler_patience_checks,
            "early_stopping_patience_epochs": patience_epochs,
            "scheduler_metric": scheduler_metric_name,
            "early_stopping_metric": early_stop_metric_name,
            "best_alias_metric_name": alias_metric_name,
            "phase2_scheduler_metric": phase2_scheduler_metric_name,
            "phase2_early_stopping_metric": phase2_early_stop_metric_name,
            "phase2_alias_metric_name": phase2_alias_metric_name,
            "phase2_reset_scheduler": phase2_reset_scheduler,
            "phase2_reset_early_stopping": phase2_reset_early_stop,
            "phase2_require_entry_before_early_stop": phase2_require_entry_before_early_stop,
        },
    )


if __name__ == "__main__":
    main()
