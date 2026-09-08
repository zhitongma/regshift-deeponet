#!/usr/bin/env python3
"""
Step 6: 运行 B1-B4 四组基础实验

B1: 核心精度验证 (M1 vs M2 vs HYDRUS)
B2: 守恒性增强验证
B3: 水文学过程指标 (锋面位置, 穿透时间)
B4: 计算效率

输出: results/experiments/B{1,2,3,4}_*.json + .npz
"""

import os
import argparse
import json
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deeponet import build_model
from src.models.shift_deeponet import build_shift_deeponet
from src.models.crossattn_deeponet import build_crossattn_deeponet
from src.models.fnn_baseline import build_fnn_model


def build_model_dispatch(model_cfg: dict):
    """Dispatch model creation based on 'arch' field in config."""
    arch = model_cfg.get("arch", "deeponet")
    if arch == "shift_deeponet":
        return build_shift_deeponet(model_cfg)
    if arch == "crossattn_deeponet":
        return build_crossattn_deeponet(model_cfg)
    return build_model(model_cfg)
from src.models.fno import build_fno_model
from src.models.grid_fnn import build_grid_fnn_model
from src.data_generation.postprocess import BRANCH_KEYS, DataScaler
from src.evaluation.metrics import (
    compute_ml_metrics_batch,
    mass_balance_error_water,
    mass_balance_error_solute,
    wetting_front_position,
    concentration_front_position,
    breakthrough_time,
    total_cost_curves,
)
from src.utils.normalization import vg_theta
from src.utils.experiment import (
    configure_logger,
    ensure_dirs,
    load_config,
    resolve_artifact_paths,
    save_config_snapshot,
    save_manifest,
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _time_curve_summary(values: np.ndarray, t: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    mean_curve = np.nanmean(values, axis=0)
    median_curve = np.nanmedian(values, axis=0)
    if len(t) > 1:
        auc_mean = float(np.trapz(mean_curve, t))
        auc_median = float(np.trapz(median_curve, t))
    else:
        auc_mean = float(mean_curve[0])
        auc_median = float(median_curve[0])
    if values.shape[1] > 1:
        time_mean = float(np.nanmean(values[:, 1:]))
    else:
        time_mean = float(np.nanmean(values))
    return {
        "time_mean_pct": time_mean,
        "mean_curve_pct": mean_curve.tolist(),
        "median_curve_pct": median_curve.tolist(),
        "auc_mean_pct_hour": auc_mean,
        "auc_median_pct_hour": auc_median,
        "t_hours": t.tolist(),
    }


def get_model_history_name(model_name: str) -> str:
    """把多个 checkpoint 视角映射回各自的训练历史文件。"""
    if model_name.startswith("m2_"):
        return "m2"
    if model_name.startswith("fnn_mass_scratch"):
        return "fnn_mass_scratch"
    if model_name.startswith("fnn_mass"):
        return "fnn_mass"
    if model_name == "fnn":
        return "fnn"
    if model_name == "grid_fnn":
        return "grid_fnn"
    experimental = ["pod_deeponet", "shift_deeponet", "crossattn_deeponet", "film_fno"]
    for exp in experimental:
        if model_name.startswith(exp):
            return exp
    return model_name


def load_model_and_data(paths, cfg, device, logger):
    """加载模型和测试数据。"""
    proc_dir = paths.processed_data
    ckpt_dir = paths.checkpoints

    test_data = np.load(proc_dir / "test.npz")
    scaler = DataScaler.load(proc_dir / "scaler.npz")

    branch = torch.from_numpy(test_data["branch_inputs"]).to(device)
    trunk = torch.from_numpy(test_data["trunk_inputs"]).to(device)

    # 原始 (未归一化) 数据
    h_raw = test_data["h_raw"]  # (N, n_z*n_t)
    c_raw = test_data["c_raw"]
    z = test_data["z"]
    t = test_data["t"]

    # 加载 M1 和 M2 的不同 checkpoint 视角
    models = {}
    model_specs = [("m1", "m1_best.pt")]
    if (ckpt_dir / "m2_best_data.pt").exists() or (ckpt_dir / "m2_best_score.pt").exists():
        if (ckpt_dir / "m2_best_data.pt").exists():
            model_specs.append(("m2_data", "m2_best_data.pt"))
        if (ckpt_dir / "m2_best_score.pt").exists():
            model_specs.append(("m2_score", "m2_best_score.pt"))
        if (ckpt_dir / "m2_phase2_best_score.pt").exists():
            model_specs.append(("m2_phase2_score", "m2_phase2_best_score.pt"))
    elif (ckpt_dir / "m2_best.pt").exists():
        model_specs.append(("m2", "m2_best.pt"))

    for name, ckpt_name in model_specs:
        model = build_model_dispatch(cfg["model"]).to(device)
        ckpt = ckpt_dir / ckpt_name
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            models[name] = model
            logger.info("加载模型: %s", ckpt.name)
        else:
            logger.warning("模型文件不存在: %s (跳过)", ckpt)

    if (ckpt_dir / "fnn_best.pt").exists():
        fnn = build_fnn_model(cfg["model"]).to(device)
        fnn.load_state_dict(torch.load(ckpt_dir / "fnn_best.pt", map_location=device, weights_only=True))
        fnn.eval()
        models["fnn"] = fnn
        logger.info("加载模型: fnn_best.pt")

    if (ckpt_dir / "grid_fnn_best.pt").exists():
        grid_cfg = {**cfg["model"], "grid_fnn": cfg.get("grid_fnn", {}), "physics": cfg.get("physics", {})}
        grid_fnn = build_grid_fnn_model(grid_cfg).to(device)
        grid_fnn.load_state_dict(
            torch.load(ckpt_dir / "grid_fnn_best.pt", map_location=device, weights_only=True)
        )
        grid_fnn.eval()
        models["grid_fnn"] = grid_fnn
        logger.info("加载模型: grid_fnn_best.pt")

    fnn_mass_variants = [
        ("fnn_mass_score", "fnn_mass_best_score.pt"),
        ("fnn_mass", "fnn_mass_best.pt"),
        ("fnn_mass_scratch_score", "fnn_mass_scratch_best_score.pt"),
        ("fnn_mass_scratch", "fnn_mass_scratch_best.pt"),
    ]
    for fnn_mass_name, fnn_mass_ckpt in fnn_mass_variants:
        if (ckpt_dir / fnn_mass_ckpt).exists():
            fnn_mass = build_fnn_model(cfg["model"]).to(device)
            fnn_mass.load_state_dict(
                torch.load(ckpt_dir / fnn_mass_ckpt, map_location=device, weights_only=True)
            )
            fnn_mass.eval()
            models[fnn_mass_name] = fnn_mass
            logger.info("加载模型: %s", fnn_mass_ckpt)

    if (ckpt_dir / "fno_best.pt").exists():
        fno_cfg = {**cfg["model"], "fno": cfg.get("fno", {}), "physics": cfg.get("physics", {})}
        fno = build_fno_model(fno_cfg).to(device)
        fno.load_state_dict(torch.load(ckpt_dir / "fno_best.pt", map_location=device, weights_only=True))
        fno.eval()
        models["fno"] = fno
        logger.info("加载模型: fno_best.pt")

    experimental_archs = [
        ("pod_deeponet", "pod_deeponet_best.pt", "_build_pod_deeponet"),
        ("shift_deeponet", "shift_deeponet_best.pt", "_build_shift_deeponet"),
        ("crossattn_deeponet", "crossattn_deeponet_best.pt", "_build_crossattn_deeponet"),
        ("film_fno", "film_fno_best.pt", "_build_film_fno"),
    ]
    for exp_name, exp_ckpt, builder_name in experimental_archs:
        if not (ckpt_dir / exp_ckpt).exists():
            continue
        try:
            if exp_name == "pod_deeponet":
                from src.models.pod_deeponet import build_pod_deeponet
                exp_model = build_pod_deeponet(cfg["model"]).to(device)
                pod_basis_path = ckpt_dir / "pod_basis.npz"
                if pod_basis_path.exists():
                    pod_data = np.load(pod_basis_path)
                    exp_model.pod_basis_h = torch.from_numpy(pod_data["pod_basis_h"]).to(device)
                    exp_model.pod_basis_c = torch.from_numpy(pod_data["pod_basis_c"]).to(device)
                    exp_model.pod_mean_h = torch.from_numpy(pod_data["pod_mean_h"]).to(device)
                    exp_model.pod_mean_c = torch.from_numpy(pod_data["pod_mean_c"]).to(device)
            elif exp_name == "shift_deeponet":
                from src.models.shift_deeponet import build_shift_deeponet
                exp_model = build_shift_deeponet(cfg["model"]).to(device)
            elif exp_name == "crossattn_deeponet":
                from src.models.crossattn_deeponet import build_crossattn_deeponet
                exp_model = build_crossattn_deeponet(cfg["model"]).to(device)
            elif exp_name == "film_fno":
                from src.models.film_fno import build_film_fno
                film_cfg = {**cfg["model"], "fno": cfg.get("fno", {}),
                            "physics": cfg.get("physics", {}), "film_fno": cfg.get("film_fno", {})}
                exp_model = build_film_fno(film_cfg).to(device)
            else:
                continue
            exp_model.load_state_dict(
                torch.load(ckpt_dir / exp_ckpt, map_location=device, weights_only=True)
            )
            exp_model.eval()
            models[exp_name] = exp_model
            logger.info("加载模型: %s", exp_ckpt)
        except Exception as e:
            logger.warning("加载 %s 失败: %s", exp_ckpt, e)

    return models, branch, trunk, h_raw, c_raw, z, t, scaler, test_data


@torch.no_grad()
def predict(model, branch, trunk, scaler, n_z, n_t):
    """模型推理, 返回物理空间的 h 和 c (N, n_z, n_t)。"""
    pred = model(branch, trunk)  # (N, M, 2)

    h_norm = pred[..., 0].cpu().numpy()  # (N, M)
    c_norm = pred[..., 1].cpu().numpy()

    h_phys = scaler.inverse_h(h_norm)
    c_phys = scaler.inverse_c(c_norm)
    c_phys = np.clip(c_phys, 0, None)

    N = h_phys.shape[0]
    return h_phys.reshape(N, n_z, n_t), c_phys.reshape(N, n_z, n_t)


def get_branch_keys(test_data) -> list[str]:
    if "branch_keys" not in test_data.files:
        return list(BRANCH_KEYS)
    return [str(x) for x in test_data["branch_keys"].tolist()]


def get_raw_params(test_data, scaler):
    """反归一化测试集参数。"""
    branch_norm = test_data["branch_inputs"]
    branch_raw = branch_norm * (scaler.input_max - scaler.input_min + 1e-8) + scaler.input_min
    branch_keys = get_branch_keys(test_data)
    return {key: branch_raw[:, i] for i, key in enumerate(branch_keys)}


def run_b1(models, branch, trunk, h_raw, c_raw, scaler, cfg, exp_dir, logger):
    """B1: 核心精度验证。"""
    logger.info("=" * 40 + " B1: 精度验证 " + "=" * 40)
    n_z = cfg["physics"]["n_spatial_nodes"]
    n_t = cfg["physics"]["n_time_steps"]
    N = h_raw.shape[0]

    results = {}
    for name, model in models.items():
        h_pred, c_pred = predict(model, branch, trunk, scaler, n_z, n_t)

        h_ref = h_raw.reshape(N, n_z, n_t)
        c_ref = c_raw.reshape(N, n_z, n_t)

        h_stats = compute_ml_metrics_batch(
            h_pred.reshape(N, -1), h_ref.reshape(N, -1))
        c_stats = compute_ml_metrics_batch(
            c_pred.reshape(N, -1), c_ref.reshape(N, -1))

        results[name] = {"h": h_stats, "c": c_stats}
        logger.info("  %s: h_L2=%.4f (median), c_L2=%.4f (median), "
                     "h_R2=%.4f, c_R2=%.4f",
                     name.upper(),
                     h_stats["rel_l2"]["median"], c_stats["rel_l2"]["median"],
                     h_stats["r2"]["median"], c_stats["r2"]["median"])

        np.savez(exp_dir / f"B1_{name}_predictions.npz",
                 h_pred=h_pred, c_pred=c_pred, h_ref=h_ref, c_ref=c_ref)

    with open(exp_dir / "B1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_b2(models, branch, trunk, h_raw, c_raw, test_data, scaler, cfg, exp_dir, logger):
    """B2: 守恒性增强验证。"""
    logger.info("=" * 40 + " B2: 守恒性 " + "=" * 40)
    n_z = cfg["physics"]["n_spatial_nodes"]
    n_t = cfg["physics"]["n_time_steps"]
    N = h_raw.shape[0]
    dz = float(test_data["z"][1] - test_data["z"][0])
    t_axis = test_data["t"]
    raw_params = get_raw_params(test_data, scaler)

    results = {}
    for name, model in models.items():
        h_pred, c_pred = predict(model, branch, trunk, scaler, n_z, n_t)

        mbe_w_all, mbe_c_all = [], []
        for i in range(N):
            params_i = {k: values[i] for k, values in raw_params.items()}
            theta_pred = vg_theta(h_pred[i], params_i["theta_r"],
                                  params_i["theta_s"], params_i["alpha"],
                                  params_i["n_vg"])

            mbe_w = mass_balance_error_water(theta_pred, test_data["water_cum_net"][i], dz=dz)
            mbe_c = mass_balance_error_solute(
                theta_pred, c_pred[i], test_data["solute_cum_net"][i], dz=dz)

            mbe_w_all.append(mbe_w)
            mbe_c_all.append(mbe_c)

        mbe_w_arr = np.array(mbe_w_all)  # (N, n_t)
        mbe_c_arr = np.array(mbe_c_all)
        mbe_w_curve = _time_curve_summary(mbe_w_arr, t_axis)
        mbe_c_curve = _time_curve_summary(mbe_c_arr, t_axis)

        results[name] = {
            "mbe_w_mean_final": float(np.nanmean(mbe_w_arr[:, -1])),
            "mbe_c_mean_final": float(np.nanmean(mbe_c_arr[:, -1])),
            "mbe_w_median_final": float(np.nanmedian(mbe_w_arr[:, -1])),
            "mbe_c_median_final": float(np.nanmedian(mbe_c_arr[:, -1])),
            "mbe_w_time_mean_pct": mbe_w_curve["time_mean_pct"],
            "mbe_c_time_mean_pct": mbe_c_curve["time_mean_pct"],
            "mbe_w_auc_mean_pct_hour": mbe_w_curve["auc_mean_pct_hour"],
            "mbe_c_auc_mean_pct_hour": mbe_c_curve["auc_mean_pct_hour"],
            "mbe_w_curve": mbe_w_curve,
            "mbe_c_curve": mbe_c_curve,
        }
        logger.info("  %s: MBE_w=%.2f%% (mean), MBE_c=%.2f%% (mean)",
                     name.upper(),
                     results[name]["mbe_w_mean_final"],
                     results[name]["mbe_c_mean_final"])

        np.savez(exp_dir / f"B2_{name}_mbe.npz",
                 mbe_w=mbe_w_arr, mbe_c=mbe_c_arr, t=test_data["t"])

    results["hydrus_reference"] = {
        "mbe_w_mean_final": float(np.nanmean(test_data["water_balance_rel"][:, -1])),
        "mbe_c_mean_final": float(np.nanmean(test_data["solute_balance_rel"][:, -1])),
        "mbe_w_median_final": float(np.nanmedian(test_data["water_balance_rel"][:, -1])),
        "mbe_c_median_final": float(np.nanmedian(test_data["solute_balance_rel"][:, -1])),
        "mbe_w_time_mean_pct": _time_curve_summary(test_data["water_balance_rel"], t_axis)["time_mean_pct"],
        "mbe_c_time_mean_pct": _time_curve_summary(test_data["solute_balance_rel"], t_axis)["time_mean_pct"],
        "mbe_w_auc_mean_pct_hour": _time_curve_summary(test_data["water_balance_rel"], t_axis)["auc_mean_pct_hour"],
        "mbe_c_auc_mean_pct_hour": _time_curve_summary(test_data["solute_balance_rel"], t_axis)["auc_mean_pct_hour"],
        "mbe_w_curve": _time_curve_summary(test_data["water_balance_rel"], t_axis),
        "mbe_c_curve": _time_curve_summary(test_data["solute_balance_rel"], t_axis),
    }
    logger.info(
        "  HYDRUS: MBE_w=%.4f%% (mean), MBE_c=%.4f%% (mean)",
        results["hydrus_reference"]["mbe_w_mean_final"],
        results["hydrus_reference"]["mbe_c_mean_final"],
    )

    with open(exp_dir / "B2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_b3(models, branch, trunk, h_raw, c_raw, test_data, scaler, cfg, exp_dir, logger):
    """B3: 水文学过程指标。"""
    logger.info("=" * 40 + " B3: 过程指标 " + "=" * 40)
    n_z = cfg["physics"]["n_spatial_nodes"]
    n_t = cfg["physics"]["n_time_steps"]
    N = h_raw.shape[0]

    z = test_data["z"]
    t = test_data["t"]
    raw_params = get_raw_params(test_data, scaler)

    h_ref = h_raw.reshape(N, n_z, n_t)
    c_ref = c_raw.reshape(N, n_z, n_t)

    results = {}
    for name, model in models.items():
        h_pred, c_pred = predict(model, branch, trunk, scaler, n_z, n_t)

        front_errors, bt_pred_list, bt_ref_list = [], [], []
        for i in range(N):
            params_i = {k: values[i] for k, values in raw_params.items()}
            theta_s = params_i["theta_s"]
            h_init = params_i.get("h_init", cfg["physics"]["initial_head"])
            theta_init = vg_theta(np.array([h_init]), params_i["theta_r"],
                                  theta_s, params_i["alpha"], params_i["n_vg"])[0]

            # 浓度锋: c_top 可能是标量或函数(c_top_00, c_top_01, ...)
            c_top_keys = [k for k in params_i if k.startswith("c_top_")]
            if "c_top" in params_i:
                c_top_val = params_i["c_top"]
            elif c_top_keys:
                c_top_val = np.mean([params_i[k] for k in c_top_keys])
            else:
                c_top_val = cfg["parameters"]["c_top"]["max"]

            c_front_pred = concentration_front_position(c_pred[i], z, c_top_val)
            c_front_ref = concentration_front_position(c_ref[i], z, c_top_val)

            valid = ~np.isnan(c_front_pred) & ~np.isnan(c_front_ref)
            if np.any(valid):
                err = np.abs(c_front_pred[valid] - c_front_ref[valid])
                front_errors.append(float(np.mean(err)))

            # 穿透时间
            bt_p = breakthrough_time(c_pred[i], t, c_top_val)
            bt_r = breakthrough_time(c_ref[i], t, c_top_val)
            bt_pred_list.append(bt_p)
            bt_ref_list.append(bt_r)

        bt_pred = np.array(bt_pred_list)
        bt_ref = np.array(bt_ref_list)
        valid_bt = np.isfinite(bt_pred) & np.isfinite(bt_ref)

        results[name] = {
            "front_error_mean_cm": float(np.mean(front_errors)) if front_errors else None,
            "bt_mae_hours": float(np.mean(np.abs(bt_pred[valid_bt] - bt_ref[valid_bt]))) if np.any(valid_bt) else None,
            "n_breakthrough": int(np.sum(valid_bt)),
        }
        logger.info("  %s: front_err=%.2f cm, bt_MAE=%.2f h, n_bt=%d",
                     name.upper(),
                     results[name]["front_error_mean_cm"] or 0,
                     results[name]["bt_mae_hours"] or 0,
                     results[name]["n_breakthrough"])

        np.savez(exp_dir / f"B3_{name}_hydro.npz",
                 bt_pred=bt_pred, bt_ref=bt_ref)

    with open(exp_dir / "B3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_b4(models, branch, trunk, paths, scaler, cfg, exp_dir, logger,
           hydrus_single_seconds: float | None = None,
           data_gen_seconds: float | None = None):
    """B4: 计算效率。"""
    logger.info("=" * 40 + " B4: 效率 " + "=" * 40)
    device = branch.device

    results = {}
    for name, model in models.items():
        # 单次推理时间 (预热 + 多次平均)
        with torch.no_grad():
            for _ in range(5):
                _ = model(branch[:1], trunk)

            times = []
            for _ in range(50):
                t0 = time.time()
                _ = model(branch[:1], trunk)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t0)

        t_infer = float(np.median(times)) * 1000  # ms

        results[name] = {"t_infer_ms": t_infer}
        logger.info("  %s: 单次推理 = %.2f ms", name.upper(), t_infer)

    # 加载训练时间
    for name in models:
        hist_name = get_model_history_name(name)
        hist_path = exp_dir / f"{hist_name}_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            results[name]["t_train_min"] = hist.get("training_time_min", 0)

    hydrus_manifest = _load_json(paths.metadata / "hydrus_batch_manifest.json") or {}
    process_manifest = _load_json(paths.metadata / "process_data_manifest.json") or {}
    if hydrus_single_seconds is None:
        hydrus_single_seconds = hydrus_manifest.get("median_sample_seconds")
    if data_gen_seconds is None:
        hydrus_total = hydrus_manifest.get("elapsed_seconds")
        if hydrus_total is not None:
            process_elapsed = process_manifest.get("elapsed_seconds", 0.0)
            data_gen_seconds = hydrus_total + process_elapsed

    if hydrus_single_seconds is None or data_gen_seconds is None:
        logger.warning("缺少实测效率 manifest，跳过总成本曲线")
        with open(exp_dir / "B4_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    for name in models:
        t_train_s = results[name].get("t_train_min", 60) * 60
        t_infer_s = results[name]["t_infer_ms"] / 1000
        curves = total_cost_curves(data_gen_seconds, t_train_s, t_infer_s, hydrus_single_seconds)
        results[name]["cost_curves"] = curves
        results[name]["n_cross"] = curves["n_cross"]
        results[name]["timing_sources"] = {
            "hydrus_single_seconds": hydrus_single_seconds,
            "data_gen_seconds": data_gen_seconds,
        }
        logger.info("  %s: 成本交叉点 N_cross = %d", name.upper(), int(curves["n_cross"]))

    with open(exp_dir / "B4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    parser = argparse.ArgumentParser(description="运行 B1-B4 实验")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated subset of loaded models, e.g. m1,m2_data,m2_phase2_score,grid_fnn")
    parser.add_argument("--hydrus-single-seconds", type=float, default=None)
    parser.add_argument("--data-gen-seconds", type=float, default=None)
    args = parser.parse_args()

    cfg, cfg_path = load_config(PROJECT_ROOT, args.config)
    paths = resolve_artifact_paths(PROJECT_ROOT, cfg, args.run_dir)
    ensure_dirs(paths, "experiments", "figures", "logs", "metadata")
    logger = configure_logger(__name__, paths.logs / "evaluate.log")
    save_config_snapshot(cfg, cfg_path, paths, "evaluate")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = paths.experiments

    models, branch, trunk, h_raw, c_raw, z, t, scaler, test_data = \
        load_model_and_data(paths, cfg, device, logger)

    if args.models:
        requested = [name.strip() for name in args.models.split(",") if name.strip()]
        missing = [name for name in requested if name not in models]
        if missing:
            logger.warning("请求的模型不存在或未加载: %s", missing)
        models = {name: models[name] for name in requested if name in models}

    if not models:
        logger.error("未找到任何训练好的模型! 请先运行 04_train_m1.py / 05_train_m2.py")
        sys.exit(1)

    b1 = run_b1(models, branch, trunk, h_raw, c_raw, scaler, cfg, exp_dir, logger)
    b2 = run_b2(models, branch, trunk, h_raw, c_raw, test_data, scaler, cfg, exp_dir, logger)
    b3 = run_b3(models, branch, trunk, h_raw, c_raw, test_data, scaler, cfg, exp_dir, logger)
    b4 = run_b4(
        models,
        branch,
        trunk,
        paths,
        scaler,
        cfg,
        exp_dir,
        logger,
        hydrus_single_seconds=args.hydrus_single_seconds,
        data_gen_seconds=args.data_gen_seconds,
    )

    save_manifest(
        paths,
        "evaluate",
        {
            "device": str(device),
            "available_models": list(models.keys()),
            "outputs": {
                "b1": str(exp_dir / "B1_results.json"),
                "b2": str(exp_dir / "B2_results.json"),
                "b3": str(exp_dir / "B3_results.json"),
                "b4": str(exp_dir / "B4_results.json"),
            },
            "summaries": {"B1": b1, "B2": b2, "B3": b3, "B4": b4},
        },
    )

    logger.info("\n所有实验完成! 结果保存在: %s", exp_dir)


if __name__ == "__main__":
    main()
