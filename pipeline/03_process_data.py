#!/usr/bin/env python3
"""
Step 3: 数据后处理 + 质量控制 + 归一化 + 数据集划分。
输出: data/processed/{train,val,test}.npz + scaler.npz + QC 摘要
"""

import os
import argparse
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.postprocess import (
    load_raw_data,
    run_quality_control,
    build_dataset,
    DataScaler,
)
from src.utils.experiment import (
    configure_logger,
    ensure_dirs,
    load_config,
    resolve_artifact_paths,
    save_config_snapshot,
    save_json,
    save_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="数据后处理")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--run-dir", type=str, default=None, help="实验根目录")
    args = parser.parse_args()

    cfg, cfg_path = load_config(PROJECT_ROOT, args.config)
    paths = resolve_artifact_paths(PROJECT_ROOT, cfg, args.run_dir)
    ensure_dirs(paths, "raw_data", "processed_data", "parameters", "logs", "metadata")
    logger = configure_logger(__name__, paths.logs / "process_data.log")

    raw_dir = paths.raw_data
    proc_dir = paths.processed_data
    params_path = paths.parameters / "lhs_params.csv"
    t_start = time.time()

    # 1. 加载原始数据
    samples = load_raw_data(raw_dir)
    if not samples:
        logger.error("未找到原始数据! 请先运行 scripts/02_run_hydrus.py")
        sys.exit(1)

    # 2. 质量控制
    threshold = cfg["data"]["qc_mass_balance_threshold"]
    qc_balance_stride = int(cfg["data"].get("qc_balance_stride", 1))
    passed, failed = run_quality_control(
        samples,
        threshold=threshold,
        balance_stride=qc_balance_stride,
    )

    if failed:
        fail_ids = [s["sample_id"] for s in failed]
        logger.info("不合格样本 ID: %s", fail_ids)

    # 3. 按 split 划分
    params_df = pd.read_csv(params_path, index_col="sample_id")
    passed_ids = {s["sample_id"] for s in passed}

    train_samples, val_samples, test_samples = [], [], []
    for s in passed:
        sid = s["sample_id"]
        if sid in params_df.index:
            split = params_df.loc[sid, "split"]
            if split == "train":
                train_samples.append(s)
            elif split == "val":
                val_samples.append(s)
            elif split == "test":
                test_samples.append(s)

    logger.info("划分结果: train=%d, val=%d, test=%d",
                len(train_samples), len(val_samples), len(test_samples))

    # 4. 构建数据集 (在训练集上 fit scaler)
    scaler = DataScaler()
    train_data = build_dataset(train_samples, params_df, scaler=scaler, fit_scaler=True)
    val_data = build_dataset(val_samples, params_df, scaler=scaler, fit_scaler=False)
    test_data = build_dataset(test_samples, params_df, scaler=scaler, fit_scaler=False)

    # 5. 保存
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out = proc_dir / f"{name}.npz"
        np.savez_compressed(
            out,
            branch_inputs=data["branch_inputs"],
            trunk_inputs=data["trunk_inputs"],
            h_targets=data["h_targets"],
            c_targets=data["c_targets"],
            h_raw=data["h_raw"],
            c_raw=data["c_raw"],
            water_flux_top=data["water_flux_top"],
            water_flux_bottom=data["water_flux_bottom"],
            water_cum_top=data["water_cum_top"],
            water_cum_bottom=data["water_cum_bottom"],
            water_cum_net=data["water_cum_net"],
            solute_flux_top=data["solute_flux_top"],
            solute_flux_bottom=data["solute_flux_bottom"],
            solute_cum_top=data["solute_cum_top"],
            solute_cum_bottom=data["solute_cum_bottom"],
            solute_cum_net=data["solute_cum_net"],
            water_balance_rel=data["water_balance_rel"],
            solute_balance_rel=data["solute_balance_rel"],
            param_keys=data["param_keys"],
            branch_keys=data["branch_keys"],
            z=data["z"],
            t=data["t"],
        )
        logger.info("保存: %s  (branch: %s, targets: %s)",
                     out.name, data["branch_inputs"].shape, data["h_targets"].shape)

    scaler.save(proc_dir / "scaler.npz")
    logger.info("保存: scaler.npz")

    qc_summary = {
        "n_raw": len(samples),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "threshold_fraction": threshold,
        "balance_stride": qc_balance_stride,
        "failed_sample_ids": [int(s["sample_id"]) for s in failed],
        "failed_reasons": {
            str(int(s["sample_id"])): s["fail_reason"]
            for s in failed
        },
    }
    save_json(paths.metadata / "qc_summary.json", qc_summary)
    save_config_snapshot(cfg, cfg_path, paths, "process_data")
    save_manifest(
        paths,
        "process_data",
        {
            "raw_dir": str(raw_dir),
            "processed_dir": str(proc_dir),
            "params_csv": str(params_path),
            "elapsed_seconds": time.time() - t_start,
            "qc_threshold_fraction": threshold,
            "qc_balance_stride": qc_balance_stride,
            "splits": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "qc_summary_path": str(paths.metadata / "qc_summary.json"),
        },
    )

    # 6. 打印数据概览
    logger.info("\n数据概览:")
    logger.info("  h 范围: [%.2f, %.2f]", train_data["h_raw"].min(), train_data["h_raw"].max())
    logger.info("  c 范围: [%.4f, %.4f]", train_data["c_raw"].min(), train_data["c_raw"].max())
    logger.info("  branch 输入维度: %d", train_data["branch_inputs"].shape[1])
    logger.info("  trunk 输入维度: %d", train_data["trunk_inputs"].shape[1])
    logger.info("  每样本时空点数: %d", train_data["h_targets"].shape[1])
    logger.info("  质量守恒序列长度: %d", train_data["water_cum_net"].shape[1])


if __name__ == "__main__":
    main()
