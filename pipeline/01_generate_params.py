#!/usr/bin/env python3
"""
Step 1: 生成 LHS 参数采样并划分数据集。
输出: data/parameters/lhs_params.csv (含 split / OOD 元信息)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.sampling import generate_lhs_samples, q_top_function_enabled, split_dataset
from src.utils.experiment import (
    configure_logger,
    ensure_dirs,
    load_config,
    resolve_artifact_paths,
    save_config_snapshot,
    save_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="生成 LHS 参数采样")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--run-dir", type=str, default=None, help="实验根目录")
    parser.add_argument("--split-mode", type=str, default=None, choices=["iid", "ood"])
    parser.add_argument("--ood-parameter", type=str, default=None)
    parser.add_argument("--ood-quantile", type=float, default=None)
    parser.add_argument("--ood-tail", type=str, default=None, choices=["high", "low"])
    args = parser.parse_args()

    cfg, cfg_path = load_config(PROJECT_ROOT, args.config)
    paths = resolve_artifact_paths(PROJECT_ROOT, cfg, args.run_dir)
    ensure_dirs(paths, "parameters", "logs", "metadata")
    logger = configure_logger(__name__, paths.logs / "generate_params.log")

    dcfg = cfg["data"]
    n_total = dcfg["n_total"]
    n_train = dcfg["n_train"]
    n_val = dcfg["n_val"]
    n_test = dcfg["n_test"]
    seed = dcfg["seed"]
    split_mode = args.split_mode or dcfg.get("split_mode", "iid")
    default_ood_parameter = "q_top_peak" if q_top_function_enabled(cfg) else "q_top"
    ood_parameter = args.ood_parameter or dcfg.get("ood_parameter", default_ood_parameter)
    ood_quantile = args.ood_quantile if args.ood_quantile is not None else dcfg.get("ood_quantile", 0.9)
    ood_tail = args.ood_tail or dcfg.get("ood_tail", "high")

    logger.info("生成 %d 组 LHS 参数采样 (seed=%d)", n_total, seed)
    logger.info(
        "数据划分: split_mode=%s, train=%d, val=%d, test=%d",
        split_mode,
        n_train,
        n_val,
        n_test,
    )
    if split_mode == "ood":
        logger.info(
            "OOD 设置: parameter=%s, quantile=%.3f, tail=%s",
            ood_parameter,
            ood_quantile,
            ood_tail,
        )

    if q_top_function_enabled(cfg):
        logger.info("启用 q_top(t) 函数输入采样")

    df = generate_lhs_samples(n_total, seed=seed, cfg=cfg)
    train_df, val_df, test_df = split_dataset(
        df,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        split_mode=split_mode,
        ood_parameter=ood_parameter,
        ood_quantile=ood_quantile,
        ood_tail=ood_tail,
    )

    full_df = pd.concat([train_df, val_df, test_df]).sort_index()
    out_path = paths.parameters / "lhs_params.csv"
    full_df.to_csv(out_path, index_label="sample_id")

    save_config_snapshot(
        cfg,
        cfg_path,
        paths,
        "generate_params",
        extra={
            "split_mode": split_mode,
            "ood_parameter": ood_parameter,
            "ood_quantile": ood_quantile,
            "ood_tail": ood_tail,
        },
    )
    save_manifest(
        paths,
        "generate_params",
        {
            "output_csv": str(out_path),
            "n_total": n_total,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "split_mode": split_mode,
            "seed": seed,
            "summary": full_df.describe().round(6).to_dict(),
        },
    )

    logger.info("参数文件已保存: %s", out_path)
    logger.info("训练集: %d, 验证集: %d, 测试集: %d", len(train_df), len(val_df), len(test_df))
    if split_mode == "ood":
        logger.info("OOD 测试样本数: %d", int(test_df["is_ood"].sum()))


if __name__ == "__main__":
    main()
