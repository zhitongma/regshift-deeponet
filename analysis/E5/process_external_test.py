#!/usr/bin/env python3
"""E5 步骤 4 (集群/装有 numpy+pandas 的机器跑, 无需 torch):
外部测试集后处理 —— 复用源 run 的 scaler, 绝不重新 fit。

⚠️ 红线: 主仓库 pipeline/03_process_data.py 总是在 train split 上
`fit_scaler=True` (03_process_data.py:96-99), **不支持复用 scaler**; 而 E5 的
参数表全部 split=test (无 train 行), 直接跑 03 会在空训练集上 fit 出垃圾
scaler 并毒化归一化。因此外部测试集必须用本脚本处理:
  - DataScaler.load(<源run>/data/processed/scaler.npz)  (postprocess.py:202-212)
  - build_dataset(..., scaler=loaded, fit_scaler=False)  (postprocess.py:266)
  - 把源 scaler.npz 原样复制进本 run 的 processed/, 供 06_evaluate 读取。

复用 QC 与数据装配逻辑均直接 import 主仓库 src (不复制代码):
  load_raw_data / run_quality_control / build_dataset / DataScaler。

用法 (在集群上):
  python3 process_external_test.py \
      --repo /path/to/终极版论文项目_lx \
      --run-dir $REPO/experiments/qtop_func/runs_revision/e5_realrain_data_v1 \
      --source-scaler $REPO/experiments/qtop_func/runs/iid_medium_v1/data/processed/scaler.npz \
      --qc-threshold 0.01

输出: <run-dir>/data/processed/{test.npz, scaler.npz} + metadata/e5_qc_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="E5 外部测试集处理 (复用源 scaler)")
    ap.add_argument("--repo", required=True, help="主仓库根目录 (含 src/)")
    ap.add_argument("--run-dir", required=True, help="E5 数据 run 目录 (含 data/raw)")
    ap.add_argument("--source-scaler", required=True,
                    help="源 run 的 data/processed/scaler.npz (训练时所用)")
    ap.add_argument("--qc-threshold", type=float, default=0.01,
                    help="质量平衡 QC 阈值 (小数), 与训练配置 data.qc_mass_balance_threshold 一致")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    from src.data_generation.postprocess import (  # noqa: E402
        DataScaler, build_dataset, load_raw_data, run_quality_control,
    )

    run_dir = Path(args.run_dir).resolve()
    raw_dir = run_dir / "data" / "raw"
    proc_dir = run_dir / "data" / "processed"
    meta_dir = run_dir / "metadata"
    params_csv = run_dir / "data" / "parameters" / "lhs_params.csv"
    proc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    src_scaler_path = Path(args.source_scaler).resolve()
    for p, what in [(raw_dir, "raw 数据目录"), (params_csv, "参数表"),
                    (src_scaler_path, "源 scaler")]:
        if not p.exists():
            sys.exit(f"[FAIL] {what}不存在: {p}")

    # 1. 加载 + QC (与 03_process_data.py 相同链路)
    samples = load_raw_data(raw_dir)
    if not samples:
        sys.exit("[FAIL] data/raw 下没有 sample_*.npz, 请先跑 02_run_hydrus.py")
    passed, failed = run_quality_control(samples, threshold=args.qc_threshold)
    print(f"QC: {len(passed)} 合格 / {len(failed)} 不合格 (共 {len(samples)})")

    params_df = pd.read_csv(params_csv, index_col="sample_id")
    test_samples = [s for s in passed if s["sample_id"] in params_df.index]
    if not test_samples:
        sys.exit("[FAIL] QC 后无可用外部测试样本")

    # 2. 复用源 scaler —— 红线: fit_scaler 必须为 False
    scaler = DataScaler.load(src_scaler_path)
    test_data = build_dataset(test_samples, params_df, scaler=scaler, fit_scaler=False)

    # 3. 完整性检查
    n_branch = test_data["branch_inputs"].shape[1]
    if scaler.input_min.shape[0] != n_branch:
        sys.exit(f"[FAIL] branch 维度 {n_branch} 与源 scaler 维度 "
                 f"{scaler.input_min.shape[0]} 不一致 —— 参数表列与训练时不同!")
    bmin, bmax = float(test_data["branch_inputs"].min()), float(test_data["branch_inputs"].max())
    if bmin < -0.05 or bmax > 1.05:
        print(f"[WARN] 归一化 branch 输入超出 [0,1] 包络: [{bmin:.3f}, {bmax:.3f}] "
              "(外部样本存在幅值外推, 请核对是否有参数越出训练范围)")

    # 4. 保存 test.npz (键与 03_process_data.py:102-128 一致) + 源 scaler 原样复制
    out = proc_dir / "test.npz"
    np.savez_compressed(
        out,
        **{k: test_data[k] for k in [
            "branch_inputs", "trunk_inputs", "h_targets", "c_targets",
            "h_raw", "c_raw",
            "water_flux_top", "water_flux_bottom",
            "water_cum_top", "water_cum_bottom", "water_cum_net",
            "solute_flux_top", "solute_flux_bottom",
            "solute_cum_top", "solute_cum_bottom", "solute_cum_net",
            "water_balance_rel", "solute_balance_rel",
            "param_keys", "branch_keys", "z", "t",
        ]},
    )
    shutil.copy2(src_scaler_path, proc_dir / "scaler.npz")
    assert sha256(proc_dir / "scaler.npz") == sha256(src_scaler_path), \
        "scaler.npz 复制后校验不一致"

    summary = {
        "n_raw": len(samples),
        "n_passed_qc": len(passed),
        "n_failed_qc": len(failed),
        "n_test_saved": len(test_samples),
        "failed_sample_ids": [int(s["sample_id"]) for s in failed],
        "failed_reasons": {str(int(s["sample_id"])): s["fail_reason"] for s in failed},
        "qc_threshold_fraction": args.qc_threshold,
        "source_scaler": str(src_scaler_path),
        "source_scaler_sha256": sha256(src_scaler_path),
        "fit_scaler": False,
        "branch_dim": n_branch,
        "branch_norm_range": [bmin, bmax],
    }
    with open(meta_dir / "e5_qc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] 保存 {out} (branch {test_data['branch_inputs'].shape}, "
          f"targets {test_data['h_targets'].shape})")
    print(f"[OK] scaler.npz 复用自 {src_scaler_path} (未重新 fit)")
    print(f"[OK] QC 摘要: {meta_dir / 'e5_qc_summary.json'}")


if __name__ == "__main__":
    main()
