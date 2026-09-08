#!/usr/bin/env python3
"""E6-B/E8 step-03 变体（集群跑）: 复用源模型 run 的 scaler，把外推集的
HYDRUS raw 输出打包成 06_evaluate.py 可直接消费的 test.npz。

与 pipeline/03_process_data.py 的差异:
  - 不重新 fit scaler —— 用 --scaler 指定的已训练 run 的 data/processed/scaler.npz
    （DataScaler.load, fit_scaler=False），保证归一化口径与被评估模型一致；
    越界参数归一化后自然落在 [0,1] 之外（外推本意）。
  - 全部 QC 通过样本都进 test.npz（本集无 train/val）。
  - 额外写 metadata/test_row_map.json（test.npz 行序对应的 sample_id 升序表，
    供 make_subsets_B.py / severity_curve.py 做分组）。

依赖 numpy+pandas+主仓库 src（不需要 torch）。在集群上运行。

用法:
  python process_extreme.py --repo $REPO \
      --set-run-dir $REPO/experiments/qtop_func/runs_revision/E6_extreme_peak_v1 \
      --scaler $REPO/experiments/qtop_func/runs/iid_medium_v1/data/processed/scaler.npz \
      --out-run-dir $REPO/experiments/qtop_func/runs_revision/e6e8_eval__E6_extreme_peak_v1__iid_baseline \
      --qc-threshold 0.01
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--set-run-dir", required=True,
                    help="外推集 run（含 data/raw 与 data/parameters/lhs_params.csv）")
    ap.add_argument("--scaler", required=True,
                    help="被评估模型 run 的 data/processed/scaler.npz")
    ap.add_argument("--out-run-dir", required=True, help="评估 run 目录（写 processed/metadata）")
    ap.add_argument("--qc-threshold", type=float, default=0.01)
    ap.add_argument("--qc-balance-stride", type=int, default=1)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import numpy as np
    import pandas as pd
    from src.data_generation.postprocess import (
        DataScaler, build_dataset, load_raw_data, run_quality_control)

    set_run = Path(args.set_run_dir)
    out_run = Path(args.out_run_dir)
    raw_dir = set_run / "data" / "raw"
    params_csv = set_run / "data" / "parameters" / "lhs_params.csv"
    proc_dir = out_run / "data" / "processed"
    meta_dir = out_run / "metadata"
    proc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    samples = load_raw_data(raw_dir)
    if not samples:
        print(f"[ERROR] {raw_dir} 下无 sample_*.npz（先跑 02_run_hydrus.py）")
        return 1

    passed, failed = run_quality_control(
        samples, threshold=args.qc_threshold, balance_stride=args.qc_balance_stride)

    params_df = pd.read_csv(params_csv, index_col="sample_id")
    n_expected = len(params_df)

    # load_raw_data 已按文件名排序 => passed 按 sample_id 升序，即 test.npz 行序
    passed = sorted(passed, key=lambda s: s["sample_id"])
    scaler = DataScaler.load(args.scaler)
    test_data = build_dataset(passed, params_df, scaler=scaler, fit_scaler=False)

    np.savez_compressed(
        proc_dir / "test.npz",
        branch_inputs=test_data["branch_inputs"],
        trunk_inputs=test_data["trunk_inputs"],
        h_targets=test_data["h_targets"],
        c_targets=test_data["c_targets"],
        h_raw=test_data["h_raw"],
        c_raw=test_data["c_raw"],
        water_flux_top=test_data["water_flux_top"],
        water_flux_bottom=test_data["water_flux_bottom"],
        water_cum_top=test_data["water_cum_top"],
        water_cum_bottom=test_data["water_cum_bottom"],
        water_cum_net=test_data["water_cum_net"],
        solute_flux_top=test_data["solute_flux_top"],
        solute_flux_bottom=test_data["solute_flux_bottom"],
        solute_cum_top=test_data["solute_cum_top"],
        solute_cum_bottom=test_data["solute_cum_bottom"],
        solute_cum_net=test_data["solute_cum_net"],
        water_balance_rel=test_data["water_balance_rel"],
        solute_balance_rel=test_data["solute_balance_rel"],
        param_keys=test_data["param_keys"],
        branch_keys=test_data["branch_keys"],
        z=test_data["z"],
        t=test_data["t"],
    )
    shutil.copy2(args.scaler, proc_dir / "scaler.npz")

    row_ids = [int(s["sample_id"]) for s in passed]
    sim_missing = sorted(set(params_df.index.astype(int)) -
                         {int(s["sample_id"]) for s in samples})
    qc_summary = {
        "n_expected": int(n_expected),
        "n_raw": len(samples),
        "n_passed": len(passed),
        "n_failed_qc": len(failed),
        "n_missing_sim": len(sim_missing),
        "missing_sim_sample_ids": sim_missing,
        "threshold_fraction": args.qc_threshold,
        "failed_sample_ids": [int(s["sample_id"]) for s in failed],
        "failed_reasons": {str(int(s["sample_id"])): s["fail_reason"] for s in failed},
        "scaler_source": str(args.scaler),
        "set_run_dir": str(set_run),
    }
    with open(meta_dir / "qc_summary.json", "w") as f:
        json.dump(qc_summary, f, indent=2, ensure_ascii=False)
    with open(meta_dir / "test_row_map.json", "w") as f:
        json.dump({"sample_ids": row_ids, "set_run_dir": str(set_run),
                   "params_csv": str(params_csv)}, f, indent=2)

    print(f"[OK] test.npz N={len(passed)} (期望 {n_expected}, 仿真缺失 {len(sim_missing)},"
          f" QC 剔除 {len(failed)}) -> {proc_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
