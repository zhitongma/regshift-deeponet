#!/usr/bin/env python3
"""E6/E8: 按 subsets JSON（regrade_tails.py 或 make_subsets_B.py 的输出）从已有
B1_<model>_predictions.npz / B3_<model>_hydro.npz 重算各分档误差。

⚠️ 需要读取 .npz —— 在集群（或已完整物化的仓库副本）上运行，不要在本机
iCloud dataless 仓库上跑。numpy-only。

对每个 run x model x axis x bin 输出:
  rel_l2_h / rel_l2_c 的 mean 与 median（逐样本 ||pred-ref||2/||ref||2，
  与 src/evaluation/metrics.relative_l2_error 同式）、bt_mae_hours、n。

一致性断言: B1 npz 的 N 必须等于 subsets JSON 的 n_test_rows，否则该 run 的
行号映射不成立（多半是 qc_summary 缺失且有 QC 剔除），报错并跳过。

用法:
  python eval_subsets.py --subsets <outputs/A_subsets.json> \
      --out-csv <outputs/A_binned_errors.csv>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def per_sample_rel_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred = pred.reshape(pred.shape[0], -1).astype(np.float64)
    ref = ref.reshape(ref.shape[0], -1).astype(np.float64)
    num = np.linalg.norm(pred - ref, axis=1)
    den = np.linalg.norm(ref, axis=1) + 1e-12
    return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", required=True, help="A_subsets.json 或 B_subsets.json")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    with open(args.subsets) as f:
        subsets = json.load(f)

    rows_out = []
    n_err = 0
    for key, run in subsets["runs"].items():
        run_dir = Path(run["run_dir"])
        exp_dir = run_dir / "results" / "experiments"
        n_expected = run["n_test_rows"]

        models = run["models"]
        if models == "auto":
            models = sorted(
                p.name[len("B1_"):-len("_predictions.npz")]
                for p in exp_dir.glob("B1_*_predictions.npz")
            )
            if not models:
                print(f"[WARN] {key}: {exp_dir} 下无 B1_*_predictions.npz，跳过")
                continue

        for model in models:
            b1_path = exp_dir / f"B1_{model}_predictions.npz"
            if not b1_path.exists():
                print(f"[WARN] {key}/{model}: 缺 {b1_path.name}，跳过")
                continue
            b1 = np.load(b1_path)
            h_pred, c_pred = b1["h_pred"], b1["c_pred"]
            h_ref, c_ref = b1["h_ref"], b1["c_ref"]
            if h_pred.shape[0] != n_expected:
                print(f"[ERROR] {key}/{model}: B1 npz N={h_pred.shape[0]} != "
                      f"行号映射 n_test_rows={n_expected}。行序不可信，跳过。"
                      f"（检查 qc_summary.json 是否缺失/QC 剔除数不一致）")
                n_err += 1
                continue

            rl2_h = per_sample_rel_l2(h_pred, h_ref)
            rl2_c = per_sample_rel_l2(c_pred, c_ref)

            bt_abs_err = None
            b3_path = exp_dir / f"B3_{model}_hydro.npz"
            if b3_path.exists():
                b3 = np.load(b3_path)
                bt_pred, bt_ref = b3["bt_pred"], b3["bt_ref"]
                if bt_pred.shape[0] == n_expected:
                    finite = np.isfinite(bt_pred) & np.isfinite(bt_ref)
                    bt_abs_err = np.full(bt_pred.shape, np.nan, dtype=np.float64)
                    bt_abs_err[finite] = np.abs(
                        bt_pred[finite] - bt_ref[finite]
                    )
                else:
                    print(f"[WARN] {key}/{model}: B3 npz N 不匹配，bt_mae 置空")

            for axis, ax in run["axes"].items():
                for label, b in ax["bins"].items():
                    idx = np.asarray(b["row_indices"], dtype=int)
                    if idx.size == 0:
                        continue
                    row = {
                        "run_key": key,
                        "scenario": run["scenario"],
                        "family": run["family"],
                        "model": model,
                        "axis": axis,
                        "bin": label,
                        "n": int(idx.size),
                        "rel_l2_h_mean": float(rl2_h[idx].mean()),
                        "rel_l2_h_median": float(np.median(rl2_h[idx])),
                        "rel_l2_c_mean": float(rl2_c[idx].mean()),
                        "rel_l2_c_median": float(np.median(rl2_c[idx])),
                        "bt_mae_hours": "",
                        "n_bt": "",
                    }
                    if bt_abs_err is not None:
                        sel = bt_abs_err[idx]
                        n_bt = int(np.isfinite(sel).sum())
                        row["n_bt"] = n_bt
                        row["bt_mae_hours"] = (
                            float(np.nanmean(sel)) if n_bt > 0 else "")
                    rows_out.append(row)
            print(f"[OK] {key}/{model}")

    if not rows_out:
        print("无输出行，检查 subsets JSON 与结果文件路径")
        return 1

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows_out[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n输出: {out_path}  ({len(rows_out)} 行)")
    return 2 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
