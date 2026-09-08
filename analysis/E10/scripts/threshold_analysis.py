#!/usr/bin/env python3
"""
E10 步骤 3 — MBE 自诊断阈值分析（precision/recall 风格表），
为 Discussion 的使用准则（"MBE > 10% 触发 HYDRUS 复核"）提供数字。

两级分析：
  run 级（本地可跑，只依赖步骤 2 的 mbe_vs_error_scatter.csv）：
      以 run×模型 的 mbe_w_mean_final 为诊断量、c_l2_mean 为真值误差，
      对阈值 5/10/15% 计算：被拦截行中高误差行占比（precision）、
      高误差行被拦截比例（recall）、拦截率、基率。
  sample 级（集群，或 B1 json + B2_<model>_mbe.npz 物化后；--sample-level 开启）：
      逐测试样本配对 B1_results.json 的 per-sample c 相对 L2（values 列表，
      顺序与 B2 npz 的样本轴一致——两者都按 test.npz 样本序循环写出，
      见 06_evaluate.py run_b1/run_b2）与 B2_<key>_mbe.npz 的 mbe_w[:, -1]。

高误差判定：--error-cutoff（默认 0.2，对应论文场误差量级上缘），
或 --error-quantile Q（用池内分位数替代固定阈值，两者同时给时 quantile 优先）。

用法：
    python3 threshold_analysis.py --scatter-csv <包>/outputs/mbe_vs_error_scatter.csv \
        --out-dir <包>/outputs
    python3 threshold_analysis.py --scatter-csv ... --sample-level \
        --runs-root "$REPO/experiments/qtop_func" --out-dir <包>/outputs
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbe_common import safe_load_json, safe_load_npz, write_csv  # noqa: E402
import mbe_vs_error  # noqa: E402  # 复用默认 run 清单


def read_scatter_csv(path):
    rows = []
    with open(path) as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def confusion_table(mbe, err, thresholds, cutoff, scope, level):
    """返回表行：每个阈值一行。mbe/err 一一对应（%和无量纲）。"""
    mask = np.isfinite(mbe) & np.isfinite(err)
    mbe, err = mbe[mask], err[mask]
    n = len(mbe)
    high = err > cutoff
    out = []
    for thr in thresholds:
        flag = mbe > thr
        n_flag = int(flag.sum())
        n_high = int(high.sum())
        tp = int((flag & high).sum())
        precision = tp / n_flag if n_flag else float("nan")
        recall = tp / n_high if n_high else float("nan")
        # 未拦截池中高误差残留占比（漏检风险）
        n_pass = n - n_flag
        miss_rate_in_pass = (n_high - tp) / n_pass if n_pass else float("nan")
        out.append([level, scope, f"{cutoff:.4f}", thr, n, n_high, n_flag,
                    f"{n_flag / n:.4f}" if n else "nan",
                    f"{n_high / n:.4f}" if n else "nan",
                    f"{precision:.4f}" if precision == precision else "nan",
                    f"{recall:.4f}" if recall == recall else "nan",
                    f"{miss_rate_in_pass:.4f}" if miss_rate_in_pass == miss_rate_in_pass else "nan"])
    return out


HEADER = ["level", "scope", "error_cutoff", "mbe_threshold_pct", "n", "n_high_error",
          "n_flagged", "flag_rate", "high_error_base_rate",
          "precision_high_in_flagged", "recall_high_flagged",
          "residual_high_rate_in_passed"]


def print_table(rows):
    print("  " + " | ".join(HEADER))
    for r in rows:
        print("  " + " | ".join(str(v) for v in r))


def sample_level_rows(runs_root, thresholds, cutoff, quantile):
    """集群/物化后：逐样本 (c-L2, MBE_w final) 池化。"""
    pooled = {}  # scope -> (mbe list, err list)
    skipped = []
    for run_rel, run_family, scen, _seed in mbe_vs_error.default_manifest():
        exp_dir = os.path.join(runs_root, run_rel, "results", "experiments")
        b1, e1 = safe_load_json(os.path.join(exp_dir, "B1_results.json"))
        if e1:
            skipped.append((run_rel, f"B1: {e1}"))
            continue
        for key in b1:
            if key == "hydrus_reference":
                continue
            vals = (b1.get(key, {}).get("c", {}).get("rel_l2", {}) or {}).get("values")
            if not vals:
                continue
            npz_path = os.path.join(exp_dir, f"B2_{key}_mbe.npz")
            npz, e2 = safe_load_npz(npz_path)
            if e2:
                skipped.append((f"{run_rel}:{key}", f"B2 npz: {e2}"))
                continue
            mbe_w = np.asarray(npz["mbe_w"])[:, -1]
            err = np.asarray(vals, dtype=np.float64)
            if len(mbe_w) != len(err):
                skipped.append((f"{run_rel}:{key}",
                                f"样本数不一致 B1={len(err)} B2={len(mbe_w)}"))
                continue
            for scope in ("all", scen):
                pooled.setdefault(scope, ([], []))
                pooled[scope][0].extend(mbe_w.tolist())
                pooled[scope][1].extend(err.tolist())
    if skipped:
        print(f"[E10] sample 级跳过 {len(skipped)} 项：")
        for r, why in skipped:
            print(f"    - {r}: {why}")
    rows = []
    for scope, (mbe, err) in sorted(pooled.items()):
        mbe = np.asarray(mbe)
        err = np.asarray(err)
        cut = cutoff
        if quantile is not None:
            cut = float(np.nanquantile(err, quantile))
            print(f"[E10] sample 级 scope={scope}: 分位数 {quantile} -> cutoff={cut:.4f}")
        rows.extend(confusion_table(mbe, err, thresholds, cut, scope, "sample"))
    return rows


def main():
    ap = argparse.ArgumentParser(description="E10 MBE 阈值 precision/recall 分析")
    ap.add_argument("--scatter-csv", default="outputs/mbe_vs_error_scatter.csv")
    ap.add_argument("--thresholds", default="5,10,15", help="MBE 阈值 [%%]")
    ap.add_argument("--error-cutoff", type=float, default=0.2,
                    help="高误差判定的 c-L2 固定阈值（默认 0.2）")
    ap.add_argument("--error-quantile", type=float, default=None,
                    help="用池内分位数(0-1)定义高误差（给出时覆盖 --error-cutoff）")
    ap.add_argument("--sample-level", action="store_true",
                    help="启用逐样本分析（需 B2_*_mbe.npz 物化/集群）")
    ap.add_argument("--runs-root", default="",
                    help="sample 级所需：qtop_func 实验根目录")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    all_rows = []

    # ---------- run 级 ----------
    if not os.path.exists(args.scatter_csv):
        sys.exit(f"[E10] 找不到 {args.scatter_csv}——先跑 mbe_vs_error.py")
    rows = read_scatter_csv(args.scatter_csv)

    def col(rows_, name):
        return np.array([float(r[name]) if r.get(name) not in ("", "nan", None)
                         else np.nan for r in rows_])

    scopes = {"all": rows}
    for scen in sorted(set(r["scenario"] for r in rows)):
        scopes[scen] = [r for r in rows if r["scenario"] == scen]

    print("[E10] ===== run 级 阈值表（诊断量 mbe_w_mean_final，真值 c_l2_mean）=====")
    for scope, srows in scopes.items():
        err = col(srows, "c_l2_mean")
        mbe = col(srows, "mbe_w_mean_final")
        cut = args.error_cutoff
        if args.error_quantile is not None:
            cut = float(np.nanquantile(err[np.isfinite(err)], args.error_quantile))
            print(f"[E10] run 级 scope={scope}: 分位数 {args.error_quantile} -> cutoff={cut:.4f}")
        all_rows.extend(confusion_table(mbe, err, thresholds, cut, scope, "run"))
    print_table([r for r in all_rows if r[0] == "run"])

    # ---------- sample 级 ----------
    if args.sample_level:
        if not args.runs_root:
            sys.exit("[E10] --sample-level 需要 --runs-root")
        print("[E10] ===== sample 级 阈值表（逐测试样本）=====")
        srows = sample_level_rows(args.runs_root, thresholds,
                                  args.error_cutoff, args.error_quantile)
        all_rows.extend(srows)
        print_table(srows)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "threshold_analysis.csv")
    write_csv(out_csv, HEADER, all_rows)
    print(f"[E10] 写出 {out_csv}")


if __name__ == "__main__":
    main()
