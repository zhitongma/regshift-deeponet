#!/usr/bin/env python3
"""
E10 步骤 2 — 跨全部已有主 benchmark runs 收集 (c-L2, front_error, MBE) 三元组，
手写 Spearman 相关（置换检验 p 值），导出散点 CSV（回应 R1-7 的"自诊断"半问）。

只读纯文本 JSON：
    <run>/results/experiments/B1_results.json   -> h/c 相对 L2（mean/median）
    <run>/results/experiments/B2_results.json   -> mbe_w/mbe_c *_final
    <run>/results/experiments/B3_results.json   -> front_error_mean_cm, bt_mae_hours
不读任何 .pt/.npz；每个 JSON 读取前做 iCloud dataless 防护（挂起风险见 REPO_FACTS §0/§5）。

默认 run 清单覆盖论文主 benchmark（三场景 × 五模型家族 × 三种子）：
    Standard DeepONet (m1/m2_*), Grid-FNN : runs_multiseed/<scen>/seed_{42,43,44}
    RegShift                              : runs/g4_regshift_{iid,ood_peak,ood_time}_s{42,123,456}
    Shift-DeepONet (A4)                   : runs/boost_A4_{iid,ood} (s42) + runs_jhydrology/A4_*
    FNN                                   : runs/fnn_baseline_* (s42) + runs_jhydrology/fnn_*
    FNO                                   : runs_jhydrology/fno_*_s{42,123,456}

用法：
    python3 mbe_vs_error.py --runs-root "$REPO/experiments/qtop_func" --out-dir <包>/outputs
    python3 mbe_vs_error.py --runs-root ... --print-files   # 只打印所需 JSON 路径（供物化检查）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbe_common import safe_load_json, spearman, write_csv  # noqa: E402


def default_manifest():
    """返回 [(run 相对路径, run 架构家族, scenario, seed), ...]。"""
    entries = []
    multiseed = [("iid", "iid_medium_v1"),
                 ("ood_peak", "ood_peak_medium_v1"),
                 ("ood_peak_time", "ood_peak_time_medium_v2")]
    for scen, d in multiseed:
        for s in (42, 43, 44):
            entries.append((f"runs_multiseed/{d}/seed_{s}", "deeponet", scen, s))

    regshift_tags = {"iid": "iid", "ood_peak": "ood_peak", "ood_time": "ood_peak_time"}
    for tag, scen in regshift_tags.items():
        for s in (42, 123, 456):
            entries.append((f"runs/g4_regshift_{tag}_s{s}", "regshift", scen, s))

    # Shift-DeepONet (A4)：seed42 的 iid/ood_peak 在 runs/ 下（见 runs_jhydrology/submission_manifest.json）
    entries.append(("runs/boost_A4_iid", "shift", "iid", 42))
    entries.append(("runs/boost_A4_ood", "shift", "ood_peak", 42))
    for scen in ("iid", "ood_peak"):
        for s in (123, 456):
            entries.append((f"runs_jhydrology/A4_{scen}_s{s}", "shift", scen, s))
            entries.append((f"runs_jhydrology/fnn_{scen}_s{s}", "fnn", scen, s))
    for s in (42, 123, 456):
        entries.append((f"runs_jhydrology/A4_ood_peak_time_s{s}", "shift", "ood_peak_time", s))
        entries.append((f"runs_jhydrology/fnn_ood_peak_time_s{s}", "fnn", "ood_peak_time", s))

    entries.append(("runs/fnn_baseline_iid_v1", "fnn", "iid", 42))
    entries.append(("runs/fnn_baseline_ood_peak_v1", "fnn", "ood_peak", 42))

    for scen in ("iid", "ood_peak", "ood_peak_time"):
        for s in (42, 123, 456):
            entries.append((f"runs_jhydrology/fno_{scen}_s{s}", "fno", scen, s))
    return entries


def load_manifest(path):
    """外部清单 CSV：run_rel,family,scenario,seed（含表头）。"""
    entries = []
    with open(path) as f:
        header = f.readline()
        assert "run_rel" in header, "清单 CSV 需含表头 run_rel,family,scenario,seed"
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) >= 4 and parts[0]:
                entries.append((parts[0], parts[1], parts[2], int(parts[3])))
    return entries


def family_for_key(run_family, key):
    """模型键 -> 家族标签：fnn/fno/grid_fnn 独立成族，m1/m2_* 归属 run 架构。"""
    if key.startswith("fnn"):
        return key
    if key in ("fno", "grid_fnn"):
        return key
    return run_family


def collect_rows(runs_root, entries):
    rows, skipped = [], []
    for run_rel, run_family, scen, seed in entries:
        run_dir = os.path.join(runs_root, run_rel)
        exp_dir = os.path.join(run_dir, "results", "experiments")
        if not os.path.isdir(exp_dir):
            skipped.append((run_rel, "run/results/experiments 目录不存在"))
            continue
        b1, e1 = safe_load_json(os.path.join(exp_dir, "B1_results.json"))
        b2, e2 = safe_load_json(os.path.join(exp_dir, "B2_results.json"))
        b3, e3 = safe_load_json(os.path.join(exp_dir, "B3_results.json"))
        err = e1 or e2 or e3
        if err:
            skipped.append((run_rel, f"B1/B2/B3 JSON: {err}"))
            continue
        for key in b1:
            if key == "hydrus_reference":
                continue
            b2k = b2.get(key, {})
            b3k = b3.get(key, {})

            def _g(d, *ks, default=float("nan")):
                cur = d
                for k in ks:
                    if not isinstance(cur, dict) or k not in cur:
                        return default
                    cur = cur[k]
                return default if cur is None else cur

            rows.append({
                "run": run_rel,
                "family": family_for_key(run_family, key),
                "model_key": key,
                "scenario": scen,
                "seed": seed,
                "h_l2_mean": _g(b1, key, "h", "rel_l2", "mean"),
                "h_l2_median": _g(b1, key, "h", "rel_l2", "median"),
                "c_l2_mean": _g(b1, key, "c", "rel_l2", "mean"),
                "c_l2_median": _g(b1, key, "c", "rel_l2", "median"),
                "front_error_mean_cm": _g(b3k, "front_error_mean_cm"),
                "bt_mae_hours": _g(b3k, "bt_mae_hours"),
                "mbe_w_mean_final": _g(b2k, "mbe_w_mean_final"),
                "mbe_w_median_final": _g(b2k, "mbe_w_median_final"),
                "mbe_c_mean_final": _g(b2k, "mbe_c_mean_final"),
            })
    return rows, skipped


PAIRS = [
    ("mbe_w_mean_final", "c_l2_mean"),
    ("mbe_w_mean_final", "h_l2_mean"),
    ("mbe_w_mean_final", "front_error_mean_cm"),
    ("mbe_w_median_final", "c_l2_median"),
    ("mbe_c_mean_final", "c_l2_mean"),
]


def main():
    ap = argparse.ArgumentParser(description="E10 MBE-误差相关性收集与检验")
    ap.add_argument("--runs-root", required=True,
                    help="qtop_func 实验根目录，本机: $REPO/experiments/qtop_func；"
                         "集群: .../project/qtop_func")
    ap.add_argument("--manifest", default="",
                    help="可选外部 run 清单 CSV（run_rel,family,scenario,seed）")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-files", action="store_true",
                    help="只打印所需 JSON 绝对路径后退出（供物化检查）")
    args = ap.parse_args()

    entries = load_manifest(args.manifest) if args.manifest else default_manifest()

    if args.print_files:
        for run_rel, _, _, _ in entries:
            for name in ("B1_results.json", "B2_results.json", "B3_results.json"):
                print(os.path.join(args.runs_root, run_rel,
                                   "results", "experiments", name))
        return

    rows, skipped = collect_rows(args.runs_root, entries)
    if skipped:
        print(f"[E10] 跳过 {len(skipped)} 个 run：")
        for r, why in skipped:
            print(f"    - {r}: {why}")
    if not rows:
        sys.exit("[E10] 未收集到任何数据行——检查 --runs-root 与文件物化状态")
    print(f"[E10] 收集 {len(rows)} 个 (run × 模型) 数据行，覆盖 "
          f"{len(set(r['run'] for r in rows))} 个 run")

    os.makedirs(args.out_dir, exist_ok=True)
    header = ["run", "family", "model_key", "scenario", "seed",
              "h_l2_mean", "h_l2_median", "c_l2_mean", "c_l2_median",
              "front_error_mean_cm", "bt_mae_hours",
              "mbe_w_mean_final", "mbe_w_median_final", "mbe_c_mean_final"]
    scatter_csv = os.path.join(args.out_dir, "mbe_vs_error_scatter.csv")
    write_csv(scatter_csv, header, [[r[k] for k in header] for r in rows])
    print(f"[E10] 写出 {scatter_csv}")

    # Spearman：总体 + 分场景
    groups = {"all": rows}
    for scen in sorted(set(r["scenario"] for r in rows)):
        groups[scen] = [r for r in rows if r["scenario"] == scen]

    stat_rows = []
    for gname, grows in groups.items():
        for mx, my in PAIRS:
            x = np.array([float(r[mx]) if r[mx] == r[mx] else np.nan for r in grows])
            y = np.array([float(r[my]) if r[my] == r[my] else np.nan for r in grows])
            res = spearman(x, y, n_perm=args.n_perm, seed=args.seed)
            stat_rows.append([gname, mx, my, res["n"],
                              f"{res['rho']:.4f}" if res["rho"] == res["rho"] else "nan",
                              f"{res['p_perm']:.5f}" if res["p_perm"] == res["p_perm"] else "nan",
                              "yes" if (res["p_perm"] == res["p_perm"] and res["p_perm"] < 0.05) else "no"])
            print(f"[E10] Spearman[{gname}] {mx} vs {my}: "
                  f"n={res['n']}, rho={res['rho']:.3f}, p_perm={res['p_perm']:.4g}")

    spearman_csv = os.path.join(args.out_dir, "mbe_vs_error_spearman.csv")
    write_csv(spearman_csv,
              ["group", "x_metric", "y_metric", "n", "spearman_rho",
               "p_perm_two_sided", "significant_at_0.05"],
              stat_rows)
    with open(os.path.join(args.out_dir, "mbe_vs_error_spearman.json"), "w") as f:
        json.dump({"pairs": PAIRS,
                   "n_perm": args.n_perm,
                   "rows": [dict(zip(["group", "x", "y", "n", "rho", "p", "sig"], r))
                            for r in stat_rows]}, f, indent=2, ensure_ascii=False)
    print(f"[E10] 写出 {spearman_csv} 及同名 .json")

    # 可选散点图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        markers = {"iid": "o", "ood_peak": "s", "ood_peak_time": "^"}
        for scen in sorted(set(r["scenario"] for r in rows)):
            xs = [r["mbe_w_mean_final"] for r in rows if r["scenario"] == scen]
            ys = [r["c_l2_mean"] for r in rows if r["scenario"] == scen]
            ax.scatter(xs, ys, marker=markers.get(scen, "o"), alpha=0.7, label=scen)
        ax.set_xlabel("terminal water MBE [%] (run mean)")
        ax.set_ylabel("c relative L2 (run mean)")
        ax.set_title("MBE vs field error across benchmark runs")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "mbe_vs_error_scatter.png")
        fig.savefig(out_png, dpi=200)
        print(f"[E10] 写出 {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[E10] matplotlib 不可用或画图失败（{e}），仅输出 CSV")


if __name__ == "__main__":
    main()
