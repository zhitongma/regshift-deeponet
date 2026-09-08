#!/usr/bin/env python3
"""E2-B 汇总: 从各 run 的 results/experiments/B1_results.json 收集 c-L2 vs σ 曲线数据。

数据点
------
内置清单覆盖:
  - E2 新训练: runs_revision/E2_<scenario>_<sigma>_s<seed>（σ ∈ {(0.1,0.1),(0.2,0.15),(0.5,0.3)}, seed 42/123）
  - 论文主实验 (0.3,0.2): runs/g4_regshift_{iid,ood_peak,ood_time}_s{42,123}（复用, 不重训）
额外端点（σ→0 ≈ Standard DeepONet, σ→∞ ≈ 无界 Shift, 引用主实验/E1 结果）通过
--extra-manifest CSV 注入, 列: scenario,sigma_scale,sigma_shift,seed,run_dir,model,label
（sigma 可填 0 与 inf）。

输出
----
  sigma_sensitivity_long.csv     每 run 一行（h/c 的 rel_l2 mean/median）
  sigma_sensitivity_summary.csv  每 (scenario, σ) 一行（跨种子均值±标准差）
  sigma_sensitivity.png          可选（matplotlib 存在时）

依赖: 仅 numpy + 标准库。
⚠️ iCloud 警告: 主仓库在 iCloud 同步盘, g4 等 run 的 B1_results.json 可能是
dataless 占位文件（实测 20-30KB 的 json 也会占位, 读取内容永久挂起）。**本地跑
汇总前必须先对各 g4 参考 run 的 B1_results.json 执行 brctl download 物化, 或直接
在集群跑汇总**。read_b1() 打开文件前会 stat 探测占位文件, 命中时输出缺失行并提示
brctl download, 不会挂起。
"""

from pathlib import Path
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))

SIGMAS = [("s010_d010", 0.1, 0.1), ("s020_d015", 0.2, 0.15), ("s050_d030", 0.5, 0.3)]
SCENARIOS = ["iid", "ood_peak", "ood_peak_time"]
SEEDS = [42, 123]
# 论文 (0.3,0.2) 参考 run 的目录名映射（注意 ood_peak_time → g4_..._ood_time）
G4_NAME = {"iid": "g4_regshift_iid", "ood_peak": "g4_regshift_ood_peak",
           "ood_peak_time": "g4_regshift_ood_time"}


def default_manifest(repo):
    rows = []
    rev = os.path.join(repo, "experiments/qtop_func/runs_revision")
    main = os.path.join(repo, "experiments/qtop_func/runs")
    for scen in SCENARIOS:
        for tag, bs, bd in SIGMAS:
            for seed in SEEDS:
                rows.append(dict(scenario=scen, sigma_scale=bs, sigma_shift=bd,
                                 seed=seed, model="m1", label=f"E2_{tag}",
                                 run_dir=os.path.join(rev, f"E2_{scen}_{tag}_s{seed}")))
        for seed in SEEDS:  # (0.3, 0.2) 复用论文主实验（与 E2 同为 2 种子, 保持可比）
            rows.append(dict(scenario=scen, sigma_scale=0.3, sigma_shift=0.2,
                             seed=seed, model="m1", label="paper_g4_regshift",
                             run_dir=os.path.join(main, f"{G4_NAME[scen]}_s{seed}")))
    return rows


def guard_icloud_placeholder(path):
    """iCloud dataless 占位探测（同 front_envelope.guard_icloud_placeholder 逻辑,
    但去掉 1MB 门槛——实测 20-30KB 的 B1_results.json 也会占位且读取永久挂起）。
    命中返回错误信息字符串（调用方作为缺失行处理, 不中止全局）, 正常返回 None。
    """
    st = os.stat(path)
    blocks = getattr(st, "st_blocks", None)
    if blocks is None:  # 非 Unix 文件系统无 st_blocks, 无法探测, 放行
        return None
    on_disk = blocks * 512
    if st.st_size > 4096 and on_disk < st.st_size * 0.05:
        return (f"{path} 疑似 iCloud dataless 占位文件 "
                f"(size={st.st_size}, on_disk={on_disk}), 读取会永久挂起, 已跳过。"
                f"请先物化: brctl download '{path}' ; 或在集群跑本汇总")
    return None


def read_b1(run_dir, model):
    path = os.path.join(run_dir, "results", "experiments", "B1_results.json")
    if not os.path.exists(path):
        return None, f"缺 {path}"
    err = guard_icloud_placeholder(path)
    if err:
        return None, err
    with open(path) as f:
        res = json.load(f)
    if model not in res:
        return None, f"{path} 中无模型 {model}（可选: {sorted(res)}）"
    out = {}
    for var in ("h", "c"):
        for stat in ("mean", "median"):
            out[f"{var}_rel_l2_{stat}"] = float(res[model][var]["rel_l2"][stat])
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("REPO", DEFAULT_REPO))
    ap.add_argument("--model", default="m1", help="B1_results.json 中的模型键")
    ap.add_argument("--extra-manifest", default=None,
                    help="补充数据点 CSV（σ→0 / σ→∞ 端点等）")
    ap.add_argument("--out-dir", default=os.path.join(PKG_DIR, "results_B"))
    args = ap.parse_args()

    manifest = default_manifest(args.repo)
    for row in manifest:
        row.setdefault("model", args.model)
    if args.extra_manifest:
        with open(args.extra_manifest, newline="") as f:
            for row in csv.DictReader(f):
                row["seed"] = int(row["seed"])
                row["sigma_scale"] = float(row["sigma_scale"])  # 'inf' -> math.inf
                row["sigma_shift"] = float(row["sigma_shift"])
                row.setdefault("model", args.model)
                row.setdefault("label", "extra")
                manifest.append(row)

    long_rows, missing = [], []
    for row in manifest:
        metrics, err = read_b1(row["run_dir"], row["model"])
        if err:
            missing.append(err)
            continue
        long_rows.append({**{k: row[k] for k in
                             ("scenario", "sigma_scale", "sigma_shift", "seed",
                              "label", "model", "run_dir")}, **metrics})

    if not long_rows:
        print("[FAIL] 一个可用 run 都没有。缺失明细:", file=sys.stderr)
        for m in missing:
            print("  -", m, file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    long_path = os.path.join(args.out_dir, "sigma_sensitivity_long.csv")
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)

    # 跨种子汇总
    groups = {}
    for r in long_rows:
        key = (r["scenario"], r["sigma_scale"], r["sigma_shift"], r["label"])
        groups.setdefault(key, []).append(r)
    summary = []
    for (scen, bs, bd, label), rows in sorted(groups.items()):
        entry = {"scenario": scen, "sigma_scale": bs, "sigma_shift": bd,
                 "label": label, "n_seeds": len(rows),
                 "seeds": "/".join(str(r["seed"]) for r in rows)}
        for m in ("c_rel_l2_mean", "c_rel_l2_median", "h_rel_l2_mean", "h_rel_l2_median"):
            vals = np.array([r[m] for r in rows], dtype=float)
            entry[f"{m}__avg"] = float(vals.mean())
            entry[f"{m}__std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary.append(entry)
    summary_path = os.path.join(args.out_dir, "sigma_sensitivity_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"[输出] {long_path} ({len(long_rows)} 行)")
    print(f"[输出] {summary_path} ({len(summary)} 组)")
    if missing:
        print(f"[提示] {len(missing)} 个 run 缺失/无结果, 已跳过:")
        for m in missing:
            print("  -", m)

    try:  # 可选绘图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.2 * len(SCENARIOS), 3.6),
                                 sharey=False)
        if len(SCENARIOS) == 1:
            axes = [axes]
        for ax, scen in zip(axes, SCENARIOS):
            pts = [s for s in summary if s["scenario"] == scen
                   and math.isfinite(s["sigma_scale"]) and s["sigma_scale"] > 0]
            pts.sort(key=lambda s: s["sigma_scale"])
            x = [p["sigma_scale"] for p in pts]
            y = [p["c_rel_l2_mean__avg"] for p in pts]
            e = [p["c_rel_l2_mean__std"] for p in pts]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3)
            for p in pts:  # 标注配对 σδ
                ax.annotate(f"σδ={p['sigma_shift']}", (p["sigma_scale"],
                            p["c_rel_l2_mean__avg"]), fontsize=7,
                            textcoords="offset points", xytext=(4, 4))
            ax.set_title(scen)
            ax.set_xlabel("σs (transform_bound_scale)")
            ax.set_ylabel("c rel-L2 (mean ± std over seeds)")
        fig.suptitle("E2: RegShift bound sensitivity (c-L2 vs σ)")
        fig.tight_layout()
        png = os.path.join(args.out_dir, "sigma_sensitivity.png")
        fig.savefig(png, dpi=200)
        print(f"[输出] {png}")
    except ImportError:
        print("[提示] 未安装 matplotlib, 跳过绘图（CSV 已生成, 可导入任意工具绘制）")


if __name__ == "__main__":
    main()
