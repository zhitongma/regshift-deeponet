#!/usr/bin/env python3
"""E5 步骤 2 (本地可跑, 仅需 numpy + 标准库): 训练池 hydrograph vs 真实降雨事件统计对比。

对比对象
----
A. 训练池: 数据源 run 的 data/parameters/lhs_params.csv 中 split=="train" 行的
   q_top_00..q_top_47 (六控制点线性插值序列, cm/h);
B. 真实事件 (重标定后): E5 参数表 lhs_params.csv 的 q_top_* 列 (--events-csv);
   若给 --events-raw-csv (events_raw.csv, mm/h) 则同时报告重标定前的干时占比,
   以便如实呈现 "0.1 cm/h 包络下限抹掉干期" 的效应。

统计量 (逐 hydrograph 计算后汇总 mean/std)
----
1. 滞后 1-8 h 自相关系数;
2. 峰均比 q_peak / q_mean;
3. 湿时占比: frac(q > 0.5 cm/h) 与 frac(q <= 0.1+1e-6) (贴包络下限 = "干期替身");
4. 峰现时间 argmax (h) 分布: 分位数 + 两样本 KS 检验 (手写实现, 渐近 p 值);
5. 逐时强度合并分布的两样本 KS 检验。

输出: --out-dir 下 stats_compare_table.csv + stats_compare.md
      (matplotlib 可用时另存 stats_compare_acf.png, 不可用则自动跳过)。

用法
----
  python3 stats_compare.py \
      --train-csv $REPO/experiments/qtop_func/runs/iid_medium_v1/data/parameters/lhs_params.csv \
      --events-csv <数据run>/data/parameters/lhs_params.csv \
      --events-raw-csv <数据run>/data/parameters/events_raw.csv \
      --out-dir ./results_stats

注意: 主仓库在 iCloud 同步盘上, 若 lhs_params.csv 尚未本地化, 读取会挂起;
先在访达中下载该文件 (或在集群上运行本脚本)。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

MAX_LAG = 8


def load_qtop_rows(path: Path, only_split: str | None = None) -> np.ndarray:
    """读取 lhs_params.csv 的 q_top_00.. 列 (仅标准库 csv, 不依赖 pandas)。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or [])
                if c.startswith("q_top_") and c.split("_")[-1].isdigit()]
        cols.sort(key=lambda c: int(c.split("_")[-1]))
        if not cols:
            raise SystemExit(f"[FAIL] {path} 中未找到 q_top_XX 序列列")
        has_split = "split" in (reader.fieldnames or [])
        rows = []
        for row in reader:
            if only_split is not None and has_split and row["split"] != only_split:
                continue
            rows.append([float(row[c]) for c in cols])
    if not rows:
        raise SystemExit(f"[FAIL] {path} 过滤 split={only_split} 后为空")
    return np.asarray(rows, dtype=float)


def load_raw_events(path: Path) -> np.ndarray:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = [c for c in (reader.fieldnames or []) if c.startswith("r_mm_")]
        cols.sort(key=lambda c: int(c.split("_")[-1]))
        rows = [[float(row[c]) for c in cols] for row in reader]
    return np.asarray(rows, dtype=float)


def autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def pool_acf(Q: np.ndarray) -> np.ndarray:
    """(N,48) -> (N, MAX_LAG) 逐样本自相关。"""
    return np.asarray([[autocorr(q, k) for k in range(1, MAX_LAG + 1)] for q in Q])


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """手写两样本 KS 检验 (渐近 p 值, Smirnov 公式), 不依赖 scipy。"""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    cdf1 = np.searchsorted(a, allv, side="right") / n1
    cdf2 = np.searchsorted(b, allv, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    # Kolmogorov 分布尾部级数
    p = 2.0 * sum((-1) ** (j - 1) * np.exp(-2.0 * (j * lam) ** 2) for j in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def summarize(Q: np.ndarray, floor: float | None) -> dict:
    acf = pool_acf(Q)
    peak_mean = Q.max(axis=1) / np.maximum(Q.mean(axis=1), 1e-12)
    wet_05 = (Q > 0.5).mean(axis=1)
    peak_time = Q.argmax(axis=1).astype(float)
    out = dict(
        n=Q.shape[0],
        acf_mean=np.nanmean(acf, axis=0), acf_std=np.nanstd(acf, axis=0),
        peak_mean_ratio=(float(peak_mean.mean()), float(peak_mean.std())),
        wet_frac_gt0p5=(float(wet_05.mean()), float(wet_05.std())),
        peak_time_q=(np.percentile(peak_time, [25, 50, 75]).tolist()),
        peak_time=peak_time, values=Q.ravel(),
    )
    if floor is not None:
        at_floor = (Q <= floor + 1e-6).mean(axis=1)
        out["frac_at_floor"] = (float(at_floor.mean()), float(at_floor.std()))
    return out


def fmt(ms: tuple[float, float]) -> str:
    return f"{ms[0]:.3f} ± {ms[1]:.3f}"


def main():
    ap = argparse.ArgumentParser(description="E5: 训练池 vs 真实事件统计对比")
    ap.add_argument("--train-csv", required=True, help="数据源 run 的 lhs_params.csv")
    ap.add_argument("--events-csv", required=True, help="E5 外部测试集 lhs_params.csv")
    ap.add_argument("--events-raw-csv", default=None, help="events_raw.csv (mm/h, 可选)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    train = load_qtop_rows(Path(args.train_csv), only_split="train")
    events = load_qtop_rows(Path(args.events_csv), only_split=None)
    # E5 参数表 = 事件 x 3 土壤, hydrograph 逐行重复; 统计只按去重后的事件计
    events = np.unique(events, axis=0)
    print(f"训练池 hydrograph: {train.shape}, 真实事件 (重标定后, 去重): {events.shape}")

    s_tr = summarize(train, floor=0.1)
    s_ev = summarize(events, floor=0.1)

    ks_int = ks_2samp(s_tr["values"], s_ev["values"])
    ks_pt = ks_2samp(s_tr["peak_time"], s_ev["peak_time"])

    raw_dry = None
    if args.events_raw_csv:
        raw = load_raw_events(Path(args.events_raw_csv))
        raw_dry = float((raw <= 1e-9).mean())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV ----
    rows = [["metric", "train_pool", "real_events"]]
    for k in range(MAX_LAG):
        rows.append([f"acf_lag{k+1}",
                     f"{s_tr['acf_mean'][k]:.3f} ± {s_tr['acf_std'][k]:.3f}",
                     f"{s_ev['acf_mean'][k]:.3f} ± {s_ev['acf_std'][k]:.3f}"])
    rows += [
        ["peak_to_mean_ratio", fmt(s_tr["peak_mean_ratio"]), fmt(s_ev["peak_mean_ratio"])],
        ["wet_frac_q_gt_0.5cmh", fmt(s_tr["wet_frac_gt0p5"]), fmt(s_ev["wet_frac_gt0p5"])],
        ["frac_at_envelope_floor(<=0.1)", fmt(s_tr["frac_at_floor"]), fmt(s_ev["frac_at_floor"])],
        ["peak_time_quartiles_h",
         "/".join(f"{v:.0f}" for v in s_tr["peak_time_q"]),
         "/".join(f"{v:.0f}" for v in s_ev["peak_time_q"])],
        ["n_hydrographs", str(s_tr["n"]), str(s_ev["n"])],
        ["KS_intensity_D(p)", f"{ks_int[0]:.3f} (p={ks_int[1]:.2e})", ""],
        ["KS_peak_time_D(p)", f"{ks_pt[0]:.3f} (p={ks_pt[1]:.2e})", ""],
    ]
    if raw_dry is not None:
        rows.append(["real_events_dry_frac_before_rescale", "", f"{raw_dry:.3f}"])
    with open(out_dir / "stats_compare_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # ---- Markdown ----
    md = ["# E5 统计对比: 训练池六控制点 hydrograph vs 真实降雨事件", "",
          f"- 训练池: {args.train_csv} (split=train, n={s_tr['n']})",
          f"- 真实事件: {args.events_csv} (n={s_ev['n']}, 已重标定到 [0.1,5.0] cm/h)", "",
          "| 指标 | 训练池 | 真实事件 |", "|---|---|---|"]
    for r in rows[1:]:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    md += ["", "## 诚实差异说明 (供回复信引用)",
           "- 六控制点线性插值使训练池 hydrograph 在短滞后上自相关偏强 (更平滑),"
           " 真实降雨间歇性更强、峰均比更高;",
           "- 训练包络下限 0.1 cm/h 意味着训练分布不含真正的干期;"
           " 真实事件重标定后干燥小时全部贴在 0.1 cm/h 下限"
           " (frac_at_envelope_floor 一行), 这是六点参数化的结构性局限, 如实报告;",
           "- 峰现时间: 训练池由 LHS 控制点决定、近似均匀,"
           " 真实事件由天气过程决定, KS 检验量化两者差异。"]
    (out_dir / "stats_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- 可选画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lags = np.arange(1, MAX_LAG + 1)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
        axs[0].errorbar(lags, s_tr["acf_mean"], yerr=s_tr["acf_std"], label="train pool")
        axs[0].errorbar(lags, s_ev["acf_mean"], yerr=s_ev["acf_std"], label="real events")
        axs[0].set_xlabel("lag (h)"); axs[0].set_ylabel("autocorr"); axs[0].legend()
        axs[1].hist([s_tr["peak_time"], s_ev["peak_time"]], bins=12, density=True,
                    label=["train pool", "real events"])
        axs[1].set_xlabel("peak time (h)"); axs[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "stats_compare_acf.png", dpi=200)
        print(f"[OK] 图: {out_dir}/stats_compare_acf.png")
    except Exception as e:  # matplotlib 不可用 -> 只出 CSV/MD
        print(f"[INFO] 跳过画图 ({e}); CSV/MD 已生成")

    print(f"[OK] {out_dir}/stats_compare_table.csv, stats_compare.md")


if __name__ == "__main__":
    main()
