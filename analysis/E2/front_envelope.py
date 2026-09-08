#!/usr/bin/env python3
"""E2-A: 浓度锋位置的跨样本变异统计 —— 为 σs=0.3 / σδ=0.2 提供物理锚点。

思路
----
RegShift 的仿射变换作用在归一化 (z,t) ∈ [0,1]^2 坐标上:
  scale ∈ [1-σs, 1+σs]（锋面"快慢"的伸缩）, shift ∈ [-σδ, σδ]（锋面"位置"的平移）。
若训练数据中锋面深度轨迹的跨样本平移量 ~ 0.2、传播速度比值的跨样本伸缩量 ~ 1±0.3,
则默认界限 (0.3, 0.2) 有明确的物理依据。

实现
----
读 <数据源run>/data/processed/train.npz 的 c_raw (N, 101, 49)（或 (N, 101*49) 自动
reshape）与 z/t; 对每个样本、每个时刻, 沿深度提取 c = --threshold (默认 0.01)
等值线的锋面深度（相邻节点线性插值）; 深度归一化到 [0,1] 后统计:
  - 平移锚点: 各时刻锋面深度的跨样本分位差 (p90-p10)/2 与 p90|d_i - median|,
    与 σδ=0.2 对比;
  - 伸缩锚点: 每样本平均传播速度 v_i（锋面深度对时间的最小二乘斜率）与中位速度
    之比 r_i, 统计 r 的 p10/p90 及 p90|r_i - 1|, 与 σs=0.3 对比。

⚠️ 执行环境
----
主仓库位于 iCloud 同步盘, processed npz 多为 dataless 占位文件, 本机直接读取会
永久挂起。**本脚本请在集群执行, 或先物化数据**
（`brctl download <文件>` / Finder 下载后再跑）。脚本启动前会用 stat 探测
占位文件并拒绝读取。依赖: 仅 numpy（matplotlib 可选, 缺失则只出 CSV）。

用法
----
  python3 front_envelope.py \
      --run-dir  <REPO>/experiments/qtop_func/runs/iid_medium_v1 \
      --split train --threshold 0.01 \
      --out-dir  <本包>/results_A
"""

from pathlib import Path
import argparse
import csv
import json
import os
import sys

import numpy as np

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))
DEFAULT_RUN = os.path.join(DEFAULT_REPO, "experiments/qtop_func/runs/iid_medium_v1")


def guard_icloud_placeholder(path):
    """dataless 占位文件: st_size>0 但磁盘块几乎为 0。读它会挂起, 先拒绝。"""
    st = os.stat(path)
    on_disk = st.st_blocks * 512
    if st.st_size > 1024 * 1024 and on_disk < st.st_size * 0.05:
        sys.exit(f"[ABORT] {path} 疑似 iCloud dataless 占位文件 "
                 f"(size={st.st_size}, on_disk={on_disk})。\n"
                 f"请在集群执行本脚本, 或先物化数据: brctl download '{path}'")


def extract_front_depth(c_zt, zn, threshold):
    """单样本单时刻: 沿深度找 c 从 >=threshold 跌破 threshold 的最深交点 (线性插值)。

    c_zt: (n_z,) 按 zn 升序（0=地表）排列的浓度剖面。
    返回归一化锋面深度 ∈ [0,1]; 地表即低于阈值返回 0.0; 整柱高于阈值返回 1.0。
    """
    above = c_zt >= threshold
    if not above[0]:
        return 0.0
    if above.all():
        return 1.0
    j = int(np.argmax(~above))  # 第一个 < threshold 的节点
    c1, c2 = c_zt[j - 1], c_zt[j]
    z1, z2 = zn[j - 1], zn[j]
    if c1 == c2:
        return float(z1)
    frac = (c1 - threshold) / (c1 - c2)
    return float(z1 + frac * (z2 - z1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN,
                    help="数据源 run 根目录（含 data/processed/train.npz）")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="锋面等值线浓度阈值（绝对值, 默认 0.01）")
    ap.add_argument("--relative", action="store_true",
                    help="改用相对阈值: threshold × 每样本最大浓度")
    ap.add_argument("--min-front-frac", type=float, default=0.5,
                    help="仅统计『至少该比例样本锋面已入渗 (>0)』的时刻")
    ap.add_argument("--out-dir", default=os.path.join(PKG_DIR, "results_A"))
    ap.add_argument("--bound-scale", type=float, default=0.3, help="对比用 σs")
    ap.add_argument("--bound-shift", type=float, default=0.2, help="对比用 σδ")
    args = ap.parse_args()

    npz_path = os.path.join(args.run_dir, "data", "processed", f"{args.split}.npz")
    if not os.path.exists(npz_path):
        sys.exit(f"[ABORT] 未找到 {npz_path}")
    guard_icloud_placeholder(npz_path)

    d = np.load(npz_path)
    z = np.asarray(d["z"], dtype=float).ravel()
    t = np.asarray(d["t"], dtype=float).ravel()
    n_z, n_t = z.size, t.size
    c = np.asarray(d["c_raw"], dtype=float)
    if c.ndim == 2:  # (N, n_z*n_t) 存储
        c = c.reshape(c.shape[0], n_z, n_t)
    assert c.shape[1:] == (n_z, n_t), f"c_raw 形状异常: {c.shape}, 期望 (N,{n_z},{n_t})"
    N = c.shape[0]

    # 深度归一化并保证升序（0=地表）。z 可能以负值/降序存储, 用 |z| 排序兜底。
    depth = np.abs(z - z[np.argmin(np.abs(z))]) if (z < 0).any() else z - z.min()
    order = np.argsort(depth)
    zn = depth[order] / depth[order][-1]
    c = c[:, order, :]

    # 锋面深度轨迹 d[i, j] ∈ [0,1]
    front = np.zeros((N, n_t))
    for i in range(N):
        thr = args.threshold * c[i].max() if args.relative else args.threshold
        for j in range(n_t):
            front[i, j] = extract_front_depth(c[i, :, j], zn, thr)

    # ---- 平移锚点: 逐时刻跨样本分布 ----
    per_time = []
    for j in range(n_t):
        fj = front[:, j]
        frac_active = float((fj > 0).mean())
        med = float(np.median(fj))
        p10, p90 = np.percentile(fj, [10, 90])
        dev_p90 = float(np.percentile(np.abs(fj - med), 90))
        per_time.append({
            "t": float(t[j]), "frac_active": frac_active,
            "front_p10": float(p10), "front_median": med, "front_p90": float(p90),
            "half_spread_p90p10": float((p90 - p10) / 2.0),
            "abs_dev_from_median_p90": dev_p90,
        })
    active = [r for r in per_time if r["frac_active"] >= args.min_front_frac]
    if not active:
        sys.exit("[ABORT] 没有满足 min-front-frac 的时刻, 请调低 --min-front-frac 或阈值")
    shift_anchor_median = float(np.median([r["half_spread_p90p10"] for r in active]))
    shift_anchor_max = float(max(r["half_spread_p90p10"] for r in active))
    dev_anchor_median = float(np.median([r["abs_dev_from_median_p90"] for r in active]))
    dev_anchor_max = float(max(r["abs_dev_from_median_p90"] for r in active))

    # ---- 伸缩锚点: 每样本传播速度 / 中位速度 ----
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-12)
    v = np.zeros(N)
    for i in range(N):
        mask = front[i] > 0
        if mask.sum() >= 3:
            tt, ff = t_norm[mask], front[i][mask]
            v[i] = np.polyfit(tt, ff, 1)[0]  # 最小二乘斜率
        else:
            v[i] = np.nan
    v_ok = v[np.isfinite(v) & (v > 0)]
    v_med = float(np.median(v_ok))
    r = v_ok / v_med
    r_p10, r_p90 = np.percentile(r, [10, 90])
    scale_anchor_p90 = float(np.percentile(np.abs(r - 1.0), 90))
    scale_anchor_max = float(np.abs(r - 1.0).max())

    summary = {
        "npz": npz_path, "split": args.split, "n_samples": int(N),
        "threshold": args.threshold, "relative": bool(args.relative),
        "n_active_times": len(active),
        "shift_anchor": {
            "half_spread_p90p10_median_over_t": shift_anchor_median,
            "half_spread_p90p10_max_over_t": shift_anchor_max,
            "abs_dev_from_median_p90_median_over_t": dev_anchor_median,
            "abs_dev_from_median_p90_max_over_t": dev_anchor_max,
            "compare_bound_shift": args.bound_shift,
        },
        "scale_anchor": {
            "n_velocity_samples": int(v_ok.size),
            "velocity_ratio_p10": float(r_p10),
            "velocity_ratio_p90": float(r_p90),
            "abs_ratio_dev_p90": scale_anchor_p90,
            "abs_ratio_dev_max": scale_anchor_max,
            "compare_bound_scale": args.bound_scale,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"front_envelope_per_time_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_time[0].keys()))
        w.writeheader()
        w.writerows(per_time)
    json_path = os.path.join(args.out_dir, f"front_envelope_summary_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[结论提示] 平移锚点(半分位差, 中位/最大) = "
          f"{shift_anchor_median:.3f}/{shift_anchor_max:.3f} vs σδ={args.bound_shift}; "
          f"伸缩锚点(|v/v_med - 1| p90/max) = "
          f"{scale_anchor_p90:.3f}/{scale_anchor_max:.3f} vs σs={args.bound_scale}")
    print(f"[输出] {csv_path}\n[输出] {json_path}")

    try:  # 可选绘图, 无 matplotlib 时静默降级为仅 CSV/JSON
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        tv = [r_["t"] for r_ in per_time]
        ax.fill_between(tv, [r_["front_p10"] for r_ in per_time],
                        [r_["front_p90"] for r_ in per_time],
                        alpha=0.3, label="front depth p10–p90")
        ax.plot(tv, [r_["front_median"] for r_ in per_time], lw=1.5, label="median")
        ax.set_xlabel("time (h)")
        ax.set_ylabel("normalized front depth")
        ax.set_title(f"Concentration-front envelope ({args.split}, c={args.threshold})")
        ax.legend()
        png = os.path.join(args.out_dir, f"front_envelope_{args.split}.png")
        fig.tight_layout()
        fig.savefig(png, dpi=200)
        print(f"[输出] {png}")
    except ImportError:
        print("[提示] 未安装 matplotlib, 跳过绘图（CSV/JSON 已生成）")


if __name__ == "__main__":
    main()
