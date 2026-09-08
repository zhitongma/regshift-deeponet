#!/usr/bin/env python3
"""E7 步骤 3 — HYDRUS 验证对比 (⚠️ 集群执行; 只需 numpy + 标准库).

前置: 02_run_hydrus.py 已在验证 run 目录上跑完 150 个方案
      (raw npz 在 <val-run-dir>/data/raw/sample_XXXX.npz)。

对比内容 (README 验收标准的数据来源):
  1. 风险概率偏差: |P_surrogate(t_b<36h) - P_HYDRUS(t_b<36h)| (验证子集内, 含分层小计);
  2. Spearman 排序相关 (手写实现, 平均秩处理并列; 不依赖 scipy):
     - t_b 的秩相关 (inf 按并列最大值处理)
     - 底部峰值浓度的秩相关
     - 综合排序分 (可行性 + 总入渗量) 的秩相关
  3. top-50 召回率: 在验证池内按 HYDRUS t_b 重排 (可行 → 总入渗量降序) 得
     HYDRUS-top-50, recall = |surrogate-top50 ∩ HYDRUS-top50| / 50
     (注意: 严格召回需对全部 10^4 方案跑 HYDRUS, 不可行; 此为验证池内近似, README 有说明);
  4. MBE 过滤 (R1-7 自诊断准则, 与 E10 交叉引用): 剔除自诊断 MBE > 10% 的方案后
     重算上述指标; 并统计过滤对"高误差方案"(t_b 误差 > 3h 或 36h 可行性误判) 的
     捕获率/查准率。

t_b 与峰值浓度的定义与 mc_screening.py / src/evaluation/metrics.py:180-201 一致:
t_b = min{t : c(L,t) >= 0.05*c_top}; c(L,·) 取 raw npz 的 c[-1, :]。

用法:
    python validate_mc.py \
        --screen-dir $REPO/experiments/qtop_func/runs_revision/e7_mc_screening_v1 \
        --val-run-dir $REPO/experiments/qtop_func/runs_revision/e7_mc_validation_v1
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="E7 HYDRUS 验证对比 (集群)")
    ap.add_argument("--screen-dir", required=True,
                    help="mc_screening.py 的 --out-dir (含 results/validation_params.csv)")
    ap.add_argument("--val-run-dir", required=True,
                    help="02_run_hydrus 的验证 run 目录 (含 data/raw/sample_XXXX.npz)")
    ap.add_argument("--out-dir", default=None,
                    help="默认 <screen-dir>/results/validation")
    ap.add_argument("--tb-threshold-hours", type=float, default=36.0)
    ap.add_argument("--bt-frac", type=float, default=0.05)
    ap.add_argument("--mbe-filter-pct", type=float, default=10.0)
    ap.add_argument("--bad-tb-err-hours", type=float, default=3.0,
                    help="定义'高误差方案'的 t_b 绝对误差阈值")
    ap.add_argument("--n-top", type=int, default=50)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# 手写统计工具 (无 scipy)
# ---------------------------------------------------------------------------
def average_ranks(x):
    """平均秩 (并列取平均), 支持 inf。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rho = Pearson(ranks_a, ranks_b), 并列用平均秩。"""
    ra, rb = average_ranks(a), average_ranks(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def ranking_score(tb, q_total, thr):
    """综合排序分: 可行 (t_b>thr) 的方案按总入渗量取正分, 不可行取负分垫底。
    与 mc_screening.py 的 ranked 序完全同序 (分高 = 排名靠前)。"""
    tb = np.asarray(tb, dtype=float)
    q = np.asarray(q_total, dtype=float)
    return np.where(tb > thr, q, q - (q.max() - q.min() + 1.0))


def compute_metrics(rows, thr, n_top):
    """rows: dict of arrays (已对齐的验证子集)。返回指标 dict。"""
    tb_s, tb_h = rows["tb_s"], rows["tb_h"]
    n = len(tb_s)
    if n == 0:
        return {"n": 0}
    feas_s, feas_h = tb_s > thr, tb_h > thr

    score_s = ranking_score(tb_s, rows["q_total"], thr)
    score_h = ranking_score(tb_h, rows["q_total"], thr)

    # 验证池内 top-50 召回
    in_top_s = rows["stratum"] == "head"  # surrogate top-50 即 head 层
    order_h = np.argsort(-score_h, kind="mergesort")
    k = min(n_top, int(feas_h.sum()))
    top_h = set(rows["scenario_id"][order_h[:k]].tolist()) if k > 0 else set()
    top_s = set(rows["scenario_id"][in_top_s].tolist())
    recall = (len(top_s & top_h) / len(top_s)) if top_s else float("nan")

    tb_err = np.where(np.isinf(tb_s) & np.isinf(tb_h), 0.0, np.abs(tb_s - tb_h))
    tb_err = np.where(np.isinf(tb_err), 48.0, tb_err)  # 一侧 inf 记满量程误差

    return {
        "n": int(n),
        "risk_prob_surrogate": float(np.mean(tb_s < thr)),
        "risk_prob_hydrus": float(np.mean(tb_h < thr)),
        "risk_prob_abs_dev": float(abs(np.mean(tb_s < thr) - np.mean(tb_h < thr))),
        "spearman_tb": spearman(tb_s, tb_h),
        "spearman_peak_c": spearman(rows["peak_s"], rows["peak_h"]),
        "spearman_ranking_score": spearman(score_s, score_h),
        "top50_recall_in_pool": recall,
        "n_hydrus_feasible_in_pool": int(feas_h.sum()),
        "feasibility_confusion": {
            "tp": int(np.sum(feas_s & feas_h)), "fp": int(np.sum(feas_s & ~feas_h)),
            "fn": int(np.sum(~feas_s & feas_h)), "tn": int(np.sum(~feas_s & ~feas_h)),
        },
        "tb_mae_hours": float(np.mean(tb_err)),
        "tb_err_p90_hours": float(np.percentile(tb_err, 90)),
        "peak_c_mae": float(np.mean(np.abs(rows["peak_s"] - rows["peak_h"]))),
    }


def main():
    args = parse_args()
    screen_dir = Path(args.screen_dir).resolve()
    val_run = Path(args.val_run_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else screen_dir / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 读取筛选阶段的验证参数表 ----------------
    csv_path = screen_dir / "results" / "validation_params.csv"
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    thr = args.tb_threshold_hours
    recs, missing = [], []
    for e in entries:
        sid = int(e["sample_id"])
        npz_path = val_run / "data" / "raw" / f"sample_{sid:04d}.npz"
        if not npz_path.exists():
            missing.append(sid)
            continue
        data = np.load(npz_path, allow_pickle=True)
        c = data["c"]; t = data["t"]
        c_top = float(e["c_top"])
        c_bot = c[-1, :]
        idx = np.where(c_bot >= args.bt_frac * c_top)[0]
        tb_h = float(t[idx[0]]) if len(idx) else float("inf")
        recs.append({
            "sample_id": sid,
            "scenario_id": int(e["scenario_id"]),
            "stratum": e["stratum"],
            "q_total": float(e["q_top_total"]),
            "mbe": float(e["surrogate_mbe_pct"]),
            "tb_s": float(e["surrogate_tb_hours"]),
            "tb_h": tb_h,
            "peak_s": float(e["surrogate_peak_c"]),
            "peak_h": float(c_bot.max()),
        })

    if not recs:
        raise SystemExit(f"未找到任何验证 raw npz ({val_run}/data/raw), "
                         f"请先运行 02_run_hydrus.py")

    rows = {k: np.asarray([r[k] for r in recs]) for k in recs[0]}
    for k in ["q_total", "mbe", "tb_s", "tb_h", "peak_s", "peak_h"]:
        rows[k] = rows[k].astype(float)

    # ---------------- 全池指标 + 分层小计 ----------------
    metrics_all = compute_metrics(rows, thr, args.n_top)
    strata_metrics = {}
    for s in sorted(set(rows["stratum"].tolist())):
        m = rows["stratum"] == s
        sub = {k: v[m] for k, v in rows.items()}
        strata_metrics[s] = {
            "n": int(m.sum()),
            "risk_prob_abs_dev": float(abs(np.mean(sub["tb_s"] < thr) - np.mean(sub["tb_h"] < thr))),
            "spearman_tb": spearman(sub["tb_s"], sub["tb_h"]),
            "tb_mae_hours": float(np.mean(np.where(
                np.isinf(sub["tb_s"]) & np.isinf(sub["tb_h"]), 0.0,
                np.minimum(np.abs(sub["tb_s"] - sub["tb_h"]), 48.0)))),
        }

    # ---------------- MBE 过滤 (R1-7 自诊断, 联动 E10) ----------------
    keep = rows["mbe"] <= args.mbe_filter_pct
    rows_keep = {k: v[keep] for k, v in rows.items()}
    metrics_filtered = compute_metrics(rows_keep, thr, args.n_top)

    tb_err = np.where(np.isinf(rows["tb_s"]) & np.isinf(rows["tb_h"]), 0.0,
                      np.abs(rows["tb_s"] - rows["tb_h"]))
    tb_err = np.where(np.isinf(tb_err), 48.0, tb_err)
    misclass = (rows["tb_s"] > thr) != (rows["tb_h"] > thr)
    bad = (tb_err > args.bad_tb_err_hours) | misclass
    removed = ~keep
    filter_stats = {
        "mbe_threshold_pct": args.mbe_filter_pct,
        "n_removed": int(removed.sum()),
        "n_bad_total": int(bad.sum()),
        "n_bad_removed": int((bad & removed).sum()),
        "bad_capture_rate": float((bad & removed).sum() / bad.sum()) if bad.any() else None,
        "filter_precision": float((bad & removed).sum() / removed.sum()) if removed.any() else None,
        "bad_rate_before": float(bad.mean()),
        "bad_rate_after": float(bad[keep].mean()) if keep.any() else None,
    }

    # ---------------- 输出 ----------------
    def _clean(o):
        if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o

    results = _clean({
        "screen_dir": str(screen_dir), "val_run_dir": str(val_run),
        "tb_threshold_hours": thr, "bt_frac": args.bt_frac,
        "n_planned": len(entries), "n_validated": len(recs),
        "missing_sample_ids": missing,
        "metrics_all": metrics_all,
        "metrics_by_stratum": strata_metrics,
        "metrics_after_mbe_filter": metrics_filtered,
        "mbe_filter": filter_stats,
        "acceptance": {
            "spearman_ranking_gt_0.9": bool((metrics_all.get("spearman_ranking_score") or 0) > 0.9),
            "top50_recall_gt_0.8": bool((metrics_all.get("top50_recall_in_pool") or 0) > 0.8),
        },
    })
    with open(out_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    table_path = out_dir / "validation_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "scenario_id", "stratum", "q_top_total",
                    "surrogate_mbe_pct", "tb_surrogate_h", "tb_hydrus_h",
                    "peak_c_surrogate", "peak_c_hydrus", "tb_abs_err_h",
                    "feasible_surrogate", "feasible_hydrus", "misclassified"])
        for i, r in enumerate(recs):
            w.writerow([r["sample_id"], r["scenario_id"], r["stratum"],
                        f"{r['q_total']:.6g}", f"{r['mbe']:.6g}",
                        r["tb_s"], r["tb_h"],
                        f"{r['peak_s']:.6g}", f"{r['peak_h']:.6g}",
                        f"{tb_err[i]:.6g}",
                        int(r["tb_s"] > thr), int(r["tb_h"] > thr), int(misclass[i])])

    print(json.dumps({"metrics_all": results["metrics_all"],
                      "mbe_filter": results["mbe_filter"],
                      "acceptance": results["acceptance"]},
                     indent=2, ensure_ascii=False))
    print(f"[done] {out_dir / 'validation_results.json'}\n       {table_path}")


if __name__ == "__main__":
    main()
