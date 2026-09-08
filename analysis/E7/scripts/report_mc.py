#!/usr/bin/env python3
"""E7 步骤 4 — 汇总 Markdown 结果卡 (本地可跑; 只需 numpy + 标准库).

输入: mc_screening_summary.json + validation_results.json (+ validation_table.csv)
      —— 从集群 rsync 回来后运行。
输出: RESULTS_E7.md (进正文"MC 情景筛选演示"新小节的数据来源)
      + 可选散点图 fig_tb_scatter.png (matplotlib 可用时; 否则跳过, CSV 已含全数据)。

用法:
    python report_mc.py --screen-dir <e7_mc_screening_v1 本地副本> \
        [--out RESULTS_E7.md]
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="E7 结果卡汇总 (本地)")
    ap.add_argument("--screen-dir", required=True,
                    help="含 results/mc_screening_summary.json 与 "
                         "results/validation/validation_results.json 的目录")
    ap.add_argument("--out", default=None, help="默认 <screen-dir>/RESULTS_E7.md")
    return ap.parse_args()


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    args = parse_args()
    screen_dir = Path(args.screen_dir).resolve()
    with open(screen_dir / "results" / "mc_screening_summary.json", encoding="utf-8") as f:
        S = json.load(f)
    val_json = screen_dir / "results" / "validation" / "validation_results.json"
    V = None
    if val_json.exists():
        with open(val_json, encoding="utf-8") as f:
            V = json.load(f)

    lines = []
    a = lines.append
    a("# E7 结果卡 — Monte Carlo / MAR 情景筛选演示 (RegShift 代理)")
    a("")
    a("> 回应 R1-8 / R4-8 (代理模型的决策应用价值 / UQ 工作流演示); "
      "MBE 自诊断过滤联动 R1-7 与 E10。")
    a("")
    a("## 1. 筛选设置")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    a(f"| 候选调度数 | {S['n_scenarios']} (6 控制点 LHS, 同训练生成逻辑) |")
    a(f"| 土壤参数 | Carsel & Parrish (1988) loam 均值 "
      f"(θr={S['loam_params']['theta_r']}, θs={S['loam_params']['theta_s']}, "
      f"α={S['loam_params']['alpha']} cm⁻¹, n={S['loam_params']['n_vg']}, "
      f"Ks={S['loam_params']['K_s']} cm/h, D_L={S['loam_params']['D_L']} cm) |")
    a(f"| 参数抖动 | {'开 (±' + str(S['jitter_frac']) + ' LHS)' if S['param_jitter'] else '关 (固定均值)'} |")
    a(f"| 代理模型 | RegShift m2 ({S['ckpt']}) |")
    a(f"| 有效界限 | scale={S['effective_transform_bound_scale']}, "
      f"shift={S['effective_transform_bound_shift']} (E0 口径) |")
    a(f"| 突破定义 | t_b = min{{t: c(L,t) ≥ {S['bt_frac']}·c_top}}, 风险阈值 {S['tb_threshold_hours']} h |")
    a("")
    a("## 2. 代理筛选结果 (10^4 方案)")
    a("")
    a("| 指标 | 值 |")
    a("|---|---|")
    a(f"| P(t_b < {fmt(S['tb_threshold_hours'],0)} h) | {fmt(S['risk_prob_tb_lt_threshold'])} |")
    a(f"| 可行方案数 (t_b > 阈值) | {S['n_feasible']} |")
    a(f"| 未突破方案数 (48h 内) | {S['tb_hours']['n_no_breakthrough']} |")
    a(f"| t_b 中位数 (有限值) | {fmt(S['tb_hours']['median_finite'])} h |")
    a(f"| top-50 最小总入渗量 | {fmt(S['q_total_cm']['top50_min'])} cm |")
    a(f"| 自诊断 MBE 中位数 / P90 | {fmt(S['mbe_self_pct']['median'])} % / "
      f"{fmt(S['mbe_self_pct']['p90'])} % |")
    a(f"| MBE > 10% 方案占比 | {fmt(S['mbe_self_pct']['frac_gt_10pct'])} |")
    a(f"| 验证子集 | {S['n_validate']} 个, 分层 {S['strata_counts']} |")
    a("")

    if V is None:
        a("## 3. HYDRUS 验证")
        a("")
        a("**尚未运行** —— 待集群完成 02_run_hydrus + validate_mc.py 后重跑本脚本。")
    else:
        M = V["metrics_all"]; F = V["metrics_after_mbe_filter"]; FL = V["mbe_filter"]
        a(f"## 3. HYDRUS 验证 ({V['n_validated']}/{V['n_planned']} 方案完成"
          + (f", 缺失 {len(V['missing_sample_ids'])}" if V["missing_sample_ids"] else "") + ")")
        a("")
        a("| 指标 | 全部验证方案 | 剔除 MBE>" + fmt(FL["mbe_threshold_pct"], 0) + "% 后 |")
        a("|---|---|---|")
        a(f"| Spearman ρ (综合排序分) | {fmt(M.get('spearman_ranking_score'))} | "
          f"{fmt(F.get('spearman_ranking_score'))} |")
        a(f"| Spearman ρ (t_b) | {fmt(M.get('spearman_tb'))} | {fmt(F.get('spearman_tb'))} |")
        a(f"| Spearman ρ (底部峰值浓度) | {fmt(M.get('spearman_peak_c'))} | "
          f"{fmt(F.get('spearman_peak_c'))} |")
        a(f"| top-50 召回率 (验证池内) | {fmt(M.get('top50_recall_in_pool'))} | "
          f"{fmt(F.get('top50_recall_in_pool'))} |")
        a(f"| 风险概率偏差 \\|ΔP\\| | {fmt(M.get('risk_prob_abs_dev'))} | "
          f"{fmt(F.get('risk_prob_abs_dev'))} |")
        a(f"| t_b MAE | {fmt(M.get('tb_mae_hours'))} h | {fmt(F.get('tb_mae_hours'))} h |")
        a(f"| 峰值浓度 MAE | {fmt(M.get('peak_c_mae'))} | {fmt(F.get('peak_c_mae'))} |")
        cm = M["feasibility_confusion"]
        a(f"| 36h 可行性混淆 (TP/FP/FN/TN) | {cm['tp']}/{cm['fp']}/{cm['fn']}/{cm['tn']} | — |")
        a("")
        a("### 分层小计")
        a("")
        a("| 层 | n | Spearman ρ (t_b) | t_b MAE (h) | \\|ΔP\\| |")
        a("|---|---|---|---|---|")
        for s, sm in V["metrics_by_stratum"].items():
            a(f"| {s} | {sm['n']} | {fmt(sm.get('spearman_tb'))} | "
              f"{fmt(sm.get('tb_mae_hours'))} | {fmt(sm.get('risk_prob_abs_dev'))} |")
        a("")
        a("### MBE 自诊断过滤 (R1-7 / E10)")
        a("")
        a("| 指标 | 值 |")
        a("|---|---|")
        a(f"| 剔除方案数 (MBE>{fmt(FL['mbe_threshold_pct'],0)}%) | {FL['n_removed']} |")
        a(f"| 高误差方案总数 (t_b 误差>3h 或误判) | {FL['n_bad_total']} |")
        a(f"| 高误差捕获率 (被剔除的占比) | {fmt(FL.get('bad_capture_rate'))} |")
        a(f"| 过滤查准率 | {fmt(FL.get('filter_precision'))} |")
        a(f"| 高误差占比 过滤前 → 后 | {fmt(FL['bad_rate_before'])} → {fmt(FL.get('bad_rate_after'))} |")
        a("")
        a("## 4. 验收判定")
        a("")
        acc = V["acceptance"]
        strong = acc["spearman_ranking_gt_0.9"] and acc["top50_recall_gt_0.8"]
        a(f"- 排序 Spearman > 0.9: **{fmt(acc['spearman_ranking_gt_0.9'])}**")
        a(f"- top-50 召回 > 0.8: **{fmt(acc['top50_recall_gt_0.8'])}**")
        a(f"- 综合: **{'强结果 ✔' if strong else '未达强结果标准 (见 README 处置预案)'}**")
        cap = FL.get("bad_capture_rate")
        a(f"- MBE 过滤支撑自诊断准则 (捕获率≥0.5): "
          f"**{fmt(cap is not None and cap >= 0.5)}** (捕获率 {fmt(cap)})")
    a("")
    a("---")
    a(f"*来源: {screen_dir}/results/(mc_screening_summary.json, "
      "validation/validation_results.json); 由 report_mc.py 生成。*")
    a("")

    out = Path(args.out) if args.out else screen_dir / "RESULTS_E7.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out}")

    # ---------------- 可选散点图 (matplotlib 缺失时降级跳过) ----------------
    table = screen_dir / "results" / "validation" / "validation_table.csv"
    if V is not None and table.exists():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[info] 无 matplotlib, 跳过散点图 (validation_table.csv 已含全数据)")
            return
        tb_s, tb_h, strat = [], [], []
        with open(table, newline="") as f:
            for row in csv.DictReader(f):
                s, h = float(row["tb_surrogate_h"]), float(row["tb_hydrus_h"])
                tb_s.append(min(s, 50.0)); tb_h.append(min(h, 50.0))  # inf → 50 处画界外
                strat.append(row["stratum"])
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        colors = {"head": "tab:blue", "mid": "tab:green", "tail": "tab:orange",
                  "high_mbe": "tab:red", "fill": "tab:gray"}
        for st in sorted(set(strat)):
            xs = [a_ for a_, s_ in zip(tb_h, strat) if s_ == st]
            ys = [b_ for b_, s_ in zip(tb_s, strat) if s_ == st]
            ax.scatter(xs, ys, s=14, alpha=0.7, label=st, color=colors.get(st, "k"))
        ax.plot([0, 50], [0, 50], "k--", lw=0.8)
        ax.axvline(36, color="gray", lw=0.6); ax.axhline(36, color="gray", lw=0.6)
        ax.set_xlabel("HYDRUS $t_b$ (h)"); ax.set_ylabel("RegShift $t_b$ (h)")
        ax.legend(fontsize=7); fig.tight_layout()
        fig_path = screen_dir / "results" / "validation" / "fig_tb_scatter.png"
        fig.savefig(fig_path, dpi=200)
        print(f"[done] {fig_path}")


if __name__ == "__main__":
    main()
