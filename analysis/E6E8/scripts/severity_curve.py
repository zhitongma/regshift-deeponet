#!/usr/bin/env python3
"""E6/E8 汇总: 把 A 部分（分位重切档）与 B 部分（越界外推档）的分档误差
合成"误差-严重度"曲线数据（论文新图数据源）。numpy-only；只读文本
CSV/JSON —— 本地可跑（前提: 输入 CSV 已生成/同步并物化）。

输入:
  --a-csv  eval_subsets.py 对 A_subsets.json 的输出（缺省可跳过）
  --b-csv  eval_subsets.py 对 B_subsets.json 的输出（缺省可跳过）
输出（--out-dir, 默认 <包>/outputs）:
  severity_curve_long.csv   统一严重度序号的长表（画图直接用）
  severity_slopes.csv       每 (scenario,family,model,axis) 的斜率 + 单调性(Spearman)
                            —— 按训练场景分开, 不跨 IID/OOD 训练检查点平均;
                            同 scenario 内跨 run_key（A 档与 B 档、多种子）
                            合并同 rank 取均值
  convergence_report.csv    B 部分各档 HYDRUS 不收敛/QC 剔除统计（"适用外推半径"）
  severity_curve.png        可选（matplotlib 存在时）

严重度序号约定:
  q_top_peak:       p00_p90=0 p90_p95=1 p95_p98=2 p98_p100=3 peak_5.5=4 peak_6.5=5 peak_8.0=6
  q_top_peak_time:  p00_p90=0 ... p98_p100=3 peaktime_44h=4 45h=5 46h=6 47h=7
  joint_forcing_x_Ks: joint_Ks0.15=1, joint_Ks40=2（对照=各家族 IID p00_p90 档）
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RANKS = {
    "q_top_peak": {"p00_p90": 0, "p90_p95": 1, "p95_p98": 2, "p98_p100": 3,
                   "peak_5.5": 4, "peak_6.5": 5, "peak_8.0": 6},
    "q_top_peak_time": {"p00_p90": 0, "p90_p95": 1, "p95_p98": 2, "p98_p100": 3,
                        "peaktime_44h": 4, "peaktime_45h": 5,
                        "peaktime_46h": 6, "peaktime_47h": 7},
    "joint_forcing_x_Ks": {"joint_Ks0.15": 1, "joint_Ks40": 2},
}


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """手写 Spearman 秩相关（scipy 不可用）。"""
    def ranks(a):
        order = np.argsort(a)
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(1, len(a) + 1)
        # 并列取平均秩
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    pkg = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-csv", default=str(pkg / "outputs" / "A_binned_errors.csv"))
    ap.add_argument("--b-csv", default=str(pkg / "outputs" / "B_binned_errors.csv"))
    ap.add_argument("--b-subsets", default=str(pkg / "outputs" / "B_subsets.json"),
                    help="用于收敛统计的 B_subsets.json（可缺省）")
    ap.add_argument("--manifest", default=str(pkg / "models_manifest.json"))
    ap.add_argument("--repo", default=None,
                    help="主仓库根路径（集群上必须传 $REPO；缺省回退到 "
                         "manifest['repo']，那是生成 manifest 时的本机路径）")
    ap.add_argument("--out-dir", default=str(pkg / "outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    for part, path in (("A", Path(args.a_csv)), ("B", Path(args.b_csv))):
        if not path.exists():
            print(f"[WARN] 缺 {path}，跳过 {part} 部分")
            continue
        for r in read_csv_rows(path):
            axis = r["axis"]
            # B 部分 subsets 的 axis 即 manifest 里的 set axis
            rank = RANKS.get(axis, {}).get(r["bin"])
            if rank is None:
                print(f"[WARN] 未知 bin {r['bin']} (axis={axis})，跳过")
                continue
            long_rows.append({
                "part": part,
                "run_key": r["run_key"],
                "scenario": r["scenario"],
                "family": r["family"],
                "model": r["model"],
                "axis": axis,
                "severity_bin": r["bin"],
                "severity_rank": rank,
                "n": int(r["n"]),
                "rel_l2_h_mean": float(r["rel_l2_h_mean"]),
                "rel_l2_h_median": float(r["rel_l2_h_median"]),
                "rel_l2_c_mean": float(r["rel_l2_c_mean"]),
                "rel_l2_c_median": float(r["rel_l2_c_median"]),
                "bt_mae_hours": r.get("bt_mae_hours", ""),
            })

    if not long_rows:
        print("[ERROR] 无输入行")
        return 1

    long_rows.sort(key=lambda r: (r["axis"], r["family"], r["model"],
                                  r["run_key"], r["severity_rank"]))
    long_path = out_dir / "severity_curve_long.csv"
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)
    print(f"输出: {long_path} ({len(long_rows)} 行)")

    # 斜率 + 单调性: 按 (scenario, family, model, axis) 聚合。
    # scenario 进分组键 => IID 训练与 OOD 训练检查点（如 iid_baseline 与
    # ood_peak_baseline 的 m1）各出一条曲线, 不被平均掩盖训练场景间差异;
    # 同 scenario 内仍跨 run_key 合并同 rank 取均值（A 档 + B 档拼接、多种子平均）。
    slope_rows = []
    keys = sorted({(r["scenario"], r["family"], r["model"], r["axis"]) for r in long_rows})
    for scen, fam, model, axis in keys:
        sub = [r for r in long_rows
               if (r["scenario"], r["family"], r["model"], r["axis"]) == (scen, fam, model, axis)]
        by_rank: dict[int, list[float]] = {}
        for r in sub:
            by_rank.setdefault(r["severity_rank"], []).append(r["rel_l2_c_mean"])
        xs = np.array(sorted(by_rank), dtype=float)
        ys = np.array([np.mean(by_rank[int(x)]) for x in xs])
        if len(xs) < 2:
            continue
        slope = float(np.polyfit(xs, ys, 1)[0])
        rho = spearman(xs, ys)
        monotone = bool(np.all(np.diff(ys) >= -1e-12))
        slope_rows.append({
            "scenario": scen, "family": fam, "model": model, "axis": axis,
            "n_bins": len(xs),
            "slope_rel_l2_c_per_rank": slope,
            "spearman_rank_corr": rho,
            "strictly_nondecreasing": monotone,
            "rel_l2_c_by_rank": ";".join(f"{int(x)}:{y:.4f}" for x, y in zip(xs, ys)),
        })
    slope_path = out_dir / "severity_slopes.csv"
    with open(slope_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(slope_rows[0].keys()))
        w.writeheader()
        w.writerows(slope_rows)
    print(f"输出: {slope_path} ({len(slope_rows)} 行)")

    # 收敛统计（适用外推半径）: 从 B_subsets + set CSV + set run 的 hydrus manifest。
    # 路径解析顺序（集群安全）:
    #   repo 前缀 = --repo（推荐, run_B_cluster.sh do_curve 传 $REPO）
    #             > manifest["repo"]（本机口径, 集群上通常不存在 -> 触发下面的兜底）;
    #   lhs_params.csv 优先读 set run, 缺失时兜底读 B_subsets 里评估 run 内的副本
    #   （do_process 已 cp 进 eval run, 是集群本地路径）; 均缺失打 [WARN] 不再静默。
    conv_rows = []
    b_subsets_path = Path(args.b_subsets)
    if b_subsets_path.exists():
        with open(args.manifest) as f:
            manifest = json.load(f)
        repo = Path(args.repo) if args.repo else Path(manifest["repo"])
        if not repo.exists():
            print(f"[WARN] repo 路径不存在: {repo} —— 若在集群运行请传 "
                  f"--repo <集群仓库路径>（缺省用的是 manifest['repo'] 的本机口径）")
        with open(b_subsets_path) as f:
            bsub = json.load(f)
        seen_sets = set()
        for run in bsub["runs"].values():
            set_key = run["set_key"]
            if set_key in seen_sets:
                continue  # 收敛统计与模型无关, 每个集合统计一次
            seen_sets.add(set_key)
            eval_dir = Path(run["run_dir"])
            qc_path = eval_dir / "metadata" / "qc_summary.json"
            qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
            if not qc:
                print(f"[WARN] {set_key}: 缺 {qc_path} —— QC 剔除数将按 0 计")
            set_run = repo / manifest["runs_revision_root"] / set_key
            hb_path = set_run / "metadata" / "hydrus_batch_manifest.json"
            sim_failed_ids = set()
            if hb_path.exists():
                hb = json.loads(hb_path.read_text())
                sim_failed_ids = {int(k) for k in hb.get("failed_samples", {})}
            else:
                print(f"[WARN] {set_key}: 缺 {hb_path} —— 仿真失败数改用 "
                      f"qc_summary.missing_sim_sample_ids 兜底（raw 缺失样本口径一致）")
            sim_failed_ids |= {int(x) for x in qc.get("missing_sim_sample_ids", [])}
            qc_failed_ids = {int(x) for x in qc.get("failed_sample_ids", [])}

            groups_of = {}
            params_csv = set_run / "data" / "parameters" / "lhs_params.csv"
            if not params_csv.exists():
                fallback_csv = eval_dir / "data" / "parameters" / "lhs_params.csv"
                if fallback_csv.exists():
                    print(f"[WARN] {set_key}: 缺 {params_csv} —— 改用评估 run 内副本 "
                          f"{fallback_csv}")
                    params_csv = fallback_csv
                else:
                    print(f"[WARN] {set_key}: lhs_params.csv 两处均缺 "
                          f"({params_csv} 与 {fallback_csv}) —— 该集合的收敛报告行"
                          f"将缺失; 请检查 --repo 是否指向集群仓库")
            if params_csv.exists():
                with open(params_csv, newline="") as f:
                    for row in csv.DictReader(f):
                        groups_of[int(row["sample_id"])] = row["severity_group"]
            all_groups = sorted(set(groups_of.values()))
            for g in all_groups:
                gids = {sid for sid, gg in groups_of.items() if gg == g}
                n_sim_failed = len(gids & sim_failed_ids)
                n_qc_failed = len(gids & qc_failed_ids)
                conv_rows.append({
                    "set_key": set_key, "severity_group": g,
                    "n_planned": len(gids),
                    "n_sim_failed": n_sim_failed,
                    "n_qc_failed": n_qc_failed,
                    "n_evaluable": len(gids) - n_sim_failed - n_qc_failed,
                    "evaluable_fraction": (len(gids) - n_sim_failed - n_qc_failed) / max(len(gids), 1),
                })
        if conv_rows:
            conv_path = out_dir / "convergence_report.csv"
            with open(conv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
                w.writeheader()
                w.writerows(conv_rows)
            print(f"输出: {conv_path} ({len(conv_rows)} 行)")
            for r in conv_rows:
                if r["evaluable_fraction"] < 0.5:
                    print(f"[NOTE] {r['set_key']}/{r['severity_group']}: 可评估率 "
                          f"{r['evaluable_fraction']:.0%} < 50% —— 论文中如实报告为"
                          f"超出适用外推半径（与 E4 联动）")
        else:
            print("[WARN] 收敛统计为空 —— convergence_report.csv 未生成"
                  "（验收标准 5 / E4 联动产物）; 请核对上方 [WARN] 的缺失路径与 --repo")
    else:
        print(f"[WARN] 缺 {b_subsets_path} —— 跳过收敛统计, convergence_report.csv 未生成")

    # 机器可读的收尾摘要；后台验收和本地审计均以此确认 A/B 两部分已合并。
    axes = sorted({r["axis"] for r in slope_rows})
    summary = {
        "n_curve_rows": len(long_rows),
        "n_slope_rows": len(slope_rows),
        "n_convergence_rows": len(conv_rows),
        "axis_summary": {
            axis: {
                "n_curves": len([r for r in slope_rows if r["axis"] == axis]),
                "n_positive_slopes": len([
                    r for r in slope_rows
                    if r["axis"] == axis
                    and r["slope_rel_l2_c_per_rank"] > 0
                ]),
                "n_strictly_nondecreasing": len([
                    r for r in slope_rows
                    if r["axis"] == axis and r["strictly_nondecreasing"]
                ]),
            }
            for axis in axes
        },
        "min_evaluable_fraction": (
            min(r["evaluable_fraction"] for r in conv_rows)
            if conv_rows else None
        ),
        "outputs": {
            "curve_long": str(long_path),
            "slopes": str(slope_path),
            "convergence_report": (
                str(out_dir / "convergence_report.csv") if conv_rows else None
            ),
        },
        "interpretation_note": (
            "joint_forcing_x_Ks 的两个档是低 Ks 与高 Ks 两种方向，不是经过"
            "经验验证的单调严重度序列；应分层报告，不能把负斜率解释成模型随"
            "外推增强而改善。"
        ),
    }
    summary_path = out_dir / "severity_curve_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"输出: {summary_path}")

    # 可选画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 全模型视角保留在 CSV；诊断图只画论文主视角 m2_data，避免几十条
        # 曲线和图例遮住坐标轴。不同训练场景/家族仍分别显示。
        plot_rows = [r for r in long_rows if r["model"] == "m2_data"]
        axes_names = sorted({r["axis"] for r in long_rows})
        fig, axs = plt.subplots(
            1, len(axes_names), figsize=(5.2 * len(axes_names), 5.8)
        )
        if len(axes_names) == 1:
            axs = [axs]
        legend_handles = {}
        for ax_plt, axis in zip(axs, axes_names):
            fams = sorted({(r["scenario"], r["family"], r["model"])
                           for r in plot_rows if r["axis"] == axis})
            for scen, fam, model in fams:
                by_rank_plot: dict[int, list[float]] = {}
                for r in plot_rows:
                    if (r["axis"], r["scenario"], r["family"], r["model"]) == (axis, scen, fam, model):
                        by_rank_plot.setdefault(r["severity_rank"], []).append(r["rel_l2_c_mean"])
                if not by_rank_plot:
                    continue
                xs = sorted(by_rank_plot)
                ys = [float(np.mean(by_rank_plot[x])) for x in xs]
                label = f"{scen}/{fam}"
                line, = ax_plt.plot(xs, ys, marker="o", label=label)
                legend_handles.setdefault(label, line)
            if axis == "joint_forcing_x_Ks":
                ax_plt.set_xticks([1, 2], ["low $K_s$\n0.15", "high $K_s$\n40"])
                ax_plt.set_xlabel("joint-shift stratum (not an ordered severity axis)")
            else:
                ax_plt.set_xlabel("severity rank")
            ax_plt.set_ylabel("rel L2 (c), mean")
            ax_plt.set_title(axis)
            ax_plt.grid(alpha=0.2)
        labels = sorted(legend_handles)
        fig.legend(
            [legend_handles[x] for x in labels],
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=7,
        )
        fig.tight_layout(rect=(0, 0.23, 1, 1))
        png = out_dir / "severity_curve.png"
        fig.savefig(png, dpi=150)
        print(f"输出: {png}")
    except ImportError:
        print("[INFO] 无 matplotlib，跳过画图（CSV 已足够）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
