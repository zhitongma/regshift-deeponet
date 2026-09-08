#!/usr/bin/env python3
"""E9 步骤4 (本地可跑, 纯标准库): break-even 核算与成本表生成。

公式 (README §设计):
    N* = (T_data + T_train) / (t_hydrus_eff − t_surrogate)
  - T_data    : 数据生成总墙钟 (HYDRUS 批量 + 后处理), 取自 costs_summary 的数据源 run;
  - T_train   : 代理模型训练墙钟, 取自 costs_summary 的训练 run (默认 RegShift m1);
  - t_hydrus_eff : HYDRUS 有效单样本秒。单核 = 实测单次; K 核并发 = 单核/K
                   (HYDRUS 单线程, 多进程吞吐线性折算);
  - t_surrogate  : 代理单样本秒, 按口径取 bench_inference 结果
                   (GPU batch1/128/1024 摊销、CPU 单线程、CPU 多线程)。
  分母 ≤ 0 (代理不比 HYDRUS 快) 时 N* = ∞, 表中记 "不回本"。

输入全部是本包 outputs/ 下的 JSON; 任一缺失可用 CLI 覆盖项手工注入。
输出: outputs/breakeven_report.md (完整成本表 + N* + 论文场景对照结论)。

用法:
  python3 break_even.py                                  # 全部用默认路径
  python3 break_even.py --hydrus-single-seconds 0.79 \
      --t-data-s 1075 --t-train-s 197                    # 手工注入口径
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import date
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
OUT_DIR = PKG_DIR / "outputs"

SCENARIOS = [
    ("Monte-Carlo 情景筛选 (论文 §5.5 宣称)", 1e4),
    ("MCMC 参数反演 (下限)", 1e5),
    ("MCMC 参数反演 (上限)", 1e6),
]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_record(costs: dict | None, run_name: str) -> dict:
    if not costs:
        return {}
    for rec in costs.get("records", []):
        if rec.get("run_name") == run_name:
            return rec
    return {}


def fmt_s(x):
    if x is None:
        return "—"
    if x >= 3600:
        return f"{x:.0f} s ({x/3600:.2f} h)"
    if x >= 60:
        return f"{x:.1f} s ({x/60:.1f} min)"
    return f"{x:.3g} s"


def fmt_n(x):
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return "不回本 (t_surrogate ≥ t_hydrus)"
    return f"{x:,.0f} (≈10^{math.log10(max(x,1)):.1f})"


def main():
    ap = argparse.ArgumentParser(description="E9: break-even 核算")
    ap.add_argument("--costs", default=str(OUT_DIR / "costs_summary.json"))
    ap.add_argument("--data-run", default="iid_medium_v1",
                    help="T_data 取哪个 run 的 hydrus+process 墙钟 (数据源 run)")
    ap.add_argument("--train-run", default="g4_regshift_iid_s42",
                    help="T_train 取哪个 run 的训练墙钟 (RegShift)")
    ap.add_argument("--train-model", default="m1",
                    help="训练墙钟字段 train_<model>_s, RegShift 走 04_train_m1 即 m1")
    ap.add_argument("--infer-glob", default=str(OUT_DIR / "bench_infer_*.json"))
    ap.add_argument("--hydrus-json", default=str(OUT_DIR / "bench_hydrus_summary.json"))
    ap.add_argument("--hydrus-single-seconds", type=float, default=None,
                    help="覆盖: 实测 HYDRUS 单次秒 (缺省从 bench_hydrus_summary/costs 取)")
    ap.add_argument("--t-data-s", type=float, default=None, help="覆盖 T_data (秒)")
    ap.add_argument("--t-train-s", type=float, default=None, help="覆盖 T_train (秒)")
    ap.add_argument("--cores", default="1,8,32", help="HYDRUS 多核折算的核数列表")
    ap.add_argument("--out", default=str(OUT_DIR / "breakeven_report.md"))
    args = ap.parse_args()

    costs = load_json(Path(args.costs))
    data_rec = find_record(costs, args.data_run)
    train_rec = find_record(costs, args.train_run)

    t_data = args.t_data_s if args.t_data_s is not None else data_rec.get("data_gen_total_s")
    t_train = (args.t_train_s if args.t_train_s is not None
               else train_rec.get(f"train_{args.train_model}_s"))

    hyd = load_json(Path(args.hydrus_json)) or {}
    t_hyd = args.hydrus_single_seconds
    t_hyd_source = "CLI 覆盖"
    if t_hyd is None and hyd.get("median_sample_seconds") is not None:
        t_hyd = float(hyd["median_sample_seconds"])
        t_hyd_source = f"bench_hydrus_summary.json (median, n={hyd.get('n_success')})"
    if t_hyd is None and data_rec.get("hydrus_median_sample_s") is not None:
        t_hyd = float(data_rec["hydrus_median_sample_s"])
        t_hyd_source = f"costs_summary: {args.data_run} 建库时 median_sample_seconds"

    # 推理口径: {label: (t_per_sample_s, 说明)}
    surrogate = {}
    for path in sorted(glob.glob(args.infer_glob)):
        bench = load_json(Path(path)) or {}
        dev = bench.get("device", "?")
        threads = bench.get("torch_threads_note", "")
        hw = bench.get("device_name", "")
        for bs, entry in (bench.get("batches") or {}).items():
            per_s = entry["per_sample_median_ms"] / 1000.0
            label = (f"{dev.upper()} batch={bs}" if dev == "cuda"
                     else f"CPU({threads} 线程) batch={bs}")
            surrogate[label] = (per_s, hw, Path(path).name)
    # B4 兜底口径 (GPU 单样本, 建库评估时记录)
    b4 = (train_rec.get("b4") or {}).get("m1") or {}
    if b4.get("t_infer_ms") is not None:
        surrogate.setdefault("B4 记录 (GPU batch=1, 评估时)",
                             (float(b4["t_infer_ms"]) / 1000.0, "评估节点 GPU",
                              "costs_summary/B4_results"))

    cores = [int(c) for c in args.cores.split(",") if c.strip()]

    lines = []
    lines.append("# E9 成本核算与 break-even 报告 (RegShift vs HYDRUS-1D)")
    lines.append("")
    lines.append(f"生成日期: {date.today().isoformat()}。所有时间为墙钟实测; "
                 "口径与来源逐项标注, 硬件混用处显式声明 (回应 R4-7)。")
    lines.append("")
    lines.append("## 1. 一次性成本 (前期投入)")
    lines.append("")
    lines.append("| 项目 | 墙钟 | 来源 |")
    lines.append("|---|---|---|")
    lines.append(f"| T_data 数据生成 (HYDRUS {data_rec.get('hydrus_n_success','?')} 成功样本"
                 f" + 后处理) | {fmt_s(t_data)} | run `{args.data_run}` manifest |")
    lines.append(f"| T_train 训练 (RegShift {args.train_model}) | {fmt_s(t_train)} "
                 f"| run `{args.train_run}` manifest |")
    if t_data is not None and t_train is not None:
        lines.append(f"| **T_data + T_train** | **{fmt_s(t_data + t_train)}** | |")
    lines.append("")
    lines.append("## 2. 单样本边际成本")
    lines.append("")
    lines.append(f"HYDRUS 单次 (单核): **{fmt_s(t_hyd)}** —— 来源: {t_hyd_source or '缺失'}")
    lines.append("")
    lines.append("| 代理推理口径 | 单样本 | 硬件 | 来源 |")
    lines.append("|---|---|---|---|")
    for label, (per_s, hw, src) in surrogate.items():
        lines.append(f"| {label} | {per_s*1000:.4f} ms | {hw} | {src} |")
    if not surrogate:
        lines.append("| (无 bench_inference 结果, 请先在集群跑步骤2) | — | — | — |")
    lines.append("")
    lines.append("## 3. Break-even 点 N*")
    lines.append("")
    lines.append("N* = (T_data + T_train) / (t_hydrus_eff − t_surrogate); "
                 "t_hydrus_eff = 单核实测 / K (HYDRUS 单线程, K 进程并发线性折算)。")
    lines.append("")

    results = []
    if t_data is not None and t_train is not None and t_hyd is not None and surrogate:
        fixed = t_data + t_train
        lines.append("| 代理口径 \\\\ HYDRUS 并发核数 | " +
                     " | ".join(f"K={k}" for k in cores) + " |")
        lines.append("|---|" + "---|" * len(cores))
        for label, (per_s, _, _) in surrogate.items():
            row = [label]
            for k in cores:
                denom = t_hyd / k - per_s
                n_star = fixed / denom if denom > 0 else float("inf")
                results.append((label, k, n_star))
                row.append(fmt_n(n_star))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("## 4. 与论文宣称应用场景对照")
        lines.append("")
        lines.append("| 场景 | 所需前向次数 N | 结论 (对各口径 N*) |")
        lines.append("|---|---|---|")
        finite = [n for (_, _, n) in results if math.isfinite(n)]
        for name, n_need in SCENARIOS:
            if finite:
                worst = max(finite)
                ok = n_need > worst
                verdict = (f"N = 10^{math.log10(n_need):.0f} "
                           + ("**高于全部口径的 N*** → 代理净收益为正"
                              if ok else
                              f"低于最不利口径 N*={fmt_n(worst)}, 需限定口径表述"))
            else:
                verdict = "缺推理实测, 待步骤2 后重跑"
            lines.append(f"| {name} | {n_need:,.0f} | {verdict} |")
        lines.append("")
        if finite:
            best = min(finite)
            worst = max(finite)
            lines.append(f"**数量级结论**: N* 介于 {fmt_n(best)} 与 {fmt_n(worst)} 之间"
                         f" (固定成本 {fmt_s(fixed)}, HYDRUS 单核单次 {fmt_s(t_hyd)})。"
                         "论文应用场景 (10^4 MC / 10^5–10^6 MCMC) 若均在最不利口径 N* 之上, "
                         "即满足 E9 验收标准。")
    else:
        missing = [n for n, v in [("T_data", t_data), ("T_train", t_train),
                                  ("t_hydrus", t_hyd)] if v is None]
        if not surrogate:
            missing.append("代理推理实测")
        lines.append(f"⚠️ 输入不全, 缺: {', '.join(missing)}。"
                     "请先跑 collect_costs.py / bench_inference.py / bench_hydrus.py, "
                     "或用 CLI 覆盖项注入。")
    lines.append("")
    lines.append("## 5. 回写评估产物")
    lines.append("")
    lines.append("拿到最终数字后, 在集群用 06_evaluate 注入实测成本重生成 B4 cost_curves:")
    lines.append("")
    lines.append("```bash")
    lines.append("python pipeline/06_evaluate.py --config <config> --run-dir <run> \\")
    lines.append(f"    --models m1,m2_data \\")
    lines.append(f"    --hydrus-single-seconds {t_hyd if t_hyd is not None else '<实测秒>'} \\")
    lines.append(f"    --data-gen-seconds {t_data if t_data is not None else '<T_data 秒>'}")
    lines.append("```")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"写出: {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
