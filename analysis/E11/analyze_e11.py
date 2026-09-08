#!/usr/bin/env python3
"""Turn the E11 throughput JSON into a paper-ready audit table and break-even report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("e11_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-generation-seconds", type=float, default=1030.0)
    ap.add_argument("--training-seconds", type=float, default=3533.963215827942)
    ap.add_argument("--gpu-batch1-seconds", type=float, default=0.0009420076385140419)
    ap.add_argument("--gpu-batch128-seconds", type=float, default=0.0005773185403086245)
    ap.add_argument("--cpu1-seconds", type=float, default=0.09484570845961571)
    args = ap.parse_args()

    source = Path(args.e11_json)
    data = json.loads(source.read_text(encoding="utf-8"))
    aggs = sorted(data["aggregates"], key=lambda x: int(x["processes"]))
    fixed = args.data_generation_seconds + args.training_seconds

    lines = [
        "# E11 HYDRUS-1D 多进程实测与 break-even 更新",
        "",
        f"数据源：`{source.name}`。固定成本口径为建库 {args.data_generation_seconds:.1f} s + "
        f"真有界模型 m1+m2 训练 {args.training_seconds:.1f} s = **{fixed:.1f} s ({fixed/3600:.2f} h)**。",
        "",
        "## 1. 实测吞吐",
        "",
        "| HYDRUS 进程 | 成功/尝试 | wall time [s] | 尝试吞吐 [s^-1] | 成功吞吐 [s^-1] | speed-up | 并行效率 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for a in aggs:
        k = int(a["processes"])
        lines.append(
            f"| {k} | {a['n_success_total']}/{a['n_attempted_total']} | "
            f"{a['wall_seconds_mean']:.2f}±{a['wall_seconds_std']:.2f} | "
            f"{a['attempted_throughput_per_s_mean']:.3f}±{a['attempted_throughput_per_s_std']:.3f} | "
            f"{a['successful_throughput_per_s_mean']:.3f}±{a['successful_throughput_per_s_std']:.3f} | "
            f"{a['speedup_vs_1_process']:.2f} | {100*a['parallel_efficiency']:.1f}% |"
        )

    lines.extend([
        "",
        "尝试吞吐计入不收敛样本消耗的墙钟成本；成功吞吐只计实际返回有效场的样本。",
        "",
        "## 2. 基于实测吞吐的 break-even",
        "",
        "\\(N^*=T_{fixed}/(t_{HYDRUS,eff}-t_{surrogate})\\)。HYDRUS 有效单样本墙钟"
        "取实测吞吐的倒数，不再使用 1/K 假设。",
        "",
        "| HYDRUS 进程 | 实测 t_eff [s/尝试] | N* GPU batch1 | N* GPU batch128 | N* CPU-1 thread |",
        "|---:|---:|---:|---:|---:|",
    ])

    for a in aggs:
        k = int(a["processes"])
        t_eff = 1.0 / a["attempted_throughput_per_s_mean"]
        crosses = []
        for ts in (args.gpu_batch1_seconds, args.gpu_batch128_seconds, args.cpu1_seconds):
            crosses.append(None if t_eff <= ts else fixed / (t_eff - ts))
        lines.append(
            f"| {k} | {t_eff:.4f} | "
            f"{fmt(crosses[0], 0)} | {fmt(crosses[1], 0)} | {fmt(crosses[2], 0)} |"
        )

    max_tp = max(aggs, key=lambda x: x["attempted_throughput_per_s_mean"])
    lines.extend([
        "",
        "## 3. 论文口径",
        "",
        f"- 本节最高实测尝试吞吐出现在 {int(max_tp['processes'])} 进程："
        f"{max_tp['attempted_throughput_per_s_mean']:.3f} 样本/s；不得再声称 HYDRUS 吞吐按核数线性提升。",
        "- GPU 与 HYDRUS 加速比是跨硬件 wall-clock 比较；CPU 单线程推理只能与 HYDRUS 单进程作近似 like-for-like 参考。",
        "- 10,000 次调用是否回本必须按实测并发度逐档判断，不再使用“必然在数分钟内且明显回本”表述。",
        "- 不收敛率是成本和适用域的一部分，不应从吞吐分母中删除。",
    ])

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
