#!/usr/bin/env python3
"""E5 步骤 7 (本地可跑, 仅需标准库): 汇总三个评估 run 的 B1/B3 指标为一张五模型表。

用法:
  python3 collect_results.py \
      --vanilla-run  $REPO/experiments/qtop_func/runs_revision/e5_realrain_eval_vanilla_s42 \
      --shift-run    $REPO/experiments/qtop_func/runs_revision/e5_realrain_eval_shift_s42 \
      --regshift-run $REPO/experiments/qtop_func/runs_revision/e5_realrain_eval_regshift_s42 \
      --out-dir ./results_eval --tag s42

模型映射:
  vanilla-run  的 B1 里取 m1 -> DeepONet, fnn -> FNN, fno -> FNO
               (grid_fnn 若存在则作为附加行);
  shift-run    的 m1 -> Shift-DeepONet;
  regshift-run 的 m1 -> RegShift 行, 行名优先取该 run 的
               results/experiments/e5_regshift_provenance.json (step6 按
               E5_REGSHIFT_MODE 写入, 例如 H1 时为「RegShift(历史, 实际无界训练)」),
               其次 --regshift-label, 最后 "RegShift(模式未标注!)"。

输出: e5_results_<tag>.csv / .md (含按 c 的 rel_l2 排序的名次列)。
B1 指标为统计字典, 本表统一取 median (与 06_evaluate 日志/论文主表口径一致)。
注意: 主仓库在 iCloud 盘, 若结果 json 尚未本地化, 先在访达下载或在集群上跑。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_b(run: Path, name: str) -> dict:
    p = run / "results" / "experiments" / f"{name}_results.json"
    if not p.exists():
        print(f"[WARN] 缺少 {p}")
        return {}
    with open(p) as f:
        return json.load(f)


def pick(b1: dict, b3: dict, key: str, label: str) -> dict | None:
    if key not in b1:
        return None
    ent = b1[key]
    row = dict(model=label)
    # B1 每个指标是统计字典 {mean, median, std, p5, p95, values}
    # (compute_ml_metrics_batch, src/evaluation/metrics.py:49-75)。
    # 取 median, 与 06_evaluate 日志及论文主表口径一致。
    # compute_ml_metrics 只产出 rel_l2 / r2 / max_abs (metrics.py:42-46)。
    for var in ("h", "c"):
        for metric in ("rel_l2", "r2", "max_abs"):
            v = ent.get(var, {}).get(metric)
            if isinstance(v, dict):
                v = v.get("median")
            if v is not None:
                row[f"{var}_{metric}"] = v
    b3e = b3.get(key, {})
    for metric in ("front_error_mean_cm", "bt_mae_hours", "n_breakthrough"):
        if metric in b3e:
            row[metric] = b3e[metric]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-run", required=True)
    ap.add_argument("--shift-run", required=True)
    ap.add_argument("--regshift-run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="s42")
    ap.add_argument("--regshift-label", default=None,
                    help="RegShift 行名覆盖 (缺 provenance 文件时的备选)")
    args = ap.parse_args()

    # RegShift 行名: provenance (step6 按 E0 定性写入) > CLI 覆盖 > 显式的未标注提示
    regshift_run = Path(args.regshift_run)
    prov_path = regshift_run / "results" / "experiments" / "e5_regshift_provenance.json"
    if prov_path.exists():
        with open(prov_path) as f:
            prov = json.load(f)
        regshift_label = prov.get("label", "RegShift")
        print(f"[OK] RegShift provenance: mode={prov.get('e5_regshift_mode')} -> 行名「{regshift_label}」")
    elif args.regshift_label:
        regshift_label = args.regshift_label
        print(f"[WARN] 缺 {prov_path.name}, 使用 --regshift-label「{regshift_label}」")
    else:
        regshift_label = "RegShift(模式未标注!)"
        print(f"[WARN] 缺 {prov_path.name} 且未给 --regshift-label —— "
              f"行名标为「{regshift_label}」, 该行进论文前必须先按 E0 定性补标注 (README §2.4)")

    rows = []
    van = Path(args.vanilla_run)
    b1, b3 = load_b(van, "B1"), load_b(van, "B3")
    for key, label in [("m1", "DeepONet"), ("fnn", "FNN"), ("fno", "FNO"),
                       ("grid_fnn", "grid_FNN(附加)")]:
        r = pick(b1, b3, key, label)
        if r:
            rows.append(r)
        elif label != "grid_FNN(附加)":
            print(f"[WARN] {van.name} 的 B1 缺少 {key}")

    for run, label in [(Path(args.shift_run), "Shift-DeepONet"),
                       (regshift_run, regshift_label)]:
        r = pick(load_b(run, "B1"), load_b(run, "B3"), "m1", label)
        if r:
            rows.append(r)
        else:
            print(f"[WARN] {run.name} 缺少 m1 结果")

    if not rows:
        raise SystemExit("[FAIL] 未收集到任何结果")

    ranked = sorted((r for r in rows if "c_rel_l2" in r), key=lambda r: r["c_rel_l2"])
    for i, r in enumerate(ranked, 1):
        r["rank_by_c_rel_l2"] = i

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "rank_by_c_rel_l2",
              "h_rel_l2", "c_rel_l2", "h_r2", "c_r2", "h_max_abs", "c_max_abs",
              "front_error_mean_cm", "bt_mae_hours", "n_breakthrough"]
    fields = [f for f in fields if any(f in r for r in rows)]
    csv_path = out_dir / f"e5_results_{args.tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md = [f"# E5 外部真实降雨测试集五模型结果 ({args.tag})", "",
          "| " + " | ".join(fields) + " |",
          "|" + "---|" * len(fields)]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r.get(f):.4f}" if isinstance(r.get(f), float) else str(r.get(f, ""))
            for f in fields) + " |")
    md += ["", "排序依据: c 的 rel_l2 (穿透/浓度场是论文主指标之一, 可按需换 h)。",
           "请把该排序与论文主表 (投稿版 IID 主基准) 的排序并排登记进 README 的结果登记表。"]
    (out_dir / f"e5_results_{args.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] {csv_path} 与同名 .md")


if __name__ == "__main__":
    main()
