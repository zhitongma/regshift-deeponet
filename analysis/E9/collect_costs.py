#!/usr/bin/env python3
"""E9 步骤1 (本地可跑): 从主 benchmark run 汇总已有成本记录。

只读纯文本 JSON:
  <run>/metadata/hydrus_batch_manifest.json      -> 数据生成 (HYDRUS 批量) 墙钟
  <run>/metadata/process_data_manifest.json      -> 后处理墙钟
  <run>/metadata/train_*_manifest.json           -> 训练墙钟 training_time_seconds
  <run>/results/experiments/m1_history.json      -> 训练历史 (training_time_min 等)
  <run>/results/experiments/m2_history.json
  <run>/results/experiments/B4_results.json      -> t_infer_ms / t_train_min / n_cross

⚠️ iCloud 安全: 主仓库在 iCloud 同步盘上, 未物化 (dataless) 文件读取会永久挂起。
本脚本对每个文件先 os.stat(): st_size>0 且 st_blocks==0 判为 dataless, 跳过并告警
(可先执行 `brctl download <文件>` 物化后重跑)。绝不读取 .npz/.pt。

用法:
  python3 collect_costs.py                       # 默认扫描主基准 run 清单
  python3 collect_costs.py --runs <run_dir> ...  # 指定 run 目录
  python3 collect_costs.py --out-dir outputs

输出: <out-dir>/costs_summary.json + costs_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO", str(Path(__file__).resolve().parents[1])))
RUNS = REPO / "experiments/qtop_func/runs"
RUNS_MULTISEED = REPO / "experiments/qtop_func/runs_multiseed"
RUNS_REVISION = REPO / "experiments/qtop_func/runs_revision"

# 主基准 run 默认清单 (存在才收集; 多种子目录会自动展开 seed_*/)
DEFAULT_RUNS = [
    RUNS / "iid_medium_v1",
    RUNS / "ood_peak_medium_v1",
    RUNS / "ood_peak_time_medium_v2",
    RUNS / "g4_regshift_iid_s42",
    RUNS / "g4_regshift_iid_s123",
    RUNS / "g4_regshift_iid_s456",
    RUNS / "g4_regshift_ood_peak_s42",
    RUNS / "g4_regshift_ood_peak_s123",
    RUNS / "g4_regshift_ood_peak_s456",
    RUNS / "g4_regshift_ood_time_s42",
    RUNS / "g4_regshift_ood_time_s123",
    RUNS / "g4_regshift_ood_time_s456",
    RUNS_MULTISEED / "iid_medium_v1",
    RUNS_MULTISEED / "ood_peak_medium_v1",
    RUNS_MULTISEED / "ood_peak_time_medium_v2",
]

TRAIN_MANIFESTS = {
    "m1": "train_m1_manifest.json",
    "m2": "train_m2_manifest.json",
    "fnn": "train_fnn_manifest.json",
    "fno": "train_fno_manifest.json",
    "grid_fnn": "train_grid_fnn_manifest.json",
}
HISTORY_FILES = ["m1_history.json", "m2_history.json"]


def is_dataless(path: Path) -> bool:
    """iCloud 占位文件: 报告 size>0 但磁盘上 0 块。读它会触发下载并可能挂起。"""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return False
    return st.st_size > 0 and getattr(st, "st_blocks", 1) == 0


def safe_load_json(path: Path, warnings: list[str]):
    if not path.exists():
        return None
    if is_dataless(path):
        warnings.append(
            f"[dataless] {path} 未物化 (iCloud 占位), 已跳过。"
            f" 物化后重跑: brctl download '{path}'")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.append(f"[error] 读取失败 {path}: {e}")
        return None


def extract_train_seconds(hist: dict | None, manifest: dict | None):
    """训练墙钟秒: manifest.training_time_seconds 优先, history 的 min/seconds 兜底。"""
    if manifest and isinstance(manifest.get("training_time_seconds"), (int, float)):
        return float(manifest["training_time_seconds"]), "manifest.training_time_seconds"
    if hist:
        for key, scale in (("training_time_seconds", 1.0), ("training_time_min", 60.0),
                           ("training_time", 1.0)):
            v = hist.get(key)
            if isinstance(v, (int, float)):
                return float(v) * scale, f"history.{key}"
    return None, None


def collect_run(run_dir: Path, warnings: list[str]) -> dict:
    meta = run_dir / "metadata"
    exp = run_dir / "results/experiments"
    name = (f"{run_dir.parent.name}/{run_dir.name}"
            if run_dir.name.startswith("seed_") else run_dir.name)
    rec: dict = {"run": str(run_dir), "run_name": name}

    hb = safe_load_json(meta / "hydrus_batch_manifest.json", warnings)
    if hb:
        rec["hydrus_elapsed_s"] = hb.get("elapsed_seconds")
        rec["hydrus_mean_sample_s"] = hb.get("mean_sample_seconds")
        rec["hydrus_median_sample_s"] = hb.get("median_sample_seconds")
        rec["hydrus_n_success"] = hb.get("n_success")
        rec["hydrus_n_fail"] = hb.get("n_fail")

    pd_manifest = safe_load_json(meta / "process_data_manifest.json", warnings)
    if pd_manifest:
        rec["process_elapsed_s"] = pd_manifest.get("elapsed_seconds")

    # 注: 不读 generate_params_manifest.json —— 主仓库 01_generate_params.py 的
    # manifest 只写 n_total/split_mode/seed/summary 等, 没有 elapsed_seconds;
    # 参数生成耗时秒级, README 亦称可忽略, 不计入成本。

    histories = {name.split("_")[0]: safe_load_json(exp / name, warnings)
                 for name in HISTORY_FILES}
    for model, fname in TRAIN_MANIFESTS.items():
        manifest = safe_load_json(meta / fname, warnings)
        hist = histories.get(model)
        seconds, source = extract_train_seconds(hist, manifest)
        if seconds is not None:
            rec[f"train_{model}_s"] = seconds
            rec[f"train_{model}_source"] = source
        if manifest:
            rec[f"train_{model}_epochs"] = manifest.get("total_epochs")

    b4 = safe_load_json(exp / "B4_results.json", warnings)
    if b4:
        rec["b4"] = {}
        for model, entry in b4.items():
            if not isinstance(entry, dict):
                continue
            rec["b4"][model] = {
                "t_infer_ms": entry.get("t_infer_ms"),
                "t_train_min": entry.get("t_train_min"),
                "n_cross": entry.get("n_cross"),
                "timing_sources": entry.get("timing_sources"),
            }

    # 数据生成总墙钟 (HYDRUS 批量 + 后处理), 与 06_evaluate 的 data_gen_seconds 口径一致
    if rec.get("hydrus_elapsed_s") is not None:
        rec["data_gen_total_s"] = float(rec["hydrus_elapsed_s"]) + float(
            rec.get("process_elapsed_s") or 0.0)
    return rec


def expand_runs(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if not p.is_dir():
            continue
        seeds = sorted(d for d in p.glob("seed_*") if d.is_dir())
        if seeds:  # runs_multiseed/<scene>/seed_*/ 结构 (父级 metadata 只有提交脚本)
            out.extend(seeds)
        elif (p / "metadata").is_dir() or (p / "results").is_dir():
            out.append(p)
    return out


CSV_COLS = [
    "run_name", "hydrus_elapsed_s", "hydrus_mean_sample_s", "hydrus_median_sample_s",
    "hydrus_n_success", "hydrus_n_fail", "process_elapsed_s", "data_gen_total_s",
    "train_m1_s", "train_m2_s", "train_fnn_s", "train_fno_s", "train_grid_fnn_s",
    "b4_m1_t_infer_ms", "b4_m2_data_t_infer_ms", "b4_m1_n_cross", "b4_m2_data_n_cross",
    "run",
]


def main():
    ap = argparse.ArgumentParser(description="E9: 汇总各主 benchmark run 的成本记录")
    ap.add_argument("--runs", nargs="*", default=None, help="run 目录列表 (缺省用内置清单)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    run_dirs = ([Path(r) for r in args.runs] if args.runs else list(DEFAULT_RUNS))
    if not args.runs and RUNS_REVISION.is_dir():  # 修回阶段的新 run 也纳入
        run_dirs += sorted(d for d in RUNS_REVISION.iterdir() if d.is_dir())
    run_dirs = expand_runs(run_dirs)

    warnings: list[str] = []
    records = [collect_run(rd, warnings) for rd in run_dirs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "costs_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"records": records, "warnings": warnings}, f,
                  indent=2, ensure_ascii=False)

    out_csv = out_dir / "costs_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            row = dict(rec)
            b4 = rec.get("b4") or {}
            for model in ("m1", "m2_data"):
                entry = b4.get(model) or {}
                row[f"b4_{model}_t_infer_ms"] = entry.get("t_infer_ms")
                row[f"b4_{model}_n_cross"] = entry.get("n_cross")
            w.writerow(row)

    print(f"收集 {len(records)} 个 run -> {out_json}\n            -> {out_csv}")
    for msg in warnings:
        print(msg)
    if warnings:
        print(f"\n共 {len(warnings)} 条告警。dataless 文件请先 brctl download 物化再重跑。")
        sys.exit(2)


if __name__ == "__main__":
    main()
