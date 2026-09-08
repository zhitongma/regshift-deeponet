#!/usr/bin/env python3
"""E9 步骤3 (集群执行, 需 HYDRUS 可执行文件 + phydrus/pandas): 单次 HYDRUS 实测计时。

做法:
  1. 新建 run 目录 $REPO/experiments/qtop_func/runs_revision/e9_hydrus_timing_iid_v1;
  2. 只把源 run (iid_medium_v1) 的 data/parameters symlink 进来 (⚠️ 故意不链接
     data/raw —— 02_run_hydrus 的断点续跑靠 raw/progress.json, 链接 raw 会把样本
     标成"已完成"而直接跳过; 也不需要 processed);
  3. 调 pipeline/02_run_hydrus.py --start 0 --end 20 重算前 20 个样本;
  4. 从新 run 的 metadata/hydrus_batch_manifest.json 读 mean/median_sample_seconds,
     摘要写回实验包 outputs/bench_hydrus_summary.json。

口径说明 (README §设计): HYDRUS-1D 为单线程程序, 单次耗时即单核耗时;
K 核并发跑 K 个独立样本时吞吐近似线性, 故多核有效单样本时间 = 单核时间 / K。

用法 (集群):
  python bench_hydrus.py --repo <集群上的仓库根> --exe hydrus \
      [--source-run <...>/runs/iid_medium_v1] [--start 0 --end 20] \
      [--out bench_hydrus_summary.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))
PKG_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="E9: HYDRUS 单次模拟计时 (集群)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--source-run", default=None,
                    help="缺省: <repo>/experiments/qtop_func/runs/iid_medium_v1")
    ap.add_argument("--run-dir", default=None,
                    help="缺省: <repo>/experiments/qtop_func/runs_revision/e9_hydrus_timing_iid_v1")
    ap.add_argument("--config", default=str(PKG_DIR.parents[1] / "configs/revision/E9_cost/e9_bench_hydrus_iid.yaml"))
    ap.add_argument("--exe", default="hydrus")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--out", default=str(PKG_DIR / "outputs/bench_hydrus_summary.json"))
    args = ap.parse_args()

    repo = Path(args.repo)
    source_run = Path(args.source_run or repo / "experiments/qtop_func/runs/iid_medium_v1")
    run_dir = Path(args.run_dir or
                   repo / "experiments/qtop_func/runs_revision/e9_hydrus_timing_iid_v1")

    params_src = source_run / "data/parameters"
    if not (params_src / "lhs_params.csv").exists():
        sys.exit(f"[FAIL] 源 run 缺 lhs_params.csv: {params_src}")

    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    link = run_dir / "data/parameters"
    if not link.exists():
        os.symlink(params_src, link)
        print(f"symlink: {link} -> {params_src}")
    elif link.resolve() != params_src.resolve():
        sys.exit(f"[FAIL] {link} 已存在且指向别处: {link.resolve()}")

    raw_dir = run_dir / "data/raw"
    if raw_dir.is_symlink():
        sys.exit(f"[FAIL] {raw_dir} 是 symlink —— 会因 progress.json 跳过全部样本, 请移除后重跑")
    progress = raw_dir / "progress.json"
    if progress.exists():
        print(f"[warn] 已有 {progress}, 其中记录的样本会被跳过 (断点续跑语义); "
              f"如需重新计时请先删除该文件与对应 sample_*.npz")

    cmd = [
        sys.executable, str(repo / "pipeline/02_run_hydrus.py"),
        "--config", str(Path(args.config).resolve()),
        "--run-dir", str(run_dir),
        "--exe", args.exe,
        "--start", str(args.start),
        "--end", str(args.end),
    ]
    print("执行:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))

    manifest_path = run_dir / "metadata/hydrus_batch_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    times = [float(v) for v in (manifest.get("sample_times_seconds") or {}).values()]
    # 首样本含目录/缓存预热 (源 run manifest 中样本 0 明显偏慢), 稳健口径用 median
    summary = {
        "source_run": str(source_run),
        "run_dir": str(run_dir),
        "config": str(args.config),
        "exe": args.exe,
        "start": args.start,
        "end": args.end,
        "n_success": manifest.get("n_success"),
        "n_fail": manifest.get("n_fail"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "mean_sample_seconds": manifest.get("mean_sample_seconds"),
        "median_sample_seconds": manifest.get("median_sample_seconds"),
        "trimmed_mean_seconds_drop_first": (
            statistics.fmean(times[1:]) if len(times) > 1 else None),
        "note": ("HYDRUS-1D 单线程; 单次耗时=单核耗时; K 核并发的有效单样本时间"
                 "=单核时间/K (线性折算, 见 README 口径说明)"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"写出: {out}")


if __name__ == "__main__":
    main()
