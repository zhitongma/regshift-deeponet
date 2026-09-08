#!/usr/bin/env python3
"""
Step 2: 批量运行 HYDRUS-1D 模拟。

支持:
  - run_dir 隔离实验产物
  - 断点续跑
  - 保存真实边界通量与 HYDRUS 平衡误差
"""

import os
import argparse
import json
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.hydrus_runner import run_single_simulation
from src.data_generation.sampling import build_sample_params
from src.utils.experiment import (
    configure_logger,
    ensure_dirs,
    load_config,
    resolve_artifact_paths,
    save_config_snapshot,
    save_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="批量 HYDRUS-1D 模拟")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--run-dir", type=str, default=None, help="实验根目录")
    parser.add_argument("--exe", type=str, default="hydrus",
                        help="HYDRUS-1D 可执行文件路径")
    parser.add_argument("--start", type=int, default=0,
                        help="起始样本 index")
    parser.add_argument("--end", type=int, default=None,
                        help="结束样本 index (不含)")
    parser.add_argument("--keep-ws", action="store_true",
                        help="保留 HYDRUS 工作目录 (调试用)")
    args = parser.parse_args()

    cfg, cfg_path = load_config(PROJECT_ROOT, args.config)
    paths = resolve_artifact_paths(PROJECT_ROOT, cfg, args.run_dir)
    ensure_dirs(paths, "parameters", "raw_data", "logs", "metadata")
    logger = configure_logger(__name__, paths.logs / "hydrus_batch.log")

    params_path = paths.parameters / "lhs_params.csv"
    if not params_path.exists():
        logger.error("参数文件不存在: %s\n请先运行 scripts/01_generate_params.py", params_path)
        sys.exit(1)

    params_df = pd.read_csv(params_path, index_col="sample_id")
    end = args.end if args.end is not None else len(params_df)
    params_df = params_df.iloc[args.start:end]

    raw_dir = paths.raw_data
    raw_dir.mkdir(parents=True, exist_ok=True)
    ws_dir = paths.root / "hydrus_workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)

    progress_file = raw_dir / "progress.json"
    completed = _load_progress(progress_file)
    sample_times: dict[int, float] = {}
    failed_samples: dict[int, str] = {}

    save_config_snapshot(
        cfg,
        cfg_path,
        paths,
        "hydrus_batch",
        extra={"exe": args.exe, "start": args.start, "end": end, "keep_ws": args.keep_ws},
    )

    logger.info("=" * 60)
    logger.info("批量 HYDRUS 模拟")
    logger.info("  run_root: %s", paths.root)
    logger.info("  总样本: %d, 已完成: %d, 待运行: %d",
                len(params_df),
                sum(1 for i in params_df.index if i in completed),
                sum(1 for i in params_df.index if i not in completed))
    logger.info("  HYDRUS 路径: %s", args.exe)
    logger.info("=" * 60)

    n_success = 0
    n_fail = 0
    t_start = time.time()

    for idx, row in params_df.iterrows():
        if idx in completed:
            continue

        params = build_sample_params(row, cfg=cfg)
        logger.info("运行样本 %d/%d (index=%d) ...", n_success + n_fail + 1,
                     len(params_df) - len(completed), idx)

        t0 = time.time()
        result = run_single_simulation(
            sample_id=idx,
            params=params,
            exe_path=args.exe,
            base_ws=str(ws_dir),
            physics_cfg=cfg["physics"],
            keep_workspace=args.keep_ws,
        )
        dt = time.time() - t0

        if result["success"]:
            out_path = raw_dir / f"sample_{idx:04d}.npz"
            np.savez_compressed(
                out_path,
                h=result["h"],
                c=result["c"],
                theta=result["theta"],
                z=result["z"],
                t=result["t"],
                water_flux_top=result["water_flux_top"],
                water_flux_bottom=result["water_flux_bottom"],
                water_cum_top=result["water_cum_top"],
                water_cum_bottom=result["water_cum_bottom"],
                water_cum_root=result["water_cum_root"],
                water_cum_runoff=result["water_cum_runoff"],
                water_cum_net=result["water_cum_net"],
                solute_flux_top=result["solute_flux_top"],
                solute_flux_bottom=result["solute_flux_bottom"],
                solute_cum_top=result["solute_cum_top"],
                solute_cum_bottom=result["solute_cum_bottom"],
                solute_cum_root=result["solute_cum_root"],
                solute_cum_runoff=result["solute_cum_runoff"],
                solute_cum_net=result["solute_cum_net"],
                water_balance_rel=result["water_balance_rel"],
                solute_balance_rel=result["solute_balance_rel"],
                params=json.dumps(params),
            )
            completed[idx] = True
            _save_progress(progress_file, completed)
            sample_times[idx] = dt
            n_success += 1
            logger.info("  -> 成功 (%.1f s), 保存: %s", dt, out_path.name)
        else:
            n_fail += 1
            failed_samples[idx] = result["error"]
            logger.warning("  -> 失败 (%.1f s): %s", dt, result["error"])

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("完成! 成功: %d, 失败: %d, 总耗时: %.1f min",
                n_success, n_fail, elapsed / 60)
    logger.info("有效数据率: %.1f%%",
                100 * n_success / max(n_success + n_fail, 1))

    save_manifest(
        paths,
        "hydrus_batch",
        {
            "params_csv": str(params_path),
            "raw_dir": str(raw_dir),
            "workspace_dir": str(ws_dir),
            "n_total_considered": len(params_df),
            "n_success": n_success,
            "n_fail": n_fail,
            "elapsed_seconds": elapsed,
            "mean_sample_seconds": float(np.mean(list(sample_times.values()))) if sample_times else None,
            "median_sample_seconds": float(np.median(list(sample_times.values()))) if sample_times else None,
            "sample_times_seconds": sample_times,
            "failed_samples": failed_samples,
        },
    )


def _load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    return {}


def _save_progress(path: Path, completed: dict):
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in completed.items()}, f)


if __name__ == "__main__":
    main()
