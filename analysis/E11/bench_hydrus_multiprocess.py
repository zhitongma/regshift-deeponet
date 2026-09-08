#!/usr/bin/env python3
"""Measure actual HYDRUS-1D ensemble throughput with multiple processes.

This benchmark closes the E9 limitation: it launches independent HYDRUS-1D
jobs concurrently and measures wall-clock throughput instead of assuming ideal
1/K scaling from a single-process timing.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import yaml


_REPO: Path | None = None
_EXE: str | None = None
_PHYSICS: dict | None = None
_ROWS: dict[int, dict] | None = None
_WS_ROOT: Path | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _init_worker(repo: str, exe: str, physics: dict, rows: dict[int, dict], ws_root: str) -> None:
    global _REPO, _EXE, _PHYSICS, _ROWS, _WS_ROOT
    _REPO = Path(repo)
    _EXE = exe
    _PHYSICS = physics
    _ROWS = rows
    _WS_ROOT = Path(ws_root)
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    logging.disable(logging.CRITICAL)


def _run_one(sample_id: int) -> dict:
    assert _REPO is not None and _EXE is not None and _PHYSICS is not None
    assert _ROWS is not None and _WS_ROOT is not None
    from src.data_generation.hydrus_runner import run_single_simulation
    from src.data_generation.sampling import build_sample_params

    row = pd.Series(_ROWS[sample_id])
    params = build_sample_params(row, cfg={"physics": _PHYSICS})
    t0 = time.perf_counter()
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            result = run_single_simulation(
                sample_id=sample_id,
                params=params,
                exe_path=_EXE,
                base_ws=str(_WS_ROOT),
                physics_cfg=_PHYSICS,
                keep_workspace=False,
            )
        success = bool(result.get("success"))
        error = None if success else str(result.get("error") or "unknown failure")
    except Exception as exc:  # benchmark must count failures without aborting the pool
        success = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "sample_id": sample_id,
        "success": success,
        "elapsed_seconds": time.perf_counter() - t0,
        "error": error,
    }


def _hardware() -> dict:
    try:
        lscpu = subprocess.run(["lscpu"], check=False, capture_output=True, text=True).stdout
    except OSError:
        lscpu = ""
    model = ""
    for line in lscpu.splitlines():
        if line.lower().startswith("model name"):
            model = line.split(":", 1)[-1].strip()
            break
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": model,
        "logical_cpu_count": os.cpu_count(),
        "lscpu": lscpu,
    }


def _write_outputs(out_json: Path, records: list[dict], metadata: dict) -> None:
    by_k: dict[int, list[dict]] = {}
    for rec in records:
        by_k.setdefault(int(rec["processes"]), []).append(rec)

    aggregates = []
    baseline = statistics.fmean(r["attempted_throughput_per_s"] for r in by_k.get(1, [])) if by_k.get(1) else None
    for k in sorted(by_k):
        group = by_k[k]
        throughputs = [r["attempted_throughput_per_s"] for r in group]
        successful_throughputs = [r["successful_throughput_per_s"] for r in group]
        mean_tp = statistics.fmean(throughputs)
        speedup = mean_tp / baseline if baseline else None
        aggregates.append({
            "processes": k,
            "n_repeats": len(group),
            "n_attempted_total": sum(r["n_attempted"] for r in group),
            "n_success_total": sum(r["n_success"] for r in group),
            "n_fail_total": sum(r["n_fail"] for r in group),
            "wall_seconds_mean": statistics.fmean(r["wall_seconds"] for r in group),
            "wall_seconds_std": statistics.stdev(r["wall_seconds"] for r in group) if len(group) > 1 else 0.0,
            "attempted_throughput_per_s_mean": mean_tp,
            "attempted_throughput_per_s_std": statistics.stdev(throughputs) if len(group) > 1 else 0.0,
            "successful_throughput_per_s_mean": statistics.fmean(successful_throughputs),
            "successful_throughput_per_s_std": statistics.stdev(successful_throughputs) if len(group) > 1 else 0.0,
            "speedup_vs_1_process": speedup,
            "parallel_efficiency": speedup / k if speedup is not None else None,
        })

    payload = {"metadata": metadata, "records": records, "aggregates": aggregates}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_json)

    out_csv = out_json.with_suffix(".csv")
    fields = [
        "processes", "n_repeats", "n_attempted_total", "n_success_total", "n_fail_total",
        "wall_seconds_mean", "wall_seconds_std", "attempted_throughput_per_s_mean",
        "attempted_throughput_per_s_std", "successful_throughput_per_s_mean",
        "successful_throughput_per_s_std", "speedup_vs_1_process", "parallel_efficiency",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--params-csv", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", default="1,2,4,8,16")
    ap.add_argument("--n-samples", type=int, default=192)
    ap.add_argument("--start", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    config_path = Path(args.config).resolve()
    params_path = Path(args.params_csv).resolve()
    exe_path = Path(args.exe).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_json = Path(args.out).resolve()
    process_counts = [int(x) for x in args.processes.split(",") if x.strip()]

    if not repo.is_dir() or not config_path.is_file() or not params_path.is_file() or not exe_path.is_file():
        raise SystemExit("repo/config/params/exe path check failed")
    if max(process_counts) > (os.cpu_count() or 1):
        raise SystemExit("requested process count exceeds logical CPU count")

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params_df = pd.read_csv(params_path, index_col="sample_id")
    ids = [int(x) for x in params_df.index if int(x) >= args.start][: args.n_samples]
    if len(ids) != args.n_samples:
        raise SystemExit(f"requested {args.n_samples} samples but found {len(ids)}")
    rows = {i: params_df.loc[i].to_dict() for i in ids}

    metadata = {
        "created_unix": time.time(),
        "repo": str(repo),
        "config": str(config_path),
        "params_csv": str(params_path),
        "exe": str(exe_path),
        "exe_sha256": _sha256(exe_path),
        "sample_start": ids[0],
        "sample_end_inclusive": ids[-1],
        "n_samples_per_cell": len(ids),
        "repeats_requested": args.repeats,
        "process_counts": process_counts,
        "warmup_samples": args.warmup,
        "timing_scope": "input writing + HYDRUS solve + output parsing; workspace cleanup included",
        "throughput_definition": "attempted samples divided by measured wall time; successful throughput also reported",
        "hardware": _hardware(),
    }

    records: list[dict] = []
    if out_json.exists():
        existing = json.loads(out_json.read_text(encoding="utf-8"))
        records = list(existing.get("records", []))
    completed = {(int(r["repeat"]), int(r["processes"])) for r in records}

    work_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup > 0 and not records:
        warm_ws = work_dir / "warmup"
        _init_worker(str(repo), str(exe_path), cfg["physics"], rows, str(warm_ws))
        warm_times = [_run_one(i) for i in ids[: args.warmup]]
        metadata["warmup_success"] = sum(int(x["success"]) for x in warm_times)

    # Counterbalanced order limits bias from cache warm-up and time drift.
    base_order = [4, 1, 16, 2, 8]
    order = [k for k in base_order if k in process_counts] + [k for k in process_counts if k not in base_order]
    for repeat in range(1, args.repeats + 1):
        run_order = order if repeat % 2 else list(reversed(order))
        for k in run_order:
            if (repeat, k) in completed:
                print(f"[skip] repeat={repeat} processes={k}", flush=True)
                continue
            ws_root = work_dir / f"repeat_{repeat}" / f"k_{k}"
            ws_root.mkdir(parents=True, exist_ok=True)
            print(f"[start] repeat={repeat} processes={k} n={len(ids)}", flush=True)
            ctx = mp.get_context("fork")
            t0 = time.perf_counter()
            with ctx.Pool(
                processes=k,
                initializer=_init_worker,
                initargs=(str(repo), str(exe_path), cfg["physics"], rows, str(ws_root)),
            ) as pool:
                results = list(pool.imap_unordered(_run_one, ids, chunksize=1))
            wall = time.perf_counter() - t0
            n_success = sum(int(r["success"]) for r in results)
            n_fail = len(results) - n_success
            rec = {
                "repeat": repeat,
                "processes": k,
                "n_attempted": len(results),
                "n_success": n_success,
                "n_fail": n_fail,
                "wall_seconds": wall,
                "attempted_throughput_per_s": len(results) / wall,
                "successful_throughput_per_s": n_success / wall,
                "worker_elapsed_median_seconds": statistics.median(r["elapsed_seconds"] for r in results),
                "worker_elapsed_mean_seconds": statistics.fmean(r["elapsed_seconds"] for r in results),
                "failure_examples": [r["error"] for r in results if not r["success"]][:5],
            }
            records.append(rec)
            _write_outputs(out_json, records, metadata)
            print(
                f"[done] repeat={repeat} processes={k} wall={wall:.3f}s "
                f"throughput={rec['attempted_throughput_per_s']:.3f}/s "
                f"success={n_success}/{len(results)}",
                flush=True,
            )

    _write_outputs(out_json, records, metadata)
    print(f"[complete] {out_json}", flush=True)


if __name__ == "__main__":
    main()
