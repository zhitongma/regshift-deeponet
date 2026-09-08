#!/usr/bin/env python3
"""E9 步骤2 (集群执行, 需 torch): RegShift checkpoint 推理计时。

同一脚本按 --device/--torch-threads 分三种口径各跑一次 (见 run.sh):
  1. GPU:           --device cuda
  2. CPU 单线程:    --device cpu --torch-threads 1   (与 HYDRUS 单核口径同硬件对比)
  3. CPU 多线程:    --device cpu --torch-threads 0   (0 = torch 默认全部线程)

批量: 1 / 128 / 1024 个参数样本, 每个样本查询整张 (n_z*n_t)=101*49=4949 时空网格
(与 B4/论文口径一致: "一次前向 = 一个完整时空场")。报告总时延与摊销单样本时延。

用法 (集群):
  python bench_inference.py \
      --config <绝对路径>/e9_bench_regshift_iid.yaml \
      --checkpoint $REPO/experiments/qtop_func/runs/g4_regshift_iid_s42/results/checkpoints/m1_best.pt \
      --run-dir   $REPO/experiments/qtop_func/runs/g4_regshift_iid_s42 \
      --device cuda --out bench_infer_gpu.json
"""
from __future__ import annotations

import argparse
import os
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("REPO", str(Path(__file__).resolve().parents[1])))


def parse_args():
    ap = argparse.ArgumentParser(description="E9: RegShift 推理计时 (集群)")
    ap.add_argument("--repo", default=str(REPO), help="主仓库根目录 (集群上按实际路径传)")
    ap.add_argument("--config", required=True, help="完整 YAML (model 段用于构建网络)")
    ap.add_argument("--checkpoint", required=True, help="RegShift checkpoint .pt")
    ap.add_argument("--run-dir", required=True,
                    help="含 data/processed/test.npz 的 run 目录 (取真实 branch/trunk 输入)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="CPU 口径的 torch 线程数; 0=torch 默认 (多线程), 1=单线程")
    ap.add_argument("--batch-sizes", default="1,128,1024")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", default="bench_infer.json")
    return ap.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo)
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    import yaml

    from src.models.shift_deeponet import build_shift_deeponet

    thread_note = "default"
    if args.device == "cpu" and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        thread_note = str(args.torch_threads)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    assert model_cfg.get("arch") == "shift_deeponet", "E9 计时对象是 RegShift/shift_deeponet"

    device = torch.device(args.device)
    model = build_shift_deeponet(model_cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    proc = Path(args.run_dir) / "data/processed/test.npz"
    test = np.load(proc)
    branch_np = test["branch_inputs"]  # (N, 57)
    trunk_np = test["trunk_inputs"]    # (M=4949, 2)
    branch_all = torch.from_numpy(branch_np).to(device)
    trunk = torch.from_numpy(trunk_np).to(device)
    n_avail = branch_all.shape[0]

    def make_batch(bs: int) -> torch.Tensor:
        reps = (bs + n_avail - 1) // n_avail
        return branch_all.repeat(reps, 1)[:bs].contiguous()

    def timed_forward(batch: torch.Tensor) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(batch, trunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    results = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(0)
                        if device.type == "cuda" else "cpu"),
        "torch_threads": (torch.get_num_threads()
                          if device.type == "cpu" else None),
        "torch_threads_note": thread_note,
        "n_trunk_points": int(trunk.shape[0]),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batches": {},
    }

    for bs in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        batch = make_batch(bs)
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = model(batch, trunk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = [timed_forward(batch) for _ in range(args.repeats)]
        med = statistics.median(times)
        results["batches"][str(bs)] = {
            "batch_size": bs,
            "total_median_s": med,
            "total_mean_s": statistics.fmean(times),
            "total_p10_s": sorted(times)[max(0, int(0.1 * len(times)) - 1)],
            "total_p90_s": sorted(times)[min(len(times) - 1, int(0.9 * len(times)))],
            "per_sample_median_ms": med / bs * 1000.0,
        }
        print(f"[{args.device}/threads={thread_note}] batch={bs}: "
              f"total={med*1000:.2f} ms, per-sample={med/bs*1000:.4f} ms")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"写出: {out}")


if __name__ == "__main__":
    main()
