#!/usr/bin/env python3
"""E2 preflight（完整版, 需 torch —— 在集群执行, 本机无 torch 会 ImportError）。

真正 import 主仓库 src/models/shift_deeponet.build_shift_deeponet,
用与 04_train_m1.py 完全相同的入口构建模型, 断言:
  1. model.transform_bound_scale == --expect-scale;
  2. model.transform_bound_shift == --expect-shift;
  3. 二者均 > 0（=0 表示无界, 说明 bound 键写错位置——历史 bug 的复现特征）;
  4. 前向抽样验证: 随机 branch 输入下, transform_net 产生的 scale/shift
     经 tanh 界限后落在 [1-σs, 1+σs] / [-σδ, σδ] 内。

用法（集群, cwd 任意）:
  REPO=/path/to/终极版论文项目_lx python preflight_torch.py \
      --config <abs>.yaml --expect-scale 0.1 --expect-shift 0.1
或 --repo 显式传主仓库根目录。
退出码: 0 通过, 1 失败。
"""

from pathlib import Path
import argparse
import os
import sys

DEFAULT_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo", default=os.environ.get("REPO", DEFAULT_REPO))
    ap.add_argument("--expect-scale", type=float, required=True)
    ap.add_argument("--expect-shift", type=float, required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)  # 使 `import src.models.shift_deeponet` 可用

    import yaml
    import torch  # 本机无 torch: 此脚本必须在集群跑
    from src.models.shift_deeponet import build_shift_deeponet

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_shift_deeponet(cfg["model"])  # 与 04_train_m1.py dispatch 同参
    errors = []
    tol = 1e-9
    if abs(model.transform_bound_scale - args.expect_scale) > tol:
        errors.append(f"model.transform_bound_scale={model.transform_bound_scale}, "
                      f"期望 {args.expect_scale}")
    if abs(model.transform_bound_shift - args.expect_shift) > tol:
        errors.append(f"model.transform_bound_shift={model.transform_bound_shift}, "
                      f"期望 {args.expect_shift}")
    if model.transform_bound_scale <= 0 or model.transform_bound_shift <= 0:
        errors.append("界限为 0（无界）——transform_bound_* 很可能写在顶层 "
                      "shift_deeponet: 块（历史 bug, REPO_FACTS.md §1）")

    if not errors:
        # 前向抽样: 界限是否真实生效
        D = cfg["model"].get("trunk_input_dim", 2)
        bdim = cfg["model"].get("branch_input_dim", 57)
        with torch.no_grad():
            x = torch.randn(256, bdim) * 3.0
            tp = model.transform_net(x)
            scale = 1.0 + model.transform_bound_scale * torch.tanh(tp[:, :D])
            shift = model.transform_bound_shift * torch.tanh(tp[:, D:])
        s_lo, s_hi = 1.0 - args.expect_scale, 1.0 + args.expect_scale
        if not (float(scale.min()) >= s_lo - 1e-6 and float(scale.max()) <= s_hi + 1e-6):
            errors.append(f"scale 越界: [{float(scale.min()):.4f}, {float(scale.max()):.4f}] "
                          f"⊄ [{s_lo}, {s_hi}]")
        if float(shift.abs().max()) > args.expect_shift + 1e-6:
            errors.append(f"|shift| 越界: max={float(shift.abs().max()):.4f} > {args.expect_shift}")

    if errors:
        print(f"[preflight-torch FAIL] {args.config}")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"[preflight-torch OK] {args.config} "
          f"scale={model.transform_bound_scale} shift={model.transform_bound_shift} "
          f"params={model.num_params}")


if __name__ == "__main__":
    main()
