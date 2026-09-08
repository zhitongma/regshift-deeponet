#!/usr/bin/env python3
"""Preflight (需 torch, 集群跑): 真实构建模型并断言有效界限值。

用法:
    python3 preflight_bounds_torch.py <config.yaml> --repo <主仓库路径> \
        --expect-scale 0.3 --expect-shift 0.2
    python3 preflight_bounds_torch.py <config.yaml> --repo ... --expect-unbounded

这是最终防线: 不管配置怎么写, 直接检查 build_shift_deeponet 构建出的
模型实例的 transform_bound_scale / transform_bound_shift 属性。
"""
import argparse
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--repo", required=True, help="主仓库根目录 (含 src/)")
    ap.add_argument("--expect-scale", type=float, default=None)
    ap.add_argument("--expect-shift", type=float, default=None)
    ap.add_argument("--expect-unbounded", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from src.models.shift_deeponet import build_shift_deeponet  # noqa: E402

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    if model_cfg.get("arch") != "shift_deeponet":
        print(f"[FAIL] model.arch={model_cfg.get('arch')!r}, 不会走 build_shift_deeponet")
        sys.exit(1)

    # 与 pipeline/04_train_m1.py:222 完全一致的构建路径
    model = build_shift_deeponet(model_cfg)
    got_s = float(model.transform_bound_scale)
    got_d = float(model.transform_bound_shift)
    print(f"有效界限: transform_bound_scale={got_s}, transform_bound_shift={got_d}")

    if args.expect_unbounded:
        ok = got_s == 0.0 and got_d == 0.0
    else:
        ok = (args.expect_scale is not None and abs(got_s - args.expect_scale) < 1e-9
              and args.expect_shift is not None and abs(got_d - args.expect_shift) < 1e-9)
    if not ok:
        print(f"[FAIL] 与期望不符 (expect scale={args.expect_scale}, "
              f"shift={args.expect_shift}, unbounded={args.expect_unbounded})")
        sys.exit(1)
    print("[OK] 模型实例的有效界限与期望一致")


if __name__ == "__main__":
    main()
