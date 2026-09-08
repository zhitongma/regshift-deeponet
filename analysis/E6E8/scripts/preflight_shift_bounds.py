#!/usr/bin/env python3
"""Shift 家族界限参数 preflight（完整版, 需 torch —— 在集群执行）。

加载 YAML -> 调 src/models/shift_deeponet.build_shift_deeponet(cfg["model"])
真实构建模型 -> 断言 model.transform_bound_scale / transform_bound_shift
等于期望值。用于捕获"界限写在顶层 shift_deeponet: 块而未生效"的历史 bug
（REPO_FACTS.md §1 / E0）。

无界（Shift-DeepONet）期望 0.0/0.0；RegShift 期望 0.3/0.2。

用法:
  python preflight_shift_bounds.py --repo $REPO \
      --config <包>/configs/eval_regshift_bounded.yaml \
      --expect-scale 0.3 --expect-shift 0.2
退出码 0=通过, 1=失败。
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect-scale", type=float, required=True)
    ap.add_argument("--expect-shift", type=float, required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import yaml
    from src.models.shift_deeponet import build_shift_deeponet  # 该模块 import torch

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    arch = cfg.get("model", {}).get("arch")
    if arch != "shift_deeponet":
        print(f"[FAIL] model.arch={arch!r}, 期望 'shift_deeponet'")
        return 1

    model = build_shift_deeponet(cfg["model"])
    ok = True
    for name, got, exp in (
        ("transform_bound_scale", float(model.transform_bound_scale), args.expect_scale),
        ("transform_bound_shift", float(model.transform_bound_shift), args.expect_shift),
    ):
        if abs(got - exp) > 1e-12:
            print(f"[FAIL] 构建后的模型 {name}={got}, 期望 {exp} —— "
                  f"检查键是否写在 model.shift_deeponet.{name}（而非顶层 shift_deeponet:）")
            ok = False
        else:
            print(f"[OK] {name}={got}")

    # 附加告警: 顶层 shift_deeponet: 块含 bound 键是历史 bug 的特征
    top = cfg.get("shift_deeponet", {}) or {}
    if any(k in top for k in ("transform_bound_scale", "transform_bound_shift")):
        print("[FAIL] 顶层 shift_deeponet: 块含 transform_bound_* 键（当前代码不读取, 历史 bug 模式）")
        ok = False

    print("PREFLIGHT PASS" if ok else "PREFLIGHT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
