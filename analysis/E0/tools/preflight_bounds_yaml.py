#!/usr/bin/env python3
"""Preflight (无 torch, 本地可跑): 校验 RegShift 界限参数的 YAML 键层级。

用法:
    python3 preflight_bounds_yaml.py <config.yaml> [--expect-scale 0.3] [--expect-shift 0.2]
    python3 preflight_bounds_yaml.py <config.yaml> --expect-unbounded   # 无界基线用

规则 (见 E0/README.md):
  - build_shift_deeponet 只读 model.shift_deeponet.transform_bound_*;
  - 顶层 shift_deeponet: 块中的 bound 键不会生效 (历史 bug), 一经发现即报错。
"""
import argparse
import sys

try:
    import yaml
except ImportError:  # 无 pyyaml 时退化为朴素解析器
    yaml = None


def _naive_yaml_sections(path):
    """极简两级 YAML 解析 (仅用于本 preflight 的键层级判断)。"""
    top = {}
    stack = [(0, top)]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if ":" not in line:
                continue
            key, _, val = line.strip().partition(":")
            val = val.split("#")[0].strip()
            while stack and indent < stack[-1][0]:
                stack.pop()
            if not stack:
                stack = [(0, top)]
            parent = stack[-1][1]
            if val == "":
                child = {}
                parent[key] = child
                stack.append((indent + 2, child))
            else:
                try:
                    parent[key] = float(val)
                except ValueError:
                    parent[key] = val
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--expect-scale", type=float, default=None)
    ap.add_argument("--expect-shift", type=float, default=None)
    ap.add_argument("--expect-unbounded", action="store_true")
    args = ap.parse_args()

    if yaml is not None:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        print("[warn] 未安装 pyyaml, 使用朴素解析器 (仅键层级判断)")
        cfg = _naive_yaml_sections(args.config)

    errors, warnings = [], []

    top_sd = cfg.get("shift_deeponet") or {}
    if any(k.startswith("transform_bound") for k in top_sd):
        errors.append(
            "顶层 shift_deeponet: 块含 transform_bound_* —— 该层级不会被 "
            "build_shift_deeponet 读取 (历史 bug, 见 E0/README.md §二), "
            "请移到 model.shift_deeponet.* 下")

    model = cfg.get("model") or {}
    msd = model.get("shift_deeponet") or {}
    scale = msd.get("transform_bound_scale")
    shift = msd.get("transform_bound_shift")
    arch = model.get("arch")

    if args.expect_unbounded:
        if scale not in (None, 0, 0.0) or shift not in (None, 0, 0.0):
            errors.append(
                f"期望无界基线, 但 model.shift_deeponet 含 bound 键: "
                f"scale={scale}, shift={shift}")
    else:
        if args.expect_scale is not None:
            if scale is None:
                errors.append(
                    "model.shift_deeponet.transform_bound_scale 缺失 "
                    f"(期望 {args.expect_scale}) —— 将静默训练无界模型!")
            elif abs(float(scale) - args.expect_scale) > 1e-9:
                errors.append(
                    f"transform_bound_scale={scale} != 期望 {args.expect_scale}")
        if args.expect_shift is not None:
            if shift is None:
                errors.append(
                    "model.shift_deeponet.transform_bound_shift 缺失 "
                    f"(期望 {args.expect_shift}) —— 将静默训练无界模型!")
            elif abs(float(shift) - args.expect_shift) > 1e-9:
                errors.append(
                    f"transform_bound_shift={shift} != 期望 {args.expect_shift}")
        if arch != "shift_deeponet":
            warnings.append(f"model.arch={arch!r} 不是 shift_deeponet, bound 键不会被用到")

    for w in warnings:
        print(f"[warn] {w}")
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        sys.exit(1)
    print(f"[OK] {args.config}: bound 键层级与取值符合预期 "
          f"(scale={scale}, shift={shift}, arch={arch})")


if __name__ == "__main__":
    main()
