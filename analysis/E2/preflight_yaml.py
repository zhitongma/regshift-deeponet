#!/usr/bin/env python3
"""E2 preflight（降级版, 纯 YAML 键层级校验, 无 torch, 本地可跑）。

校验目标（对应 REPO_FACTS.md §1 的历史 bug）:
  1. model.arch == "shift_deeponet"；
  2. RegShift 界限参数必须位于 model.shift_deeponet.transform_bound_scale /
     transform_bound_shift, 且等于期望值；
  3. 顶层不得出现携带 transform_bound_* 的 shift_deeponet: 块
     （该位置的键 build_shift_deeponet 不会读取, 属静默失效）。

用法:
  python3 preflight_yaml.py --config <abs>.yaml --expect-scale 0.1 --expect-shift 0.1

依赖: 仅标准库 + PyYAML（本机已装 6.x）; 若无 PyYAML 会退化为基于缩进的
文本扫描（只做键定位, 不解析完整语义）。
退出码: 0 通过, 1 失败。
"""

import argparse
import re
import sys


def load_cfg_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cfg_fallback(path):
    """无 PyYAML 时的降级解析: 仅提取本脚本要检查的键。"""
    cfg = {"model": {}}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    stack = []  # (indent, key)
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip())
        m = re.match(r"^\s*([A-Za-z_][\w]*)\s*:\s*(.*)$", stripped)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([k for _, k in stack] + [key])
        stack.append((indent, key))
        if dotted == "model.arch":
            cfg["model"]["arch"] = val
        elif dotted == "model.shift_deeponet.transform_bound_scale":
            cfg["model"].setdefault("shift_deeponet", {})[
                "transform_bound_scale"] = float(val)
        elif dotted == "model.shift_deeponet.transform_bound_shift":
            cfg["model"].setdefault("shift_deeponet", {})[
                "transform_bound_shift"] = float(val)
        elif dotted in ("shift_deeponet.transform_bound_scale",
                        "shift_deeponet.transform_bound_shift"):
            cfg.setdefault("shift_deeponet", {})[key] = val
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect-scale", type=float, required=True)
    ap.add_argument("--expect-shift", type=float, required=True)
    args = ap.parse_args()

    try:
        cfg = load_cfg_yaml(args.config)
        mode = "pyyaml"
    except ImportError:
        cfg = load_cfg_fallback(args.config)
        mode = "fallback(indent-scan)"

    errors = []
    model = cfg.get("model") or {}
    if model.get("arch") != "shift_deeponet":
        errors.append(f"model.arch != shift_deeponet (got: {model.get('arch')!r})")

    sd = model.get("shift_deeponet") or {}
    tol = 1e-9
    for key, expect in (("transform_bound_scale", args.expect_scale),
                        ("transform_bound_shift", args.expect_shift)):
        got = sd.get(key)
        if got is None:
            errors.append(f"缺少 model.shift_deeponet.{key}（历史 bug 位置校验失败）")
        elif abs(float(got) - expect) > tol:
            errors.append(f"model.shift_deeponet.{key}={got}, 期望 {expect}")

    top_sd = cfg.get("shift_deeponet")
    if isinstance(top_sd, dict) and (
            "transform_bound_scale" in top_sd or "transform_bound_shift" in top_sd):
        errors.append("检测到顶层 shift_deeponet: 块携带 transform_bound_*"
                      "——该位置不会被 build_shift_deeponet 读取（历史 bug, REPO_FACTS.md §1）")

    if errors:
        print(f"[preflight-yaml FAIL] {args.config} (parser={mode})")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"[preflight-yaml OK] {args.config} (parser={mode}) "
          f"scale={args.expect_scale} shift={args.expect_shift}")


if __name__ == "__main__":
    main()
