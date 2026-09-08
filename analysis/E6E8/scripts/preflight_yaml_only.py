#!/usr/bin/env python3
"""Shift 家族界限参数 preflight（降级版, 无 torch, 本地可跑）。

只做键层级检查（不真实构建模型）:
  1) model.arch == shift_deeponet；
  2) model.shift_deeponet.transform_bound_scale/shift 等于期望值
     （期望 0.0 时允许键缺省——build_shift_deeponet 默认 0.0 = 无界）；
  3) 顶层 shift_deeponet: 块不得含 transform_bound_*（历史 bug 模式, 当前
     代码路径不读取, 见 REPO_FACTS.md §1）。

优先用 PyYAML；无 PyYAML 时退化为面向本包配置风格的最小缩进解析器
（仅解析 key: value 层级, 足够检查上述键）。

用法:
  python preflight_yaml_only.py --config <yaml> --expect-scale 0.3 --expect-shift 0.2
退出码 0=通过, 1=失败。
"""

import argparse
import sys
from pathlib import Path


def parse_yaml_minimal(text: str) -> dict:
    """极简 YAML 子集解析: 缩进的 key: value / key: 嵌套映射。忽略列表与流式集合的内部结构。"""
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- ") or stripped == "-":
            continue  # 列表项: 本检查不需要
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key, val = key.strip(), val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if val.startswith("{"):
                parent[key] = val  # 流式映射按原文存储（本检查不需要其内部）
            else:
                try:
                    parent[key] = float(val)
                except ValueError:
                    parent[key] = val
    return root


def load_cfg(path: Path) -> dict:
    text = path.read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        print("[INFO] 无 PyYAML, 使用最小解析器（仅键层级检查）")
        return parse_yaml_minimal(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect-scale", type=float, required=True)
    ap.add_argument("--expect-shift", type=float, required=True)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    ok = True

    model = cfg.get("model", {}) or {}
    if model.get("arch") != "shift_deeponet":
        print(f"[FAIL] model.arch={model.get('arch')!r}, 期望 'shift_deeponet'")
        ok = False

    sd = model.get("shift_deeponet", {}) or {}
    if not isinstance(sd, dict):
        print("[FAIL] model.shift_deeponet 不是映射")
        sd, ok = {}, False
    for name, exp in (("transform_bound_scale", args.expect_scale),
                      ("transform_bound_shift", args.expect_shift)):
        got = sd.get(name)
        if got is None:
            if exp == 0.0:
                print(f"[OK] model.shift_deeponet.{name} 缺省 (= 默认 0.0, 无界)")
            else:
                print(f"[FAIL] 缺 model.shift_deeponet.{name} (期望 {exp}) —— "
                      f"若写在了顶层 shift_deeponet: 块则不会生效")
                ok = False
        else:
            if abs(float(got) - exp) > 1e-12:
                print(f"[FAIL] model.shift_deeponet.{name}={got}, 期望 {exp}")
                ok = False
            else:
                print(f"[OK] model.shift_deeponet.{name}={got}")

    top = cfg.get("shift_deeponet", {}) or {}
    if isinstance(top, dict) and any(
            k in top for k in ("transform_bound_scale", "transform_bound_shift")):
        print("[FAIL] 顶层 shift_deeponet: 块含 transform_bound_* 键（历史 bug 模式, 不会生效）")
        ok = False

    print("PREFLIGHT(YAML-ONLY) PASS" if ok else "PREFLIGHT(YAML-ONLY) FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
