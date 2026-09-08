#!/usr/bin/env python3
"""E7 preflight (无 torch, 本地可跑) — 两项 YAML 层级检查。

1. 包内 configs/e7_mc_regshift_iid.yaml: RegShift 界限键必须在
   model.shift_deeponet.transform_bound_scale/shift = 0.3/0.2
   (REPO_FACTS §1; build_shift_deeponet 只读该层级,
    src/models/shift_deeponet.py:154-172);
2. checkpoint 源 run 的 metadata/config.yaml: 报告其"训练时有效界限"。
   历史 g4_regshift_* 快照把 bound 写在顶层 shift_deeponet: 块 (E0 历史 bug),
   训练时不生效 → checkpoint 实为无界。本检查不判失败, 只输出结论, 并给出
   mc_screening.py 应使用的 --expect-* 参数, 供 run.sh 传参。

torch 级最终防线: 集群上由 mc_screening.py 内置断言完成 (真实 build 后比对
model.transform_bound_*), 亦可直接用 E0 的 preflight_bounds_torch.py。

用法:
    python3 preflight_e7.py --pkg-config ../configs/e7_mc_regshift_iid.yaml \
        --ckpt-run-config <ckpt_run>/metadata/config.yaml
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def _naive_yaml_sections(path):
    """极简层级 YAML 解析 (与 E0/tools/preflight_bounds_yaml.py 同实现)。"""
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


def load_cfg(path):
    if yaml is not None:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    print("[warn] 未安装 pyyaml, 使用朴素解析器 (仅键层级判断)")
    return _naive_yaml_sections(path)


def effective_bounds(cfg):
    """按 build_shift_deeponet 的真实读取路径推有效界限 (shift_deeponet.py:154-172)。"""
    msd = (cfg.get("model") or {}).get("shift_deeponet") or {}
    return (float(msd.get("transform_bound_scale", 0.0) or 0.0),
            float(msd.get("transform_bound_shift", 0.0) or 0.0))


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--pkg-config", default=str(here.parents[2] / "configs/revision/E7_screen/e7_mc_regshift_iid.yaml"))
    ap.add_argument("--ckpt-run-config", default=None,
                    help="checkpoint 源 run 的 metadata/config.yaml (可选)")
    ap.add_argument("--expect-scale", type=float, default=0.3)
    ap.add_argument("--expect-shift", type=float, default=0.2)
    args = ap.parse_args()

    failed = False

    # ---- 检查 1: 包内配置 ----
    cfg = load_cfg(args.pkg_config)
    model = cfg.get("model") or {}
    top_sd = cfg.get("shift_deeponet") or {}
    if any(str(k).startswith("transform_bound") for k in top_sd):
        print(f"[FAIL] {args.pkg_config}: 顶层 shift_deeponet: 块含 transform_bound_* "
              "(不会被 build_shift_deeponet 读取, E0 历史 bug)")
        failed = True
    scale, shift = effective_bounds(cfg)
    if model.get("arch") != "shift_deeponet":
        print(f"[FAIL] {args.pkg_config}: model.arch={model.get('arch')!r} != shift_deeponet")
        failed = True
    if abs(scale - args.expect_scale) > 1e-9 or abs(shift - args.expect_shift) > 1e-9:
        print(f"[FAIL] {args.pkg_config}: 有效界限 scale={scale}, shift={shift} "
              f"!= 期望 {args.expect_scale}/{args.expect_shift}")
        failed = True
    if not failed:
        print(f"[OK] {args.pkg_config}: model.shift_deeponet.transform_bound_scale={scale}, "
              f"transform_bound_shift={shift}")

    # ---- 检查 2: checkpoint 源 run 快照 (只报告, 不判失败) ----
    if args.ckpt_run_config:
        ck = load_cfg(args.ckpt_run_config)
        ck_scale, ck_shift = effective_bounds(ck)
        ck_top = ck.get("shift_deeponet") or {}
        buggy = any(str(k).startswith("transform_bound") for k in ck_top)
        print(f"[info] {args.ckpt_run_config}: 训练时有效界限 "
              f"scale={ck_scale}, shift={ck_shift}"
              + (" (顶层 shift_deeponet: 含 bound 键但不生效 → checkpoint 实为无界, "
                 "E0 历史 bug)" if buggy and ck_scale == 0.0 else ""))
        if ck_scale == 0.0 and ck_shift == 0.0:
            print("[info] mc_screening.py 应传 --expect-unbounded (推理构建与训练自洽)")
        else:
            print(f"[info] mc_screening.py 应传 --expect-scale {ck_scale} "
                  f"--expect-shift {ck_shift}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
