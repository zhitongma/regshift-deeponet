#!/usr/bin/env python3
"""E1 preflight (FULL version -- requires torch, run on the CLUSTER).

Builds the actual model via src.models.shift_deeponet.build_shift_deeponet and
asserts that the RegShift bound parameters really landed on the model instance.
This guards against the historical bug where bounds written at the TOP-LEVEL
`shift_deeponet:` block were silently ignored (build_shift_deeponet only reads
cfg["model"]["shift_deeponet"], see src/models/shift_deeponet.py:155-172 and
REPO_FACTS.md §1 / package E0).

Usage (cluster; --repo must point to the cluster's copy of the main repo):
  python preflight_bounds.py --config configs/e1_regshift_lowbudget_iid.yaml \
      --expect-scale 0.3 --expect-shift 0.2 [--repo /path/to/终极版论文项目_lx]
  python preflight_bounds.py --config configs/e1_shift_highbudget_iid.yaml \
      --expect-scale 0.0 --expect-shift 0.0

Exit code 0 = pass, 1 = fail. No torch on the local Mac -> use
preflight_yaml_only.py locally instead.
"""
import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect-scale", type=float, required=True)
    ap.add_argument("--expect-shift", type=float, required=True)
    ap.add_argument("--repo", default=os.environ.get("REPO", DEFAULT_REPO),
                    help="Main repo root (cluster path may differ from the Mac path)")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    repo = Path(args.repo).resolve()
    if not (repo / "src" / "models" / "shift_deeponet.py").exists():
        print(f"[preflight] FAIL: {repo} does not look like the main repo "
              "(src/models/shift_deeponet.py missing); pass --repo")
        return 1
    sys.path.insert(0, str(repo))

    # NOTE: importing this module imports torch -> cluster only.
    from src.models.shift_deeponet import build_shift_deeponet

    arch = cfg.get("model", {}).get("arch")
    if arch != "shift_deeponet":
        print(f"[preflight] FAIL: model.arch={arch!r}, expected 'shift_deeponet'")
        return 1

    model = build_shift_deeponet(cfg["model"])
    ok = True
    for attr, expect in (("transform_bound_scale", args.expect_scale),
                         ("transform_bound_shift", args.expect_shift)):
        got = float(getattr(model, attr))
        status = "OK" if abs(got - expect) < 1e-9 else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"[preflight] {attr}: built={got} expected={expect} -> {status}")

    if not ok:
        print("[preflight] FAIL: bounds did not reach the model. Check that they "
              "are under model.shift_deeponet.* (NOT top-level shift_deeponet:).")
        return 1
    print(f"[preflight] PASS: {Path(args.config).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
