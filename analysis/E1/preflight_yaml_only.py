#!/usr/bin/env python3
"""E1 preflight (DEGRADED version -- no torch, runs locally on the Mac).

Only checks the YAML key hierarchy: RegShift bounds must live at
model.shift_deeponet.transform_bound_scale / transform_bound_shift, and a
top-level `shift_deeponet:` block must NOT carry bound keys (historical bug:
top-level bounds are silently ignored by build_shift_deeponet, see
REPO_FACTS.md §1). It does NOT build the model; the full check is
preflight_bounds.py (cluster, torch).

Usage:
  python3 preflight_yaml_only.py --config configs/e1_regshift_lowbudget_iid.yaml \
      --expect-scale 0.3 --expect-shift 0.2
  python3 preflight_yaml_only.py --all   # check every configs/e1_*.yaml with
                                         # expectations inferred from filename
"""
import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent

BUDGETS = {
    # filename tag -> (lr, epochs, early_stopping_patience, lr_patience)
    "lowbudget": (1.0e-3, 2000, 800, 400),
    "highbudget": (5.0e-4, 5000, 2000, 800),
}


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def check(path, expect_scale, expect_shift):
    cfg = load_yaml(path)
    name = Path(path).name
    errors = []

    model = cfg.get("model", {})
    sd = model.get("shift_deeponet", {}) or {}
    got_scale = float(sd.get("transform_bound_scale", 0.0))
    got_shift = float(sd.get("transform_bound_shift", 0.0))
    if abs(got_scale - expect_scale) > 1e-12:
        errors.append(f"model.shift_deeponet.transform_bound_scale={got_scale}, expected {expect_scale}")
    if abs(got_shift - expect_shift) > 1e-12:
        errors.append(f"model.shift_deeponet.transform_bound_shift={got_shift}, expected {expect_shift}")

    top_sd = cfg.get("shift_deeponet")
    if isinstance(top_sd, dict) and (
            "transform_bound_scale" in top_sd or "transform_bound_shift" in top_sd):
        errors.append("top-level shift_deeponet: carries bound keys -- these are "
                      "SILENTLY IGNORED by build_shift_deeponet (historical bug)")

    if expect_scale > 0 or expect_shift > 0:
        if model.get("arch") != "shift_deeponet":
            errors.append(f"model.arch={model.get('arch')!r}, expected 'shift_deeponet'")

    tr = cfg.get("training", {})
    if tr.get("n_coords_train") != 1024:
        errors.append(f"training.n_coords_train={tr.get('n_coords_train')}, "
                      "expected 1024 (E1 fairness control)")
    if tr.get("seed") != 42:
        errors.append(f"training.seed={tr.get('seed')}, expected 42 (real seed via --seed)")

    for tag, (lr, ep, es, lp) in BUDGETS.items():
        if tag in name:
            checks = [("learning_rate", lr), ("epochs", ep),
                      ("early_stopping_patience", es), ("lr_patience", lp)]
            for key, want in checks:
                if abs(float(tr.get(key, -1)) - want) > 1e-12:
                    errors.append(f"training.{key}={tr.get(key)}, expected {want} ({tag})")

    if errors:
        print(f"[yaml-preflight] FAIL {name}")
        for e in errors:
            print(f"    - {e}")
        return False
    print(f"[yaml-preflight] PASS {name} "
          f"(scale={got_scale}, shift={got_shift}, n_coords_train=1024)")
    return True


def infer_expectations(name):
    if "regshift" in name:
        return 0.3, 0.2
    return 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--expect-scale", type=float)
    ap.add_argument("--expect-shift", type=float)
    ap.add_argument("--all", action="store_true",
                    help="check every configs/e1_*.yaml (expectations from filename)")
    args = ap.parse_args()

    if args.all:
        ok = True
        files = sorted((PKG.parents[1] / "configs/revision/E1_budget").glob("e1_*.yaml"))
        if not files:
            print("[yaml-preflight] no configs/e1_*.yaml found")
            return 1
        for f in files:
            es, eh = infer_expectations(f.name)
            ok = check(f, es, eh) and ok
        return 0 if ok else 1

    if not args.config:
        ap.error("--config required unless --all")
    es = args.expect_scale
    eh = args.expect_shift
    if es is None or eh is None:
        es, eh = infer_expectations(Path(args.config).name)
    return 0 if check(args.config, es, eh) else 1


if __name__ == "__main__":
    sys.exit(main())
