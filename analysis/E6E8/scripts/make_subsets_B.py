#!/usr/bin/env python3
"""E6-B/E8: 把每个评估 run 的 test 行按 severity_group 分组，生成与
regrade_tails.py 同 schema 的 subsets JSON，供 eval_subsets.py 复用。

纯文本输入（lhs_params.csv + metadata/test_row_map.json），numpy-only。
在评估 run 已由 process_extreme.py 建好、06_evaluate 已跑完后执行（集群，
或结果同步回本地且已物化后本地跑）。

用法:
  python make_subsets_B.py --manifest <包>/models_manifest.json \
      --eval-root $REPO/experiments/qtop_func/runs_revision \
      --out <包>/outputs/B_subsets.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def read_groups(params_csv: Path) -> dict:
    groups = {}
    with open(params_csv, newline="") as f:
        for row in csv.DictReader(f):
            groups[int(row["sample_id"])] = row["severity_group"]
    return groups


def main():
    pkg = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(pkg / "models_manifest.json"))
    ap.add_argument("--eval-root", default=None,
                    help="默认 <repo>/experiments/qtop_func/runs_revision")
    ap.add_argument("--out", default=str(pkg / "outputs" / "B_subsets.json"))
    ap.add_argument("--include-optional", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    repo = Path(manifest["repo"])
    eval_root = Path(args.eval_root) if args.eval_root else repo / manifest["runs_revision_root"]

    runs_out = {}
    for src in manifest["part_B_model_sources"]:
        if src.get("optional") and not args.include_optional:
            continue
        for st in manifest["part_B_sets"]:
            eval_key = f"e6e8_eval__{st['key']}__{src['key']}"
            eval_dir = eval_root / eval_key
            row_map_path = eval_dir / "metadata" / "test_row_map.json"
            if not row_map_path.exists():
                print(f"[WARN] 跳过 {eval_key}: 缺 {row_map_path}")
                continue
            with open(row_map_path) as f:
                row_map = json.load(f)
            sample_ids = [int(x) for x in row_map["sample_ids"]]
            params_csv = eval_dir / "data" / "parameters" / "lhs_params.csv"
            if not params_csv.exists():
                params_csv = Path(row_map["params_csv"])
            groups_of = read_groups(params_csv)

            bins = {}
            for r, sid in enumerate(sample_ids):
                g = groups_of.get(sid, "UNKNOWN")
                bins.setdefault(g, {"sample_ids": [], "row_indices": []})
                bins[g]["sample_ids"].append(sid)
                bins[g]["row_indices"].append(r)

            runs_out[eval_key] = {
                "key": eval_key,
                "scenario": src["scenario"],
                "family": src["family"],
                "model_source_key": src["key"],
                "set_key": st["key"],
                "primary_axis": st["axis"],
                "run_dir": str(eval_dir),
                "models": "auto",  # eval_subsets.py 会 glob B1_*_predictions.npz
                "n_test_rows": len(sample_ids),
                "row_sample_ids": sample_ids,
                "axes": {st["axis"]: {"thresholds": {}, "bins": bins}},
            }
            counts = ", ".join(f"{g}:{len(b['sample_ids'])}" for g, b in bins.items())
            print(f"[OK] {eval_key}: n={len(sample_ids)}, groups={{{counts}}}")

    out = {"kind": "E6E8_B_severity_subsets", "runs": runs_out}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n输出: {out_path}  ({len(runs_out)} 个评估 run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
