#!/usr/bin/env python3
"""Aggregate existing E0/E1 training manifests; does not train or evaluate models."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


GROUPS = {
    "e1_deeponet_highbudget_": "Standard DeepONet, high",
    "e1_shift_highbudget_": "Unbounded Shift, high",
    "e0_regshift_fixed_": "Bounded RegShift, high",
    "e1_regshift_lowbudget_": "Bounded RegShift, low",
    "e1_fnn_highbudget_": "FNN, high",
}


def group_for(name: str) -> str | None:
    return next((label for prefix, label in GROUPS.items() if name.startswith(prefix)), None)


def scenario_for(name: str) -> str:
    stem = re.sub(r"_s(?:42|123|456)$", "", name)
    if stem.endswith("ood_peak_time"):
        return "OOD peak-time"
    if stem.endswith("ood_peak"):
        return "OOD peak"
    if stem.endswith("iid"):
        return "IID"
    raise ValueError(name)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_revision")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    root = Path(args.runs_revision)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    detailed: list[dict[str, object]] = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        group = group_for(run.name)
        if group is None:
            continue
        meta = run / "metadata"
        fnn = meta / "train_fnn_manifest.json"
        if fnn.exists():
            d = load(fnn)
            detailed.append({
                "group": group,
                "scenario": scenario_for(run.name),
                "run": run.name,
                "total_epochs": int(d["total_epochs"]),
                "best_epoch": int(d["best_epoch"]),
                "best_val_loss": float(d["best_val_loss"]),
                "wall_minutes": float(d["training_time_seconds"]) / 60.0,
            })
            continue

        p1, p2 = meta / "train_m1_manifest.json", meta / "train_m2_manifest.json"
        if not (p1.exists() and p2.exists()):
            continue
        m1, m2 = load(p1), load(p2)
        detailed.append({
            "group": group,
            "scenario": scenario_for(run.name),
            "run": run.name,
            "total_epochs": int(m1["total_epochs"]) + int(m2["total_epochs"]),
            "best_epoch": int(m2.get("best_data_epoch", m2.get("best_epoch", 0))),
            "best_val_loss": float(m2["best_val_loss"]),
            "wall_minutes": (float(m1["training_time_seconds"]) + float(m2["training_time_seconds"])) / 60.0,
        })

    fields = ["group", "scenario", "run", "total_epochs", "best_epoch", "best_val_loss", "wall_minutes"]
    with (out / "training_manifest_detailed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detailed)

    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detailed:
        buckets[(str(row["group"]), str(row["scenario"]))].append(row)
    summary: list[dict[str, object]] = []
    for (group, scenario), rows in sorted(buckets.items()):
        epochs = [float(r["total_epochs"]) for r in rows]
        best = [float(r["best_epoch"]) for r in rows]
        losses = [float(r["best_val_loss"]) for r in rows]
        wall = [float(r["wall_minutes"]) for r in rows]
        summary.append({
            "group": group,
            "scenario": scenario,
            "n": len(rows),
            "total_epochs_mean": mean(epochs),
            "total_epochs_sd": sd(epochs),
            "stage2_best_data_epoch_mean": mean(best),
            "stage2_best_data_epoch_sd": sd(best),
            "best_val_loss_mean": mean(losses),
            "best_val_loss_sd": sd(losses),
            "wall_minutes_mean": mean(wall),
            "wall_minutes_sd": sd(wall),
        })
    summary_fields = list(summary[0]) if summary else []
    with (out / "training_manifest_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    print(f"runs={len(detailed)} cells={len(summary)} out={out}")


if __name__ == "__main__":
    main()
