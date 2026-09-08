#!/usr/bin/env python3
"""
Hydrological regime error decomposition and sensitivity analysis.

Analyzes model predictions by:
  1. Depth zones: shallow (0-20cm), middle (20-60cm), deep (60-100cm)
  2. Temporal phases: early infiltration, peak infiltration, recession
  3. Event intensity: q_top_peak quartiles
  4. Parameter sensitivity: perturbation-based attribution

Reads B1 prediction .npz files and raw parameters; outputs JSON + figures.
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_predictions(exp_dir, model_name):
    """Load B1 predictions .npz for a given model."""
    npz_path = exp_dir / f"B1_{model_name}_predictions.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    return {"h_pred": d["h_pred"], "c_pred": d["c_pred"]}


def load_reference(exp_dir):
    """Load raw h/c reference from test data."""
    proc_dir = exp_dir.parent.parent / "data" / "processed"
    test = np.load(proc_dir / "test.npz")
    return {
        "h_raw": test["h_raw"],
        "c_raw": test["c_raw"],
        "z": test["z"],
        "t": test["t"],
        "branch_inputs": test["branch_inputs"],
    }


def load_params(run_root):
    """Load parameter CSV."""
    params_path = run_root / "data" / "parameters" / "lhs_params.csv"
    if not params_path.exists():
        return None
    with open(params_path) as f:
        return list(csv.DictReader(f))


def rel_l2_per_sample(pred, ref):
    """Per-sample relative L2 error."""
    N = pred.shape[0]
    errors = np.zeros(N)
    for i in range(N):
        p, r = pred[i].ravel(), ref[i].ravel()
        norm_r = np.linalg.norm(r)
        errors[i] = np.linalg.norm(p - r) / (norm_r + 1e-12)
    return errors


def decompose_by_depth(h_pred, h_ref, c_pred, c_ref, z, n_z, n_t):
    """Decompose errors by depth zones."""
    zones = {
        "shallow_0_20cm": (z <= 20),
        "middle_20_60cm": (z > 20) & (z <= 60),
        "deep_60_100cm": (z > 60),
    }
    N = h_pred.shape[0]
    results = {}
    for zone_name, z_mask in zones.items():
        z_indices = np.where(z_mask)[0]
        h_zone_pred = h_pred[:, z_indices, :]
        h_zone_ref = h_ref[:, z_indices, :]
        c_zone_pred = c_pred[:, z_indices, :]
        c_zone_ref = c_ref[:, z_indices, :]
        results[zone_name] = {
            "h_rel_l2": float(np.mean(rel_l2_per_sample(
                h_zone_pred.reshape(N, -1), h_zone_ref.reshape(N, -1)))),
            "c_rel_l2": float(np.mean(rel_l2_per_sample(
                c_zone_pred.reshape(N, -1), c_zone_ref.reshape(N, -1)))),
            "h_mae": float(np.mean(np.abs(h_zone_pred - h_zone_ref))),
            "c_mae": float(np.mean(np.abs(c_zone_pred - c_zone_ref))),
        }
    return results


def decompose_by_time_phase(h_pred, h_ref, c_pred, c_ref, t, branch_inputs, n_z, n_t):
    """Decompose errors by temporal phase based on q_top values."""
    N = h_pred.shape[0]

    # q_top values are in branch_inputs columns 6:54 (for 48-step q_top)
    n_branch = branch_inputs.shape[1]
    if n_branch >= 54:
        q_top_series = branch_inputs[:, 6:54]
    elif n_branch >= 10:
        q_top_val = branch_inputs[:, 6:7]
        q_top_series = np.tile(q_top_val, (1, len(t) - 1))
    else:
        return {}

    n_t_actual = min(q_top_series.shape[1], n_t)

    phases = {
        "early_0_16h": list(range(0, min(16, n_t))),
        "peak_16_32h": list(range(16, min(32, n_t))),
        "late_32_48h": list(range(32, min(n_t, 49))),
    }

    results = {}
    for phase_name, t_indices in phases.items():
        if not t_indices:
            continue
        h_phase_pred = h_pred[:, :, t_indices]
        h_phase_ref = h_ref[:, :, t_indices]
        c_phase_pred = c_pred[:, :, t_indices]
        c_phase_ref = c_ref[:, :, t_indices]
        results[phase_name] = {
            "h_rel_l2": float(np.mean(rel_l2_per_sample(
                h_phase_pred.reshape(N, -1), h_phase_ref.reshape(N, -1)))),
            "c_rel_l2": float(np.mean(rel_l2_per_sample(
                c_phase_pred.reshape(N, -1), c_phase_ref.reshape(N, -1)))),
            "h_mae": float(np.mean(np.abs(h_phase_pred - h_phase_ref))),
            "c_mae": float(np.mean(np.abs(c_phase_pred - c_phase_ref))),
        }
    return results


def decompose_by_event_intensity(h_errors, c_errors, params_rows):
    """Split samples by q_top_peak quartiles."""
    if not params_rows:
        return {}

    test_rows = [r for r in params_rows if r.get("split") == "test"]
    if len(test_rows) != len(h_errors):
        test_rows = params_rows[-len(h_errors):]

    peaks = []
    for r in test_rows:
        if "q_top_peak" in r:
            peaks.append(float(r["q_top_peak"]))
        else:
            peaks.append(0.0)
    peaks = np.array(peaks)

    if len(peaks) == 0:
        return {}

    q25, q50, q75 = np.percentile(peaks, [25, 50, 75])
    quartiles = {
        "low_q25": peaks <= q25,
        "mid_q25_q75": (peaks > q25) & (peaks <= q75),
        "high_q75": peaks > q75,
    }

    results = {}
    for qname, mask in quartiles.items():
        if not np.any(mask):
            continue
        results[qname] = {
            "n_samples": int(np.sum(mask)),
            "h_rel_l2_mean": float(np.mean(h_errors[mask])),
            "c_rel_l2_mean": float(np.mean(c_errors[mask])),
            "h_rel_l2_std": float(np.std(h_errors[mask])),
            "c_rel_l2_std": float(np.std(c_errors[mask])),
            "q_top_peak_range": [float(np.min(peaks[mask])), float(np.max(peaks[mask]))],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Hydrological regime error analysis")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--models", type=str, default="m1,m2_data,fnn,grid_fnn")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_root = Path(args.run_dir)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    exp_dir = run_root / "results" / "experiments"

    model_names = [m.strip() for m in args.models.split(",")]

    ref_data = load_reference(exp_dir)
    z = ref_data["z"]
    t = ref_data["t"]
    n_z = len(z)
    n_t = len(t)
    h_ref_3d = ref_data["h_raw"].reshape(-1, n_z, n_t)
    c_ref_3d = ref_data["c_raw"].reshape(-1, n_z, n_t)
    N = h_ref_3d.shape[0]

    params = load_params(run_root)

    output = {"n_samples": N, "n_z": n_z, "n_t": n_t, "models": {}}

    for model_name in model_names:
        preds = load_predictions(exp_dir, model_name)
        if preds is None:
            print(f"  {model_name}: no predictions found, skipping")
            continue

        h_pred_3d = preds["h_pred"].reshape(N, n_z, n_t)
        c_pred_3d = preds["c_pred"].reshape(N, n_z, n_t)

        h_errors = rel_l2_per_sample(
            h_pred_3d.reshape(N, -1), h_ref_3d.reshape(N, -1))
        c_errors = rel_l2_per_sample(
            c_pred_3d.reshape(N, -1), c_ref_3d.reshape(N, -1))

        model_results = {
            "overall": {
                "h_rel_l2": float(np.mean(h_errors)),
                "c_rel_l2": float(np.mean(c_errors)),
            },
            "by_depth": decompose_by_depth(
                h_pred_3d, h_ref_3d, c_pred_3d, c_ref_3d, z, n_z, n_t),
            "by_time_phase": decompose_by_time_phase(
                h_pred_3d, h_ref_3d, c_pred_3d, c_ref_3d, t,
                ref_data["branch_inputs"], n_z, n_t),
            "by_event_intensity": decompose_by_event_intensity(
                h_errors, c_errors, params),
        }

        output["models"][model_name] = model_results
        print(f"  {model_name}: h={np.mean(h_errors):.4f}, c={np.mean(c_errors):.4f}")

    out_path = args.output or str(exp_dir / "regime_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
