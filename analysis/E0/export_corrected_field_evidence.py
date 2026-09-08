#!/usr/bin/env python3
"""Export compact, audited field evidence from corrected E0/E1 runs.

The exporter is intended to run where the large prediction archives live. It
does not retrain or reevaluate a model. It verifies model identity, test-set
alignment, finite arrays, and agreement between field arrays and archived B1
per-sample errors before exporting only the data needed for the corrected
process, representative-case, and regime figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


SCENARIOS = ("iid", "ood_peak", "ood_peak_time")
SEEDS = (42, 123, 456)
MODELS = ("regshift", "shift", "fnn")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value)
    return hashlib.sha256(data.view(np.uint8)).hexdigest()


def rel_l2_per_sample(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred_flat = pred.reshape(pred.shape[0], -1).astype(np.float64)
    ref_flat = ref.reshape(ref.shape[0], -1).astype(np.float64)
    return np.linalg.norm(pred_flat - ref_flat, axis=1) / (
        np.linalg.norm(ref_flat, axis=1) + 1.0e-8
    )


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    return ranks


def run_name(model: str, scenario: str, seed: int) -> str:
    if model == "regshift":
        return f"e0_regshift_fixed_{scenario}_s{seed}"
    if model == "shift":
        return f"e1_shift_highbudget_{scenario}_s{seed}"
    if model == "fnn":
        return f"e1_fnn_highbudget_{scenario}_s{seed}"
    raise KeyError(model)


def prediction_name(model: str) -> str:
    return "B1_fnn_predictions.npz" if model == "fnn" else "B1_m2_data_predictions.npz"


def result_key(model: str) -> str:
    return "fnn" if model == "fnn" else "m2_data"


def load_snapshot(run_dir: Path) -> dict:
    path = run_dir / "metadata" / "train_m2_config_snapshot.yaml"
    if not path.exists():
        path = run_dir / "metadata" / "train_fnn_config_snapshot.yaml"
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return document.get("config", document)


def verify_model_identity(model: str, config: dict, source: Path) -> dict:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch")
    nested = model_cfg.get("shift_deeponet") or {}
    top_level = config.get("shift_deeponet") or {}
    if any(key.startswith("transform_bound") for key in top_level):
        raise AssertionError(f"Ignored top-level bound key found in {source}")
    scale = float(nested.get("transform_bound_scale", 0.0) or 0.0)
    shift = float(nested.get("transform_bound_shift", 0.0) or 0.0)
    if model == "regshift":
        if arch != "shift_deeponet" or (scale, shift) != (0.3, 0.2):
            raise AssertionError(
                f"Corrected bounded identity failed for {source}: "
                f"arch={arch}, bounds={(scale, shift)}"
            )
    elif model == "shift":
        if arch != "shift_deeponet" or (scale, shift) != (0.0, 0.0):
            raise AssertionError(
                f"Unbounded Shift identity failed for {source}: "
                f"arch={arch}, bounds={(scale, shift)}"
            )
    elif model == "fnn":
        # The dedicated FNN trainer predates the common model.arch dispatch and
        # identifies itself through the nested FNN configuration plus its
        # train_fnn manifest/prediction naming convention.
        if not (config.get("fnn") or model_cfg.get("fnn")):
            raise AssertionError(f"FNN configuration missing in {source}")
        arch = arch or "fnn_pipeline"
    return {"arch": arch, "bound_scale": scale, "bound_shift": shift}


def verify_b1_values(run_dir: Path, model: str, c_errors: np.ndarray) -> None:
    with (run_dir / "results" / "experiments" / "B1_results.json").open(
        encoding="utf-8"
    ) as stream:
        summary = json.load(stream)
    archived = np.asarray(
        summary[result_key(model)]["c"]["rel_l2"]["values"], dtype=np.float64
    )
    if archived.shape != c_errors.shape or not np.allclose(
        archived, c_errors, rtol=2.0e-5, atol=2.0e-7
    ):
        delta = float(np.max(np.abs(archived - c_errors)))
        raise AssertionError(f"B1 field/summary mismatch in {run_dir}; max delta={delta}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs_root = args.repo / "experiments" / "qtop_func" / "runs_revision"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance: dict = {
        "export_protocol": "corrected-field-evidence-v1",
        "selection_rule": (
            "For each scenario, compute each sample's concentration relative L2 "
            "for RegShift, Shift, and FNN at each of three seeds; average within "
            "model across seeds; convert each model's mean errors to within-scenario "
            "percentile ranks; select the sample minimizing the sum of squared "
            "distances of the three ranks from 0.5. Ties use the lowest index."
        ),
        "scenarios": {},
        "sources": {},
    }
    selected_rows: list[dict] = []
    regime_rows: list[dict] = []
    process_export: dict[str, np.ndarray] = {"seeds": np.asarray(SEEDS, dtype=np.int64)}
    heatmap_export: dict[str, np.ndarray] = {}

    for scenario in SCENARIOS:
        arrays: dict[str, list[dict[str, np.ndarray]]] = {model: [] for model in MODELS}
        errors: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
        canonical_ref: tuple[np.ndarray, np.ndarray] | None = None
        canonical_test: dict[str, np.ndarray] | None = None

        for model in MODELS:
            for seed in SEEDS:
                run_dir = runs_root / run_name(model, scenario, seed)
                pred_path = run_dir / "results" / "experiments" / prediction_name(model)
                test_path = run_dir / "data" / "processed" / "test.npz"
                if not pred_path.exists() or not test_path.exists():
                    raise FileNotFoundError(f"Missing field archive: {pred_path} or {test_path}")

                config = load_snapshot(run_dir)
                identity = verify_model_identity(model, config, run_dir)
                pred_npz = np.load(pred_path)
                test_npz = np.load(test_path)
                fields = {key: np.asarray(pred_npz[key]) for key in pred_npz.files}
                test = {
                    "branch_inputs": np.asarray(test_npz["branch_inputs"]),
                    "z": np.asarray(test_npz["z"]),
                    "t": np.asarray(test_npz["t"]),
                }
                for key, value in fields.items():
                    if not np.isfinite(value).all():
                        raise AssertionError(f"Non-finite {key} in {pred_path}")

                if canonical_ref is None:
                    canonical_ref = (fields["h_ref"], fields["c_ref"])
                    canonical_test = test
                else:
                    if not np.array_equal(canonical_ref[0], fields["h_ref"], equal_nan=True):
                        raise AssertionError(f"h reference mismatch in {pred_path}")
                    if not np.array_equal(canonical_ref[1], fields["c_ref"], equal_nan=True):
                        raise AssertionError(f"c reference mismatch in {pred_path}")
                    assert canonical_test is not None
                    for key in ("branch_inputs", "z", "t"):
                        if not np.array_equal(canonical_test[key], test[key], equal_nan=True):
                            raise AssertionError(f"Test-set {key} mismatch in {run_dir}")

                c_error = rel_l2_per_sample(fields["c_pred"], fields["c_ref"])
                verify_b1_values(run_dir, model, c_error)
                arrays[model].append(fields)
                errors[model].append(c_error)
                provenance["sources"][f"{model}:{scenario}:s{seed}"] = {
                    "run": str(run_dir.relative_to(args.repo)),
                    "prediction_file": str(pred_path.relative_to(args.repo)),
                    "prediction_sha256": sha256_file(pred_path),
                    "test_file": str(test_path.relative_to(args.repo)),
                    "test_sha256": sha256_file(test_path),
                    "identity": identity,
                    "n_samples": int(c_error.size),
                    "median_c_rel_l2": float(np.median(c_error)),
                }

        assert canonical_ref is not None and canonical_test is not None
        mean_errors = {
            model: np.mean(np.stack(errors[model], axis=0), axis=0) for model in MODELS
        }
        ranks = {model: percentile_ranks(mean_errors[model]) for model in MODELS}
        centrality = sum((ranks[model] - 0.5) ** 2 for model in MODELS)
        selected = int(np.argmin(centrality))

        selected_record = {
            "scenario": scenario,
            "selected_zero_based_index": selected,
            "centrality_score": float(centrality[selected]),
        }
        for model in MODELS:
            selected_record[f"{model}_three_seed_mean_c_rel_l2"] = float(
                mean_errors[model][selected]
            )
            selected_record[f"{model}_within_scenario_percentile"] = float(
                ranks[model][selected]
            )
        selected_rows.append(selected_record)
        provenance["scenarios"][scenario] = {
            **selected_record,
            "n_samples": int(canonical_ref[0].shape[0]),
            "reference_h_sha256": sha256_array(canonical_ref[0]),
            "reference_c_sha256": sha256_array(canonical_ref[1]),
            "branch_inputs_sha256": sha256_array(canonical_test["branch_inputs"]),
        }

        prefix = scenario
        process_export[f"{prefix}_selected_index"] = np.asarray(selected, dtype=np.int64)
        process_export[f"{prefix}_z"] = canonical_test["z"]
        process_export[f"{prefix}_t"] = canonical_test["t"]
        process_export[f"{prefix}_q_top"] = canonical_test["branch_inputs"][selected, 6:54]
        process_export[f"{prefix}_h_ref"] = canonical_ref[0][selected]
        process_export[f"{prefix}_c_ref"] = canonical_ref[1][selected]
        for model in MODELS:
            process_export[f"{prefix}_{model}_h_pred"] = np.stack(
                [entry["h_pred"][selected] for entry in arrays[model]], axis=0
            )
            process_export[f"{prefix}_{model}_c_pred"] = np.stack(
                [entry["c_pred"][selected] for entry in arrays[model]], axis=0
            )

        z = canonical_test["z"]
        t = canonical_test["t"]
        branch = canonical_test["branch_inputs"]
        depth_groups = {
            "0-20 cm": np.where(z <= 20.0)[0],
            "20-60 cm": np.where((z > 20.0) & (z <= 60.0))[0],
            "60-100 cm": np.where(z > 60.0)[0],
        }
        time_groups = {
            "0-16 h": np.where(t < 16.0)[0],
            "16-32 h": np.where((t >= 16.0) & (t < 32.0))[0],
            "32-48 h": np.where(t >= 32.0)[0],
        }
        q_peak = branch[:, 6:54].max(axis=1)
        q25, q75 = np.quantile(q_peak, [0.25, 0.75])
        intensity_groups = {
            "Low": np.where(q_peak <= q25)[0],
            "Mid": np.where((q_peak > q25) & (q_peak <= q75))[0],
            "High": np.where(q_peak > q75)[0],
        }
        provenance["scenarios"][scenario]["q_peak_quartiles"] = [
            float(q25),
            float(q75),
        ]

        for model in MODELS:
            for seed_position, seed in enumerate(SEEDS):
                pred = arrays[model][seed_position]["c_pred"]
                ref = canonical_ref[1]
                for group, indices in depth_groups.items():
                    regime_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "group_type": "depth",
                            "group": group,
                            "model": model,
                            "n_samples": ref.shape[0],
                            "n_grid_cells_per_sample": int(indices.size * ref.shape[2]),
                            "c_mae": float(np.mean(np.abs(pred[:, indices, :] - ref[:, indices, :]))),
                        }
                    )
                for group, indices in time_groups.items():
                    regime_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "group_type": "time",
                            "group": group,
                            "model": model,
                            "n_samples": ref.shape[0],
                            "n_grid_cells_per_sample": int(ref.shape[1] * indices.size),
                            "c_mae": float(np.mean(np.abs(pred[:, :, indices] - ref[:, :, indices]))),
                        }
                    )
                for group, indices in intensity_groups.items():
                    regime_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "group_type": "intensity",
                            "group": group,
                            "model": model,
                            "n_samples": int(indices.size),
                            "n_grid_cells_per_sample": int(ref.shape[1] * ref.shape[2]),
                            "c_mae": float(np.mean(np.abs(pred[indices] - ref[indices]))),
                        }
                    )

        heatmaps = []
        for seed_position, _seed in enumerate(SEEDS):
            reg = arrays["regshift"][seed_position]["c_pred"]
            shift = arrays["shift"][seed_position]["c_pred"]
            ref = canonical_ref[1]
            heatmaps.append(
                np.mean(np.abs(shift - ref) - np.abs(reg - ref), axis=0)
            )
        heatmap_export[f"{scenario}_z"] = z
        heatmap_export[f"{scenario}_t"] = t
        heatmap_export[f"{scenario}_shift_minus_regshift_abs_error"] = np.stack(
            heatmaps, axis=0
        )

    process_path = output_dir / "corrected_process_and_case_fields.npz"
    heatmap_path = output_dir / "corrected_regime_heatmaps.npz"
    np.savez_compressed(process_path, **process_export)
    np.savez_compressed(heatmap_path, **heatmap_export)

    selected_path = output_dir / "representative_case_selection.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)

    regime_path = output_dir / "corrected_regime_summary.csv"
    with regime_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(regime_rows[0]))
        writer.writeheader()
        writer.writerows(regime_rows)

    provenance_path = output_dir / "corrected_field_provenance.json"
    with provenance_path.open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)

    output_hashes = {}
    for path in (process_path, heatmap_path, selected_path, regime_path, provenance_path):
        output_hashes[path.name] = sha256_file(path)
    with (output_dir / "SHA256SUMS.json").open("w", encoding="utf-8") as stream:
        json.dump(output_hashes, stream, indent=2, sort_keys=True)

    print(json.dumps({"status": "AUDIT_OK", "outputs": output_hashes}, indent=2))


if __name__ == "__main__":
    main()
