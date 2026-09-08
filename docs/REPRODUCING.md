# Training and analysis

## Data contract

The training pipeline expects the existing processed dataset schema produced by `pipeline/03_process_data.py`. Supply a directory with `processed/{train,val,test,scaler}.npz` and any corresponding `parameters/` or `raw/` needed by your analysis. The figure-only arrays in `data/plot_inputs/` are not training datasets.

All model/solver parameters remain in the original YAML format. Bounds must appear under `model.shift_deeponet`. The portable launcher validates model identity and bounds before starting a run.

## Final benchmark configurations

| Model / experiment | Configuration directory | Launcher model |
|---|---|---|
| Corrected bounded RegShift, high budget | `configs/revision/E0_bounded/` | `regshift` |
| Standard DeepONet, high budget | `configs/revision/E1_budget/e1_deeponet_highbudget_*.yaml` | `deeponet` |
| Unbounded Shift, high budget | `configs/revision/E1_budget/e1_shift_highbudget_*.yaml` | `shift` |
| FNN, high budget | `configs/revision/E1_budget/e1_fnn_highbudget_*.yaml` | `fnn` |
| Bounded RegShift, low budget | `configs/revision/E1_budget/e1_regshift_lowbudget_*.yaml` | `regshift` |
| Bound sensitivity | `configs/revision/E2_bounds/` | `regshift` |
| Tuned FNO | `configs/revision/E3_fno_final/` | `fno` |

Use seeds 42, 123, and 456 and the scenario-specific `iid`, `ood_peak`, and `ood_peak_time` files when following the final benchmark. `base/boost_A4_shift.yaml` is a configuration fragment, not a standalone complete run configuration.

For example:

```bash
python scripts/run_experiment.py \
  --config configs/revision/E3_fno_final/fno_best_iid.yaml \
  --model fno --seed 42 --device cuda \
  --data-dir /path/to/iid/data \
  --run-dir outputs/runs/e3_fno_final_iid_s42
```

The default `train,evaluate` stages run both M1 and M2 for the DeepONet family, or the relevant standalone trainer for FNN/FNO/GridFNN, then the B1–B4 evaluation pipeline. `--epochs` overrides each training stage and is intended for smoke runs, not paper comparisons. Seeds and budgets should be taken from the selected YAML when reproducing the paper.

Use `--stages evaluate` to evaluate existing checkpoints in a specified run directory. The full pipeline defaults and additional switches remain available through each `pipeline/*.py --help` entry point. Avoid reusing a completed run directory unless deliberately continuing that run; checkpoints can be replaced by the original training routines.

## Generate new simulation data

The solver executable is external. Install the optional Python interface with `python -m pip install -r requirements/hydrus.txt` and supply a compatible HYDRUS-1D executable:

```bash
python scripts/run_experiment.py \
  --config configs/revision/E0_bounded/e0_regshift_highbudget_iid.yaml \
  --model regshift --seed 42 \
  --run-dir outputs/runs/new_iid_dataset \
  --stages parameters,simulate,process \
  --hydrus-exe /path/to/hydrus
```

Do not pass `--data-dir` with data-generation stages: it mounts existing data. HYDRUS integration was preserved from the source archive but not exercised during this cleanup. The external solver version is not bundled or inferred.

## Analysis index

| Directory | Purpose | Inputs beyond the bundled summaries |
|---|---|---|
| E0 | Bound checks and export of representative field evidence | Checkpoints/full prediction arrays for export |
| E1 | Budget parity, training manifests and aggregate metrics | Run metadata and results |
| E2 | Bound sensitivity and concentration-front envelopes | Configurations, processed fields and run results |
| E3 | FNO search configuration generation and validation | Search/final run results |
| E4 | Sampling/QC selection diagnostics | Parameter tables and solver QC records |
| E5 | Observed-rainfall events and stress-test evaluation | Rainfall series and external-test results |
| E6E8 | Extreme/joint shifts, severity and subset analysis | User-supplied run/model manifests and full predictions |
| E7 | Monte Carlo screening and HYDRUS confirmation | Checkpoints, screening fields and solver outputs |
| E9 | Inference/solver costs and break-even analysis | Run metadata, checkpoints or solver executable |
| E10 | Mass-balance diagnostic and controlled noise analysis | Field arrays or per-sample metric CSVs |
| E11 | HYDRUS concurrency benchmark and report | Bundled summary suffices for report generation |

These are specialist scripts retained from the final local package. Supply the explicit `--repo`, `--run-dir`, `--config`, `--manifest`, or `--out-dir` arguments they describe; older default run identifiers encode the historical experiment layout. Some no-argument modes need files from the original server workspace that are not distributed here. They are not invoked by the figure command. New training uses `scripts/run_experiment.py`, which has no dependency on the historical layout or old shell launchers.

E3's final width/modes/padding/learning-rate values are confirmed by the frozen search/final records. The source search generator is preserved as an experiment utility; the published final configuration files are the benchmark entry points.

Historical `g4_regshift_*` checkpoints used an unbounded map because of misplaced bound settings. Keep their historical identity when working with old results. The final bounded rows use the corrected E0 runs; this assembly does not relabel old checkpoints or claim the unavailable full server training has been reproduced.
