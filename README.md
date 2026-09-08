# RegShift-DeepONet

Code accompanying **Benchmarking a bounded-coordinate DeepONet for unsaturated flow and solute transport under time-varying infiltration**, accepted by *Journal of Hydrology* (article reference HYDROL 136388).

[中文使用说明](docs/README.zh-CN.md) · [Training and analysis](docs/REPRODUCING.md) · [Figure index](docs/FIGURES.md) · [Source provenance](docs/PROVENANCE.md)

This repository was assembled from the final local revision source package. It contains the model implementations, data-generation and training pipeline, final experiment configurations, analysis utilities, and scripts for all **22 figure files**: 13 main-text files (Fig. 1a–d and Fig. 2–10) and 9 supplementary figures.

Small frozen result tables and representative plotting arrays are included. Full HYDRUS fields, training/test arrays, prediction archives, and trained checkpoints are **not included**. Figure reproduction works with the bundled inputs; reproducing training requires the external datasets. See [data availability](data/README.md).

## Repository layout

```text
regshift-deeponet/
├── src/                    # Models, sampling, HYDRUS interface, metrics, utilities
├── pipeline/               # Parameter generation → simulation → processing → training → evaluation
├── configs/revision/       # Final E0–E9 experiment configurations
├── analysis/               # E0–E11 analysis and diagnostics
├── plotting/               # Final plotting scripts, figure index and two TikZ diagrams
├── data/
│   ├── plot_inputs/        # Small inputs needed to redraw the figures
│   ├── frozen_results/     # Frozen CSV/JSON result and timing summaries
│   └── SHA256SUMS.json     # Integrity manifest for bundled data
├── scripts/                # Portable run, figure, table and verification commands
├── tests/                  # Model/configuration regression checks
├── requirements/           # Plotting, training and solver dependency lists
├── docs/                   # Usage, provenance and publication notes
└── outputs/                # Created when running commands; ignored by Git
```

## Install

Use Python **3.12**, matching the recorded server Python version family and the local validation environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Drawing figures and exporting frozen result tables:
python -m pip install -r requirements/plotting.txt

# Also running model checks, training and evaluation:
python -m pip install -r requirements/training.txt
```

Run the following commands from the repository root. The launchers also locate source/data paths from their own file locations, so the folder can be moved without editing personal machine paths.

## Reproduce figures and tables

```bash
python scripts/check_repo.py
python scripts/reproduce_figures.py
python scripts/export_tables.py
```

Figures are written as `outputs/figures/Figure_1a.pdf`, …, `Figure_10.pdf`, and `Figure_S1.pdf`, …, `Figure_S9.pdf`, with PNG previews for the 20 Python-generated plots. Tables and the regenerated HYDRUS timing report go into `outputs/tables/`.

Fig. 1d and Fig. 2 require `pdflatex` plus the `standalone` and `tikz` packages. To reproduce just the 20 Python-generated figures:

```bash
python scripts/reproduce_figures.py --skip-diagrams
```

The figure command stages intermediate files in a temporary directory and leaves the frozen inputs unchanged. [The figure index](docs/FIGURES.md) maps every figure to its script and input source. Fig. S5 retains the final supplement's historical Standard DeepONet/FNN sparse-query comparison; it is not a new bounded-model benchmark.

## Train with external data

First inspect the commands without starting training:

```bash
python scripts/run_experiment.py \
  --config configs/revision/E0_bounded/e0_regshift_highbudget_iid.yaml \
  --model regshift --seed 42 --dry-run
```

Then point to a data directory containing `processed/`, `parameters/`, and optionally `raw/`:

```bash
python scripts/run_experiment.py \
  --config configs/revision/E0_bounded/e0_regshift_highbudget_iid.yaml \
  --model regshift --seed 42 --device cuda \
  --data-dir /path/to/external/iid/data
```

The command mounts the existing data by directory symlinks and writes checkpoints/logs/results under a new `outputs/runs/` directory. It refuses to replace an existing different data directory. [Training documentation](docs/REPRODUCING.md) covers the other models, simulation stages, and analysis utilities.

## Verification

```bash
python scripts/check_repo.py --models
python scripts/check_repo.py --figures
# Without a TeX installation:
python scripts/check_repo.py --figures --skip-diagrams
```

Checks cover source syntax, all bundled-input hashes, final configuration loading, effective coordinate bounds through the training/evaluation builders, FNO padding and checkpoint-schema compatibility, and figure generation. Full server training and HYDRUS solver runs were not repeated during repository assembly.

## Provenance and publication

The original local FNO implementation did not contain its accompanying final-revision padding patch. That supplied patch is applied here and retained in `docs/patches/`. The final-revision model hyperparameters and the remaining model implementations were preserved. Misplaced top-level coordinate-bound settings now raise an error instead of silently creating an unbounded model.

The final-revision server checkpoint archive was not available locally. This is therefore a documented reconstruction from the local final package and its supplied patch, not a verified mirror of the complete server checkout. Details and file-level provenance are in [PROVENANCE.md](docs/PROVENANCE.md).

This folder has no configured GitHub remote. No code license or permanent data DOI has been invented; the authors can add them before publication. See [GitHub preparation](docs/GITHUB.md).
