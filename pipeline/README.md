# Pipeline entry points

1. `01_generate_params.py`: parameter and forcing sampling.
2. `02_run_hydrus.py`: external HYDRUS-1D simulation.
3. `03_process_data.py`: QC, splits and training normalization.
4. `04_train_m1.py`: first-stage DeepONet-family training.
5. `05_train_m2.py`: second-stage DeepONet-family training.
6. `06_evaluate.py`: B1-B4 accuracy, conservation, hydrology and timing evaluation.

`04b_train_fnn.py`, `04c_train_fno.py` and `04e_train_grid_fnn.py` are baseline trainers.

Use `python scripts/run_experiment.py --help` from the repository root for the portable launcher. Direct pipeline scripts accept `--config` and `--run-dir`; their `--help` lists additional options. Defaults write under `outputs/default_run/`.
