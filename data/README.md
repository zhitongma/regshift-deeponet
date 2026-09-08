# Bundled and external data

`plot_inputs/` and `frozen_results/` contain only the compact files needed to redraw the final figures and inspect the archived results.

- Three losslessly compressed parameter CSVs support Fig. 1 and S1. The IID table is byte-identical to the original parameter table used for Fig. S1.
- Two compact NPZ files contain representative cases and aggregated heatmaps exported from corrected final-revision evidence. They are not full training/prediction archives.
- Frozen CSV/JSON/TSV files contain the final benchmark, training, FNO search, stress-test and timing summaries.
- `sparse_crossres.json` contains only the retained historical Standard DeepONet/FNN comparison needed for Fig. S5; discarded open-operator and historical bounded-model claims are not included in it.
- `SHA256SUMS.json` verifies all distributed data files. Drawing figures must not modify them.

Full HYDRUS raw fields, train/validation/test NPZ arrays, model weights, complete prediction archives and run directories are omitted. The corresponding final server artifacts were not present in the inspected local final-revision archive. No download URL or access guarantee is supplied here.

Keep recovered datasets outside this repository, or under ignored `data/external/`. Pass their location to the launcher with `--data-dir`. This must point to a directory containing `processed/`, `parameters/` and, when needed, `raw/`.

Some preserved metadata strings record the original server paths. They identify historical provenance and are not download links or portable runtime paths. The active figure and training launchers do not use them to locate files.
