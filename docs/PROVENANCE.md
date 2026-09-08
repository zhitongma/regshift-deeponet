# Source provenance

This assembly uses the local `最终回复/06_补充代码/` as its main source. The accompanying archive contained the same 266 non-hidden files. The manuscript's three clean TeX copies were identical, and `07_源文件/` supplied the additional final plotting script and diagram sources.

## Imported and retained

- Model, data generation, metric and utility source from the final supplementary code package.
- Training/evaluation pipeline and the final YAML configurations.
- E0–E11 Python analysis utilities.
- Corrected aggregate and representative-field plot inputs.
- `07_源文件/scripts/generate_restored_figures.py`, which was missing from the original supplementary code package.
- `fig1d_workflow.tex` and `fig2_architecture.tex` from the final manuscript's figure directory.
- Losslessly compressed parameter tables from the older local project, needed by the final retained Fig. 1 and S1. These parameter tables contain sampling/forcing values rather than solver fields or model predictions.
- The historical sparse/cross-resolution Standard DeepONet/FNN subset used by the final Fig. S5.

`source_inventory.json` records each imported/derived file's local relative source and original SHA-256. The data manifest independently records the distributed input hashes. Metadata imported for historical traceability can contain old run names and server path labels.

## Integrated changes

1. Applied the archive's supplied `analysis/E3/fno_padding.patch` to `src/models/fno.py`. The patch file is retained under `docs/patches/`. The final FNO configurations already specify width 32, modes 32×16, padding 8×8, and learning rate 1e-3; stale placeholder comments were corrected after cross-checking the frozen E3 record.
2. Added explicit rejection of top-level `shift_deeponet` settings in the configuration loader. Final nested configurations remain unchanged. This prevents a known historical silent failure when loading new or misnested configurations.
3. Preserved the existing Shift-DeepONet transform initialization. An optional historical patch suggestion was not applied because there was no final-server evidence authorizing a change to the experimental initialization.
4. Added portable figure, training, table and verification launchers. The default unqualified pipeline output directory is now `outputs/default_run/` instead of the source root.
5. Extracted only the retained task-setup and supplementary-context plotting functions from broader old plotting scripts; final scientific values and plot functions were preserved.
6. Redirected generated field-plot summaries/manifests into the output directory so they do not overwrite frozen evidence.
7. Replaced personal absolute repository defaults in analysis Python scripts with a root derived from the script location. Specialist analyses may still require the original external run/manifest schema and explicit arguments.

## Omitted

No raw simulation archive, trained weights, complete prediction fields, LaTeX manuscript, compiled manuscript PDF, reviewer letters, internal planning documents, cache files, execution logs, old server queue monitors, or machine-specific shell launchers are included. The final manuscript drawing routines are present, but superseded historical benchmark plots and the discarded open-operator plotting path are not offered as final-paper defaults. The obsolete historical-unbounded E5 configuration was excluded.

## Verification boundary

The final server source tree and checkpoints were not available locally for a byte-for-byte comparison. This repository reconstructs the final workflow from the available final local files and an explicitly supplied model patch. It does not establish that every server-side change was recovered or that fresh training will exactly reproduce the published numerical results. This limitation is documented separately from the local checks in `VALIDATION.md`.
