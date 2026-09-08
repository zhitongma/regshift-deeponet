# Local validation

Validation date: 2026-09-08. Environment: macOS, Python 3.12.4, PyTorch 2.3.0, NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1, PyYAML 6.0.1, Matplotlib 3.9.0 and seaborn 0.13.2. Diagram builds used the installed pdfLaTeX/TikZ toolchain.

## Passed

- Syntax checks for all 87 Python source and test files.
- SHA-256 verification of all 69 distributed data files, before and after figure generation.
- All 43 YAML files parse. The 12 E1 configs, three final FNO configs and E7 config also pass their archive-specific validation commands.
- Nine regression tests pass: training/evaluation builders receive effective coordinate bounds; misnested bounds fail explicitly; the FNO final configuration activates 8×8 padding; padding affects the forward pass while retaining checkpoint keys; shapes and gradients remain valid; the launcher orders stages correctly, handles paths with spaces, and protects existing data directories.
- All nine pipeline command-line help entry points work from a working directory outside the repository.
- RegShift, Shift, Standard DeepONet, FNN and FNO launcher dry runs succeed without datasets or training.
- The table export and HYDRUS concurrency report regeneration succeed from the frozen inputs.
- All 22 figure PDFs regenerate, including the two TikZ diagrams, using bundled compact inputs.
- Rendered all 22 regenerated PDFs and compared them with the final submission figure PDFs. All extracted figure text agrees; 17 figures are pixel-identical at the comparison resolution. Five figures have rendering differences, including a one-pixel raster-height difference for Figures 8 and 9. The numerical input files and retained plotting functions were preserved, and the figure overview was visually checked.
- The repository was copied to a different folder whose path contains spaces. Integrity checks, training dry-run command resolution and all 22 figure builds pass there from an unrelated working directory.
- Git ignores external data, checkpoints, generated outputs, caches and local credentials, while retaining both small representative plotting NPZ files. No embedded credential pattern was detected in the scanned text files.

## Not tested or established

Full model training, final-server checkpoint loading, complete numerical reproduction of the paper, and the external HYDRUS solver stages were not rerun. The final server checkout and checkpoint archive were not available for direct comparison. The source reconstruction and supplied FNO patch are documented in PROVENANCE.md.

Pixel comparison concerns figure rendering, not retraining. Matching final figure inputs does not replace an independent training reproduction.
