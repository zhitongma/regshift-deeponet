# Final figure index

The figure launcher emits the submission filenames below. Python plots also emit PNG previews; TikZ diagrams emit vector PDFs.

| Figure file | Plotting source | Input |
|---|---|---|
| `Figure_1a.pdf` | [`generate_task_setup.py`](../plotting/generate_task_setup.py) | compressed parameter CSVs |
| `Figure_1b.pdf` | [`generate_task_setup.py`](../plotting/generate_task_setup.py) | compressed parameter CSVs |
| `Figure_1c.pdf` | [`generate_task_setup.py`](../plotting/generate_task_setup.py) | compressed parameter CSVs |
| `Figure_1d.pdf` | [`diagrams/fig1d_workflow.tex`](../plotting/diagrams/fig1d_workflow.tex) | TikZ source only |
| `Figure_2.pdf` | [`diagrams/fig2_architecture.tex`](../plotting/diagrams/fig2_architecture.tex) | TikZ source only |
| `Figure_3.pdf` | [`generate_revision_figures.py`](../plotting/generate_revision_figures.py) | frozen benchmark/breakthrough CSVs |
| `Figure_4.pdf` | [`generate_corrected_field_figures.py`](../plotting/generate_corrected_field_figures.py) | corrected representative-case arrays and regime summaries |
| `Figure_5.pdf` | [`generate_revision_figures.py`](../plotting/generate_revision_figures.py) | frozen benchmark/breakthrough CSVs |
| `Figure_6.pdf` | [`generate_corrected_field_figures.py`](../plotting/generate_corrected_field_figures.py) | corrected representative-case arrays and regime summaries |
| `Figure_7.pdf` | [`generate_corrected_field_figures.py`](../plotting/generate_corrected_field_figures.py) | corrected representative-case arrays and regime summaries |
| `Figure_8.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_9.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_10.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_S1.pdf` | [`generate_supplementary_context.py`](../plotting/generate_supplementary_context.py) | compressed IID parameters and retained sparse/cross-resolution summaries |
| `Figure_S2.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_S3.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_S4.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_S5.pdf` | [`generate_supplementary_context.py`](../plotting/generate_supplementary_context.py) | compressed IID parameters and retained sparse/cross-resolution summaries |
| `Figure_S6.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_S7.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_S8.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |
| `Figure_S9.pdf` | [`generate_restored_figures.py`](../plotting/generate_restored_figures.py) | frozen E1-E11 tables and summaries |

All 22 figure files were regenerated during local verification. The original final PDFs are not bundled; see [validation](VALIDATION.md) for the comparison scope.

`Figure_S5` reproduces the final supplement's historical Standard DeepONet/FNN comparison. It does not add a new claim about corrected bounded RegShift training.
