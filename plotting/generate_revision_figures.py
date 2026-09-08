#!/usr/bin/env python3
"""Generate corrected revision figures from the frozen aggregate CSV files."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plot_inputs"
OUT = ROOT / "outputs" / "figures"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def benchmark() -> None:
    rows = read_csv("revision_main_benchmark.csv")
    models = ["FNN", "Standard DeepONet", "Shift-DeepONet", "RegShift", "FNO"]
    scenarios = ["IID", "OOD peak", "OOD peak-time"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    x = np.arange(len(scenarios), dtype=float)
    width = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
    for panel, (metric, label) in enumerate((("h", r"Pressure-head median relative $L_2$"),
                                              ("c", r"Concentration median relative $L_2$"))):
        ax = axes[panel]
        for index, (model, color) in enumerate(zip(models, colors)):
            selected = [next(r for r in rows if r["model"] == model and r["scenario"] == scenario)
                        for scenario in scenarios]
            means = [float(r[f"{metric}_mean"]) for r in selected]
            sds = [float(r[f"{metric}_sd"]) for r in selected]
            offset = (index - (len(models) - 1) / 2) * width
            ax.bar(x + offset, means, width=width, color=color, label=model,
                   yerr=sds, capsize=2.5, error_kw={"elinewidth": 0.9})
        ax.set_xticks(x, scenarios)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.text(0.01, 0.98, f"({chr(97 + panel)})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, "fig3_main_benchmark")


def breakthrough() -> None:
    rows = read_csv("revision_breakthrough.csv")
    scenarios = [r["scenario"] for r in rows]
    means = [float(r["mae_mean_h"]) for r in rows]
    sds = [float(r["mae_sd_h"]) for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    bars = ax.bar(x, means, yerr=sds, capsize=5, width=0.62,
                  color=["#4C78A8", "#F58518", "#54A24B"],
                  error_kw={"elinewidth": 1.2})
    ax.set_xticks(x, scenarios)
    ax.set_ylabel(r"Breakthrough-time MAE, $|\Delta t_b|$ [h]")
    ax.set_ylim(0, max(np.add(means, sds)) * 1.22)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, mean, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.12,
                f"{mean:.2f} ± {sd:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, "fig5_bt_summary")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark()
    breakthrough()
