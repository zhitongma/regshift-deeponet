#!/usr/bin/env python3
"""Reproduce final Fig. 1a-c from the original sampled parameter tables."""
import argparse
import csv
import gzip
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plot_inputs"
FIG_DIR = ROOT / "outputs" / "figures"
def load_csv_rows(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
sns.set_theme(
    style="ticks",
    context="paper",
    font="serif",
    font_scale=1.8,
    rc={
        "font.family":         "serif",
        "mathtext.fontset":    "cm",
        "figure.dpi":          300,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.08,
        "axes.linewidth":      1.0,
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.minor.width":   0.5,
        "ytick.minor.width":   0.5,
        "xtick.major.size":    4.5,
        "ytick.major.size":    4.5,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "axes.grid":           False,
        "grid.alpha":          0.3,
        "grid.linewidth":      0.6,
        "grid.linestyle":      "--",
        "legend.frameon":      False,
        "legend.fontsize":     14,
        "axes.labelsize":      16,
        "axes.titlesize":      17,
        "xtick.labelsize":     14,
        "ytick.labelsize":     14,
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
    },
)

SCENARIO_COLORS = {"iid": "#3C7EC2", "ood_peak": "#E8873D", "ood_peak_time": "#4EA84E"}
def save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")
    plt.close(fig)
    print(f"  -> {stem}.pdf / .png")


def forcing_cols(header):
    cols = [c for c in header if c.startswith("q_top_") and c.rsplit("_",1)[-1].isdigit()]
    return sorted(cols, key=lambda x: int(x.rsplit("_",1)[-1]))


def add_grid(ax, axis="y"):
    ax.grid(True, axis=axis, alpha=0.3, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


def fig1_task_setup():
    """Generate panels (a), (b), (c) as separate PDFs; panel (d) is TikZ."""
    print("Fig. 1: Task setup (3 separate panels)")
    iid_rows = load_csv_rows(DATA / "parameters_iid.csv.gz")
    ood_p    = load_csv_rows(DATA / "parameters_ood_peak.csv.gz")
    ood_t    = load_csv_rows(DATA / "parameters_ood_peak_time.csv.gz")
    q_cols   = forcing_cols(list(iid_rows[0].keys()))
    tax      = np.arange(len(q_cols))

    # --- (a) forcing examples ---
    # Cleaner layout: faint grey training envelope as median + shaded band,
    # plus three distinctly styled coloured test hydrographs (IID, OOD-peak,
    # OOD-peak-time) so curves remain legible without visual clash.
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    train_vals = np.array([
        [float(r[c]) for c in q_cols]
        for r in iid_rows if r["split"] == "train"
    ])
    # Background envelope: 10--90th percentile band + faint median line.
    q10 = np.percentile(train_vals, 10, axis=0)
    q50 = np.percentile(train_vals, 50, axis=0)
    q90 = np.percentile(train_vals, 90, axis=0)
    ax.fill_between(tax, q10, q90, color="#CFD8DC", alpha=0.55,
                    label="Training 10--90%", linewidth=0)
    ax.plot(tax, q50, color="#78909C", lw=1.0, ls=":",
            label="Training median")
    # Three test hydrographs, one per scenario.
    test_iid = next(r for r in iid_rows if r["split"] == "test")
    test_op  = next(r for r in ood_p  if r["split"] == "test")
    test_ot  = next(r for r in ood_t  if r["split"] == "test")
    ax.plot(tax, [float(test_iid[c]) for c in q_cols],
            color=SCENARIO_COLORS["iid"], lw=2.2, ls="-", label="IID test")
    ax.plot(tax, [float(test_op[c])  for c in q_cols],
            color=SCENARIO_COLORS["ood_peak"], lw=2.2, ls="--",
            label="OOD (peak) test")
    ax.plot(tax, [float(test_ot[c])  for c in q_cols],
            color=SCENARIO_COLORS["ood_peak_time"], lw=2.2, ls="-.",
            label="OOD (peak-time) test")
    ax.set(xlabel="Time [h]",
           ylabel=r"$q_{\mathrm{top}}(t)$ [cm h$^{-1}$]",
           xlim=(0, 48), ylim=(0, 5.6))
    add_grid(ax, "both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False, fontsize=9, handlelength=2.4,
              columnspacing=1.2)
    sns.despine(fig=fig)
    fig.tight_layout()
    save(fig, "fig1a")

    # Helper: overlay train+val (step outline) with test (filled bars) so
    # tail overlap is readable and the OOD cutoff is explicit.
    def _split_hist(ax, tr, te, bins, cutoff, test_label, xlabel):
        tr_counts, _ = np.histogram(tr, bins=bins)
        te_counts, _ = np.histogram(te, bins=bins)
        ymax_data = float(max(tr_counts.max(), te_counts.max()))
        ax.hist(tr, bins=bins, color=SCENARIO_COLORS["iid"], alpha=0.85,
                edgecolor="white", linewidth=0.5, label="Train + val",
                zorder=2)
        ax.hist(te, bins=bins, color=SCENARIO_COLORS["ood_peak"], alpha=0.95,
                edgecolor="white", linewidth=0.5, label=test_label,
                zorder=3)
        ax.axvline(cutoff, color="#424242", lw=1.0, ls="--", zorder=4)
        ax.set(xlabel=xlabel, ylabel="Count")
        ax.set_ylim(0, 1.28 * ymax_data)
        add_grid(ax, "y")
        ax.legend(loc="upper left", frameon=False, fontsize=9,
                  handlelength=1.4, handletextpad=0.5, borderaxespad=0.6)

    # --- (b) OOD-peak histogram ---
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ptr = np.array([float(r["q_top_peak"]) for r in ood_p
                    if r["split"] in {"train", "val"}])
    pte = np.array([float(r["q_top_peak"]) for r in ood_p
                    if r["split"] == "test"])
    lo, hi = float(min(ptr.min(), pte.min())), float(max(ptr.max(), pte.max()))
    bins = np.linspace(lo, hi, 18)
    cutoff = float(np.percentile(np.concatenate([ptr, pte]), 90))
    _split_hist(ax, ptr, pte, bins, cutoff,
                "Test (top 10%)",
                r"$q_{\mathrm{top}}^{\mathrm{peak}}$ [cm h$^{-1}$]")
    sns.despine(fig=fig)
    fig.tight_layout()
    save(fig, "fig1b")

    # --- (c) OOD-peak-time histogram ---
    # Use bins aligned to the 8-hour control-point spacing to avoid
    # spurious gaps produced by the original 2-hour bins.
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ttr = np.array([float(r["q_top_peak_time"]) for r in ood_t
                    if r["split"] in {"train", "val"}])
    tte = np.array([float(r["q_top_peak_time"]) for r in ood_t
                    if r["split"] == "test"])
    bins = np.arange(-0.5, 49.5, 8.0)
    cutoff = float(np.percentile(np.concatenate([ttr, tte]), 90))
    _split_hist(ax, ttr, tte, bins, cutoff,
                "Test (latest 10%)",
                r"$t_{\mathrm{peak}}$ [h]")
    sns.despine(fig=fig)
    fig.tight_layout()
    save(fig, "fig1c")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    FIG_DIR = parser.parse_args().output_dir.resolve()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_task_setup()
