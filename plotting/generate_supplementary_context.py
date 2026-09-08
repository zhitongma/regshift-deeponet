#!/usr/bin/env python3
"""Reproduce final Fig. S1 and S5; S5 retains the historical standard-model/FNN scope."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plot_inputs"
FIG_DIR = ROOT / "outputs" / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
def load(path):
    return json.loads(path.read_text())
def fig_sparse_crossres():
    si = load(DATA / "sparse_crossres.json")
    sc = si["ablations"]["sparse_crossres"]
    mb = si["main_benchmark"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    # (a) Sparse-query scan c-L2 IID
    q_labels = [200, 500, 1000]
    std_iid = [sc["sparse_q200"]["iid"]["StandardDeepONet"]["c_l2_mean"],
               sc["sparse_q500"]["iid"]["StandardDeepONet"]["c_l2_mean"],
               mb["StandardDeepONet"]["iid"]["c_l2_mean"]]
    std_iid_e = [sc["sparse_q200"]["iid"]["StandardDeepONet"]["c_l2_std"],
                 sc["sparse_q500"]["iid"]["StandardDeepONet"]["c_l2_std"],
                 mb["StandardDeepONet"]["iid"]["c_l2_std"]]
    fnn_iid = [sc["sparse_q200"]["iid"]["FNN"]["c_l2_mean"],
               sc["sparse_q500"]["iid"]["FNN"]["c_l2_mean"],
               mb["FNN"]["iid"]["c_l2_mean"]]
    fnn_iid_e = [sc["sparse_q200"]["iid"]["FNN"]["c_l2_std"],
                 sc["sparse_q500"]["iid"]["FNN"]["c_l2_std"],
                 mb["FNN"]["iid"]["c_l2_std"]]
    axes[0].errorbar(q_labels, std_iid, yerr=std_iid_e, marker="o",
                     color="#228833", label="Std DeepONet (IID)")
    axes[0].errorbar(q_labels, fnn_iid, yerr=fnn_iid_e, marker="s",
                     color="#4477AA", label="FNN (IID)")
    std_ood = [sc["sparse_q200"]["ood"]["StandardDeepONet"]["c_l2_mean"],
               sc["sparse_q500"]["ood"]["StandardDeepONet"]["c_l2_mean"],
               mb["StandardDeepONet"]["ood_peak"]["c_l2_mean"]]
    std_ood_e = [sc["sparse_q200"]["ood"]["StandardDeepONet"]["c_l2_std"],
                 sc["sparse_q500"]["ood"]["StandardDeepONet"]["c_l2_std"],
                 mb["StandardDeepONet"]["ood_peak"]["c_l2_std"]]
    fnn_ood = [sc["sparse_q200"]["ood"]["FNN"]["c_l2_mean"],
               sc["sparse_q500"]["ood"]["FNN"]["c_l2_mean"],
               mb["FNN"]["ood_peak"]["c_l2_mean"]]
    fnn_ood_e = [sc["sparse_q200"]["ood"]["FNN"]["c_l2_std"],
                 sc["sparse_q500"]["ood"]["FNN"]["c_l2_std"],
                 mb["FNN"]["ood_peak"]["c_l2_std"]]
    axes[0].errorbar(q_labels, std_ood, yerr=std_ood_e, marker="o",
                     color="#228833", linestyle="--", label="Std DeepONet (OOD peak)")
    axes[0].errorbar(q_labels, fnn_ood, yerr=fnn_ood_e, marker="s",
                     color="#4477AA", linestyle="--", label="FNN (OOD peak)")
    axes[0].set_xlabel("training query points per sample")
    axes[0].set_ylabel("concentration rel. $L_2$")
    axes[0].set_xticks(q_labels)
    axes[0].set_title("(a) Sparse-query training")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.3)

    # (b) Cross-resolution: bars for T=24 vs T=49
    cats = ["Std DeepONet\nIID", "Std DeepONet\nOOD", "FNN\nIID", "FNN\nOOD"]
    t24 = [sc["crossres_t24"]["iid"]["StandardDeepONet"]["c_l2_mean"],
           sc["crossres_t24"]["ood"]["StandardDeepONet"]["c_l2_mean"],
           sc["crossres_t24"]["iid"]["FNN"]["c_l2_mean"],
           sc["crossres_t24"]["ood"]["FNN"]["c_l2_mean"]]
    t49 = [mb["StandardDeepONet"]["iid"]["c_l2_mean"],
           mb["StandardDeepONet"]["ood_peak"]["c_l2_mean"],
           mb["FNN"]["iid"]["c_l2_mean"],
           mb["FNN"]["ood_peak"]["c_l2_mean"]]
    x = np.arange(len(cats))
    axes[1].bar(x - 0.18, t24, 0.36, label="$T{=}24$ (train)", color="#999999")
    axes[1].bar(x + 0.18, t49, 0.36, label="$T{=}49$ (reference)", color="#4477AA")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats, fontsize=8)
    axes[1].set_ylabel("concentration rel. $L_2$")
    axes[1].set_title("(b) Cross-resolution generalisation")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figS8_sparse_crossres.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / ("figS8_sparse_crossres.png" if len(fig.axes) == 2 else "figS10_param_sampling.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_param_sampling():
    csv_path = DATA / "parameters_iid.csv.gz"
    with gzip.open(csv_path, "rt", newline="", encoding="utf-8") as h:
        reader = csv.DictReader(h)
        rows = list(reader)
    cols = ["theta_r", "theta_s", "alpha", "n_vg", "K_s", "D_L", "c_top",
            "q_top_peak", "q_top_mean", "q_top_peak_time"]
    titles = [r"$\theta_r$", r"$\theta_s$", r"$\alpha$ (log)", "$n$",
              "$K_s$ (log)", "$D_L$", "$c_{\\mathrm{top}}$",
              "peak $q_{\\mathrm{top}}$", "mean $q_{\\mathrm{top}}$",
              "peak-time index"]
    fig, axes = plt.subplots(2, 5, figsize=(12, 4.6))
    for ax, col, title in zip(axes.flat, cols, titles):
        vals = np.array([float(r[col]) for r in rows])
        log = col in ("alpha", "K_s")
        if log:
            ax.hist(np.log10(vals), bins=30, color="#4477AA", alpha=0.8, edgecolor="white")
            ax.set_xlabel(f"$\\log_{{10}}$ {title.split(' (')[0]}")
        else:
            ax.hist(vals, bins=30, color="#4477AA", alpha=0.8, edgecolor="white")
            ax.set_xlabel(title)
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    fig.suptitle("Marginal distribution of Latin hypercube samples (IID pool, $n=1024$)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figS10_param_sampling.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / ("figS8_sparse_crossres.png" if len(fig.axes) == 2 else "figS10_param_sampling.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR)
    FIG_DIR = parser.parse_args().output_dir.resolve()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_sparse_crossres()
    fig_param_sampling()
