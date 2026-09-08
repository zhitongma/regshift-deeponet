#!/usr/bin/env python3
"""Generate provenance-traceable figures restored in the final revision.

Every plotted number is read from a frozen revision CSV.  The script does not
read historical ``g4_regshift`` prediction arrays, because those checkpoints
did not activate the coordinate bounds and are not valid bounded-model
evidence.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen_results"
AUDIT = FROZEN / "server_audit_20260723"
E11 = FROZEN / "E11"
OUT = ROOT / "outputs" / "figures"

COLORS = {
    "blue": "#2F6690",
    "orange": "#E67E22",
    "green": "#2A9D8F",
    "red": "#C94C4C",
    "purple": "#7B6D8D",
    "grey": "#6C757D",
    "light": "#DCE6F1",
}
SCENARIO_COLORS = {
    "iid": COLORS["blue"],
    "ood_peak": COLORS["orange"],
    "ood_peak_time": COLORS["green"],
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    return float(value)


def mean_sd(value: str) -> tuple[float, float]:
    left, right = value.replace("±", "+-").split("+-")
    return float(left.strip()), float(right.strip())


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10)
    ax.set_title(title, loc="left", pad=7)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_applicability() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)

    stats = rows(AUDIT / "E5" / "stats_compare_table.csv")
    acf = [r for r in stats if r["metric"].startswith("acf_lag")]
    lags = np.arange(1, len(acf) + 1)
    train = np.array([mean_sd(r["train_pool"])[0] for r in acf])
    train_sd = np.array([mean_sd(r["train_pool"])[1] for r in acf])
    real = np.array([mean_sd(r["real_events"])[0] for r in acf])
    real_sd = np.array([mean_sd(r["real_events"])[1] for r in acf])
    ax = axes[0]
    ax.errorbar(lags, train, yerr=train_sd, color=COLORS["blue"], marker="o", ms=3,
                lw=1.5, capsize=2, label="Training pool")
    ax.errorbar(lags, real, yerr=real_sd, color=COLORS["orange"], marker="s", ms=3,
                lw=1.5, capsize=2, label="Observed events")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set(xlabel="Lag [h]", ylabel="Autocorrelation", xticks=lags, ylim=(-0.35, 1.08))
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "a", "Forcing-shape mismatch")

    realrain = rows(AUDIT / "E5" / "e5_results_realrain_s42.csv")
    display = {
        "DeepONet": "Standard\nDeepONet",
        "Shift-DeepONet": "Unbounded\nShift",
        "RegShift(修复代码重跑, 真有界)": "Bounded\nRegShift",
    }
    order = ["DeepONet", "Shift-DeepONet", "RegShift(修复代码重跑, 真有界)"]
    vals = {r["model"]: f(r["c_rel_l2"]) for r in realrain}
    ax = axes[1]
    bars = ax.bar(np.arange(3), [vals[k] for k in order],
                  color=[COLORS["grey"], COLORS["purple"], COLORS["green"]], width=0.68)
    ax.set_xticks(np.arange(3), [display[k] for k in order])
    ax.set_ylabel("Mean concentration relative $L_2$")
    ax.set_ylim(0, 3.25)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.4)
    panel(ax, "b", "Observed-rainfall stress test")

    stress = rows(AUDIT / "E6E8" / "E6E8_results_long.csv")
    sets = ["E6_extreme_peak_v1", "E6_extreme_peaktime_v1", "E8_joint_shift_v1"]
    sources = [
        "regshift_e0_iid_s42",
        "regshift_e0_ood_peak_s42",
        "regshift_e0_ood_peak_time_s42",
    ]
    source_labels = ["IID-trained", "Peak-shift-trained", "Peak-time-trained"]
    set_labels = ["Extreme\npeak", "Extreme\npeak-time", "Joint forcing\n× $K_s$"]
    lookup = {
        (r["set"], r["source"]): f(r["c_rel_l2_mean"])
        for r in stress
        if r["model"] == "m2_data" and r["source"] in sources
    }
    ax = axes[2]
    x = np.arange(len(sets))
    width = 0.24
    for i, (source, label, color) in enumerate(
        zip(sources, source_labels, [COLORS["blue"], COLORS["orange"], COLORS["green"]])
    ):
        ax.bar(x + (i - 1) * width, [lookup[(s, source)] for s in sets], width,
               label=label, color=color)
    ax.set_xticks(x, set_labels)
    ax.set_ylabel("Mean concentration relative $L_2$")
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    panel(ax, "c", "Corrected bounded-model stress sets")
    save(fig, "fig5_applicability_envelope")


def plot_reliability() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)

    qc = rows(AUDIT / "E4" / "qc_counts.csv")
    scenarios = ["iid", "ood_peak", "ood_peak_time"]
    splits = ["train", "val", "test"]
    labels = ["IID", "OOD peak", "OOD peak-time"]
    ax = axes[0]
    x = np.arange(3)
    width = 0.24
    split_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for i, (split, color) in enumerate(zip(splits, split_colors)):
        y = [100 * f(next(r["total_attrition_rate"] for r in qc
                          if r["scene"] == s and r["subset"] == split)) for s in scenarios]
        ax.bar(x + (i - 1) * width, y, width, label={"train": "Train", "val": "Validation", "test": "Test"}[split], color=color)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("Total attrition [%]")
    ax.set_ylim(0, 25)
    ax.legend(frameon=False, ncols=3, fontsize=6.7, loc="upper left")
    panel(ax, "a", "Solver + QC selection")

    mbe = rows(AUDIT / "E10" / "source_stats" / "mbe_vs_error_scatter.csv")
    ax = axes[1]
    for scenario, label in zip(scenarios, labels):
        sub = [r for r in mbe if r["scenario"] == scenario]
        ax.scatter([f(r["mbe_w_mean_final"]) for r in sub],
                   [f(r["c_l2_mean"]) for r in sub], s=16, alpha=0.75,
                   color=SCENARIO_COLORS[scenario], label=label, edgecolor="white", linewidth=0.25)
    ax.axvline(10, color=COLORS["red"], lw=1, ls="--", label="10% trigger")
    ax.set(xlabel="Terminal water MBE [%]", ylabel="Mean concentration relative $L_2$")
    ax.text(0.98, 0.96, r"Spearman $\rho=0.588$", ha="right", va="top", transform=ax.transAxes, fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    panel(ax, "b", "Run-level MBE association")

    trigger = [r for r in rows(AUDIT / "E10" / "threshold_analysis.csv")
               if r["level"] == "run" and r["scope"] == "all"]
    trigger.sort(key=lambda r: f(r["mbe_threshold_pct"]))
    thresholds = [f(r["mbe_threshold_pct"]) for r in trigger]
    ax = axes[2]
    ax.plot(thresholds, [100 * f(r["precision_high_in_flagged"]) for r in trigger],
            marker="o", color=COLORS["blue"], label="Precision")
    ax.plot(thresholds, [100 * f(r["recall_high_flagged"]) for r in trigger],
            marker="s", color=COLORS["green"], label="Recall")
    ax.plot(thresholds, [100 * f(r["flag_rate"]) for r in trigger],
            marker="^", color=COLORS["orange"], label="Flag rate")
    ax.set(xlabel="Water-MBE trigger [%]", ylabel="Rate [%]", xticks=thresholds, ylim=(0, 105))
    ax.legend(frameon=False, loc="lower left")
    panel(ax, "c", "Trigger trade-off")
    save(fig, "fig6_reliability_diagnostics")


def plot_efficiency() -> None:
    data = rows(E11 / "bench_hydrus_multiprocess.csv")
    proc = np.array([f(r["processes"]) for r in data])
    thr = np.array([f(r["attempted_throughput_per_s_mean"]) for r in data])
    thr_sd = np.array([f(r["attempted_throughput_per_s_std"]) for r in data])
    speed = np.array([f(r["speedup_vs_1_process"]) for r in data])
    eff = np.array([100 * f(r["parallel_efficiency"]) for r in data])
    fixed_cost = 4564.0
    inference = 0.000942
    breakeven = fixed_cost / (1.0 / thr - inference) / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(7.55, 2.45), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(proc, thr, yerr=thr_sd, color=COLORS["blue"], marker="o", capsize=3, lw=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(proc, [str(int(x)) for x in proc])
    ax.set(xlabel="HYDRUS processes", ylabel="Attempted throughput [s$^{-1}$]", ylim=(0, 5.1))
    panel(ax, "a", "Measured throughput")

    ax = axes[1]
    ax.plot(proc, speed, color=COLORS["orange"], marker="s", lw=1.6, label="Measured")
    ax.plot(proc, proc, color=COLORS["grey"], ls="--", lw=1, label="Ideal linear")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(proc, [str(int(x)) for x in proc])
    ax.set(xlabel="HYDRUS processes", ylabel="Speed-up", ylim=(0.8, 20))
    ax.legend(frameon=False, loc="upper left")
    panel(ax, "b", "Scaling saturation")

    ax = axes[2]
    bars = ax.bar(np.arange(len(proc)), breakeven, color=COLORS["green"], width=0.66)
    ax.set_xticks(np.arange(len(proc)), [str(int(x)) for x in proc])
    ax.set(xlabel="HYDRUS processes", ylabel="GPU break-even [10$^3$ evaluations]", ylim=(0, 24))
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.1)
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(proc)), eff, color=COLORS["red"], marker="o", lw=1.2, label="Efficiency")
    ax2.set_ylabel("Parallel efficiency [%]", color=COLORS["red"])
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", colors=COLORS["red"], labelsize=7.4)
    ax2.spines["right"].set_visible(True)
    panel(ax, "c", "Break-even and efficiency")
    save(fig, "fig7_computational_scaling")


def plot_bound_sensitivity() -> None:
    data = rows(AUDIT / "E2" / "sigma_sensitivity_summary.csv")
    keep = [r for r in data if r["label"] != "paper_g4_regshift"]
    settings = [(0.1, 0.1), (0.2, 0.15), (0.5, 0.3)]
    scenarios = ["iid", "ood_peak", "ood_peak_time"]
    fig, ax = plt.subplots(figsize=(5.8, 3.55), constrained_layout=True)
    x = np.arange(len(settings))
    width = 0.24
    for i, scenario in enumerate(scenarios):
        means, sds = [], []
        for ss, sd in settings:
            r = next(r for r in keep if r["scenario"] == scenario and f(r["sigma_scale"]) == ss and f(r["sigma_shift"]) == sd)
            means.append(f(r["c_rel_l2_median__avg"]))
            sds.append(f(r["c_rel_l2_median__std"]))
        ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=2,
               label={"iid": "IID", "ood_peak": "OOD peak", "ood_peak_time": "OOD peak-time"}[scenario],
               color=SCENARIO_COLORS[scenario])
    ax.set_xticks(x, [f"({ss:.2f}, {sd:.2f})" for ss, sd in settings])
    ax.set(xlabel=r"Bound pair $(\sigma_s,\sigma_\delta)$", ylabel="Median concentration relative $L_2$", ylim=(0, 0.105))
    ax.legend(frameon=False, ncols=3, loc="upper center")
    save(fig, "figS_bound_sensitivity")


def plot_fno_validation() -> None:
    ranking = rows(AUDIT / "E3" / "search_ranking.csv")
    final = rows(AUDIT / "E3" / "final_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    ax = axes[0]
    params = np.array([f(r["model_params_M"]) for r in ranking])
    err = np.array([f(r["c_rel_l2_mean"]) for r in ranking])
    runtime = np.array([f(r["training_time_min"]) for r in ranking])
    sc = ax.scatter(params, err, c=runtime, cmap="viridis", s=34, alpha=0.85)
    best = next(r for r in ranking if r["rank"] == "1")
    ax.scatter([f(best["model_params_M"])], [f(best["c_rel_l2_mean"])], marker="*", s=130,
               color=COLORS["red"], edgecolor="white", linewidth=0.6, label="Selected")
    ax.set(xlabel="Parameters [million]", ylabel="IID mean concentration relative $L_2$", xscale="log")
    ax.legend(frameon=False)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Training time [min]")
    panel(ax, "a", "Padded FNO search")

    ax = axes[1]
    scenarios = ["iid", "ood_peak", "ood_peak_time"]
    labels = ["IID", "OOD peak", "OOD peak-time"]
    values, errors = [], []
    for scenario in scenarios:
        subset = [f(r["c_rel_l2_median"]) for r in final if r["scenario"] == scenario]
        values.append(float(np.mean(subset)))
        errors.append(float(np.std(subset, ddof=1)))
    bars = ax.bar(np.arange(3), values, yerr=errors, capsize=3,
                  color=[SCENARIO_COLORS[s] for s in scenarios], width=0.66)
    ax.set_xticks(np.arange(3), labels, rotation=12)
    ax.set(ylabel="Median concentration relative $L_2$", ylim=(0, 0.17))
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=7.4)
    panel(ax, "b", "Three-seed final replication")
    save(fig, "figS_fno_validation")


def plot_training_audit() -> None:
    data = rows(FROZEN / "training_manifest_detailed.csv")
    groups = [
        "Standard DeepONet, high",
        "Unbounded Shift, high",
        "Bounded RegShift, high",
        "FNN, high",
    ]
    labels = ["Standard\nDeepONet", "Unbounded\nShift", "Bounded\nRegShift", "FNN"]
    colors = [COLORS["grey"], COLORS["purple"], COLORS["green"], COLORS["blue"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    for ax, key, ylabel, scale in [
        (axes[0], "total_epochs", "Total archived epochs", 1.0),
        (axes[1], "wall_minutes", "Wall time [min]", 1.0),
    ]:
        values, errors = [], []
        for group in groups:
            arr = np.array([f(r[key]) for r in data if r["group"] == group]) * scale
            values.append(float(np.mean(arr)))
            errors.append(float(np.std(arr, ddof=1)))
        bars = ax.bar(np.arange(len(groups)), values, yerr=errors, capsize=3,
                      color=colors, width=0.68)
        ax.set_xticks(np.arange(len(groups)), labels)
        ax.set_ylabel(ylabel)
        ax.bar_label(bars, labels=[f"{v:.0f}" if key == "total_epochs" else f"{v:.1f}" for v in values],
                     padding=4, fontsize=7.4)
    panel(axes[0], "a", "Observed optimisation exposure")
    panel(axes[1], "b", "Architecture-dependent wall time")
    save(fig, "figS_training_audit")


def plot_mar_feasibility() -> None:
    data = rows(AUDIT / "E7" / "validation_table.csv")
    strata = ["head", "high_mbe", "mid", "tail"]
    labels = ["Top-ranked", "High MBE", "Middle", "Tail"]
    colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), constrained_layout=True)

    ax = axes[0]
    for stratum, label, color in zip(strata, labels, colors):
        sub = [r for r in data if r["stratum"] == stratum]
        ax.scatter([f(r["tb_hydrus_h"]) for r in sub], [f(r["tb_surrogate_h"]) for r in sub],
                   s=20, alpha=0.72, color=color, label=label, edgecolor="white", linewidth=0.25)
    ax.plot([15, 40], [15, 40], color="black", ls="--", lw=0.9)
    ax.axhline(36, color=COLORS["grey"], lw=0.8)
    ax.axvline(36, color=COLORS["grey"], lw=0.8)
    ax.set(xlabel="HYDRUS breakthrough time [h]", ylabel="Surrogate breakthrough time [h]",
           xlim=(15, 40), ylim=(15, 40))
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.text(0.98, 0.04, r"Spearman $\rho=0.222$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.4)
    panel(ax, "a", "Breakthrough-time confirmation")

    ax = axes[1]
    for stratum, label, color in zip(strata, labels, colors):
        sub = [r for r in data if r["stratum"] == stratum]
        ax.scatter([f(r["peak_c_hydrus"]) for r in sub], [f(r["peak_c_surrogate"]) for r in sub],
                   s=20, alpha=0.72, color=color, label=label, edgecolor="white", linewidth=0.25)
    lim = (0.35, 0.82)
    ax.plot(lim, lim, color="black", ls="--", lw=0.9)
    ax.set(xlabel="HYDRUS bottom peak concentration", ylabel="Surrogate bottom peak concentration",
           xlim=lim, ylim=lim)
    ax.text(0.98, 0.04, r"Spearman $\rho=0.164$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.4)
    panel(ax, "b", "Peak-concentration confirmation")
    save(fig, "figS_mar_feasibility")


def plot_qc_selection() -> None:
    qc = rows(AUDIT / "E4" / "qc_counts.csv")
    ks = rows(AUDIT / "E4" / "ks_removed_vs_retained.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)

    ax = axes[0]
    labels = [{"train": "Train", "val": "Val.", "test": "Test"}[r["subset"]] for r in qc]
    sim_failed = np.array([100 * f(r["n_sim_failed"]) / f(r["n_csv"]) for r in qc])
    qc_failed = np.array([100 * f(r["n_qc_failed"]) / f(r["n_csv"]) for r in qc])
    retained = 100 - sim_failed - qc_failed
    x = np.arange(len(qc))
    ax.bar(x, retained, color=COLORS["green"], label="Retained")
    ax.bar(x, sim_failed, bottom=retained, color=COLORS["grey"], label="Solver failed")
    ax.bar(x, qc_failed, bottom=retained + sim_failed, color=COLORS["orange"], label="QC failed")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set(ylabel="Share of initial pool [%]", ylim=(0, 100))
    ax.legend(frameon=False, ncols=3, fontsize=7, loc="lower center")
    for xpos in (2.5, 5.5):
        ax.axvline(xpos, color=COLORS["grey"], lw=0.7, alpha=0.5)
    for xpos, label in zip((1, 4, 7), ("IID", "OOD peak", "OOD peak-time")):
        ax.text(xpos, 1.015, label, ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=7.3)
    panel(ax, "a", "Subset-specific attrition")

    ax = axes[1]
    variables = ["q_top_peak", "q_top_peak_time", "K_s", "n_vg", "alpha"]
    var_labels = [r"$q_{top}$ peak", "Peak time", r"$K_s$", r"$n$", r"$\alpha$"]
    width = 0.24
    scenarios = ["iid", "ood_peak", "ood_peak_time"]
    for i, scenario in enumerate(scenarios):
        y = [f(next(r["ks_D"] for r in ks if r["scene"] == scenario and r["variable"] == var)) for var in variables]
        ax.bar(np.arange(len(variables)) + (i - 1) * width, y, width,
               color=SCENARIO_COLORS[scenario], label={"iid": "IID", "ood_peak": "OOD peak", "ood_peak_time": "OOD peak-time"}[scenario])
    ax.set_xticks(np.arange(len(variables)), var_labels)
    ax.set(ylabel="Removed-vs-retained KS statistic", ylim=(0, 0.55))
    ax.legend(frameon=False, fontsize=7)
    panel(ax, "b", "QC changes hydraulic marginals")
    save(fig, "figS_qc_selection")


def plot_rainfall_stress() -> None:
    stats = rows(AUDIT / "E5" / "stats_compare_table.csv")
    acf = [r for r in stats if r["metric"].startswith("acf_lag")]
    events = rows(AUDIT / "E5" / "events_raw.csv")
    realrain = rows(AUDIT / "E5" / "e5_results_realrain_s42.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65), constrained_layout=True)

    ax = axes[0]
    lags = np.arange(1, len(acf) + 1)
    for key, label, color, marker in [
        ("train_pool", "Training pool", COLORS["blue"], "o"),
        ("real_events", "Observed events", COLORS["orange"], "s"),
    ]:
        means = [mean_sd(r[key])[0] for r in acf]
        sds = [mean_sd(r[key])[1] for r in acf]
        ax.errorbar(lags, means, yerr=sds, color=color, marker=marker, ms=3, capsize=2, label=label)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set(xlabel="Lag [h]", ylabel="Autocorrelation", xticks=lags, ylim=(-0.35, 1.08))
    ax.legend(frameon=False)
    panel(ax, "a", "Autocorrelation envelope")

    ax = axes[1]
    for r in events:
        rain = [f(r[f"r_mm_{i:02d}"]) for i in range(48)]
        ax.plot(np.arange(48), rain, lw=0.9, alpha=0.62)
    ax.set(xlabel="Event time [h]", ylabel="Observed rainfall [mm h$^{-1}$]", xlim=(0, 47))
    panel(ax, "b", "Nine observed 48-h events")

    ax = axes[2]
    order = ["DeepONet", "Shift-DeepONet", "RegShift(修复代码重跑, 真有界)"]
    short = ["Standard", "Unbounded", "Bounded"]
    values = {r["model"]: f(r["c_rel_l2"]) for r in realrain}
    bars = ax.bar(np.arange(3), [values[k] for k in order], color=[COLORS["grey"], COLORS["purple"], COLORS["green"]])
    ax.set_xticks(np.arange(3), short)
    ax.set(ylabel="Mean concentration relative $L_2$", ylim=(0, 3.25))
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)
    panel(ax, "c", "Single-seed temporal-shape stress")
    save(fig, "figS_rainfall_stress")


def plot_mbe_trigger() -> None:
    mbe = rows(AUDIT / "E10" / "source_stats" / "mbe_vs_error_scatter.csv")
    trigger = [r for r in rows(AUDIT / "E10" / "threshold_analysis.csv")
               if r["level"] == "run" and r["scope"] == "all"]
    trigger.sort(key=lambda r: f(r["mbe_threshold_pct"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)

    ax = axes[0]
    for scenario, label in [("iid", "IID"), ("ood_peak", "OOD peak"), ("ood_peak_time", "OOD peak-time")]:
        sub = [r for r in mbe if r["scenario"] == scenario]
        ax.scatter([f(r["mbe_w_mean_final"]) for r in sub], [f(r["c_l2_mean"]) for r in sub],
                   s=18, alpha=0.75, color=SCENARIO_COLORS[scenario], label=label,
                   edgecolor="white", linewidth=0.25)
    ax.axvline(10, color=COLORS["red"], ls="--", lw=1, label="10% trigger")
    ax.set(xlabel="Terminal water MBE [%]", ylabel="Mean concentration relative $L_2$")
    ax.legend(frameon=False, fontsize=7)
    ax.text(0.98, 0.97, r"All runs: $\rho=0.588$", ha="right", va="top", transform=ax.transAxes)
    panel(ax, "a", "MBE is associated with field error")

    ax = axes[1]
    thresholds = np.array([f(r["mbe_threshold_pct"]) for r in trigger])
    metrics = [
        ("precision_high_in_flagged", "Precision", COLORS["blue"]),
        ("recall_high_flagged", "Recall", COLORS["green"]),
        ("flag_rate", "Flag rate", COLORS["orange"]),
        ("residual_high_rate_in_passed", "Residual high-error rate", COLORS["red"]),
    ]
    for key, label, color in metrics:
        ax.plot(thresholds, [100 * f(r[key]) for r in trigger], marker="o", color=color, label=label)
    ax.set(xlabel="Water-MBE trigger [%]", ylabel="Rate [%]", xticks=thresholds, ylim=(0, 105))
    ax.legend(frameon=False, fontsize=7)
    panel(ax, "b", "Trigger sensitivity")
    save(fig, "figS_mbe_trigger")


def main() -> None:
    style()
    plot_applicability()
    plot_reliability()
    plot_efficiency()
    plot_bound_sensitivity()
    plot_fno_validation()
    plot_training_audit()
    plot_qc_selection()
    plot_rainfall_stress()
    plot_mar_feasibility()
    plot_mbe_trigger()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir.resolve()
    main()
