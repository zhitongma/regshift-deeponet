#!/usr/bin/env python3
"""Generate corrected field, case, and regime figures from frozen evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = ("iid", "ood_peak", "ood_peak_time")
SCENARIO_LABELS = {
    "iid": "IID",
    "ood_peak": "OOD (peak)",
    "ood_peak_time": "OOD (peak-time)",
}
SCENARIO_COLORS = {
    "iid": "#4477AA",
    "ood_peak": "#EE7733",
    "ood_peak_time": "#009988",
}
MODEL_COLORS = {
    "regshift": "#0072B2",
    "shift": "#D55E00",
    "fnn": "#737B87",
    "reference": "#172033",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.7,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 8.1,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".png"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def add_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color="#D8DEE8", linewidth=0.48, alpha=0.65)
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color="#172033",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.1},
    )


def concentration_front(field: np.ndarray, z: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Deepest node above threshold for every time; zero if absent."""
    mask = field >= threshold
    indices = np.where(mask, np.arange(z.size)[:, None], -1).max(axis=0)
    return np.where(indices >= 0, z[np.maximum(indices, 0)], 0.0)


def plot_seed_band(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    mean = values.mean(axis=0)
    low = values.min(axis=0)
    high = values.max(axis=0)
    ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.45, linestyle=linestyle, label=label)


def generate_process_chain(data: np.lib.npyio.NpzFile, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.25, 6.65),
        sharex="col",
        gridspec_kw={"height_ratios": (0.78, 1.0, 1.0)},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.075, top=0.88, wspace=0.18, hspace=0.14)
    for col, scenario in enumerate(SCENARIOS):
        z = data[f"{scenario}_z"]
        t = data[f"{scenario}_t"]
        q = data[f"{scenario}_q_top"]
        selected = int(data[f"{scenario}_selected_index"])
        c_ref = data[f"{scenario}_c_ref"]
        c_reg = data[f"{scenario}_regshift_c_pred"]
        c_fnn = data[f"{scenario}_fnn_c_pred"]

        ax = axes[0, col]
        q_time = t[: q.size]
        ax.step(q_time, q, where="post", color=SCENARIO_COLORS[scenario], linewidth=1.7)
        ax.fill_between(
            q_time,
            0,
            q,
            step="post",
            color=SCENARIO_COLORS[scenario],
            alpha=0.13,
        )
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"{SCENARIO_LABELS[scenario]}  ·  case {selected + 1}",
            pad=7,
        )
        add_grid(ax, "y")
        add_panel_label(ax, f"({'abc'[col]})")

        ax = axes[1, col]
        ref_front = concentration_front(c_ref, z)
        reg_front = np.stack([concentration_front(field, z) for field in c_reg])
        fnn_front = np.stack([concentration_front(field, z) for field in c_fnn])
        ax.plot(t, ref_front, color=MODEL_COLORS["reference"], linewidth=1.8, label="HYDRUS")
        plot_seed_band(ax, t, reg_front, MODEL_COLORS["regshift"], "RegShift")
        plot_seed_band(ax, t, fnn_front, MODEL_COLORS["fnn"], "FNN", "--")
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(float(z.max()), float(z.min()))
        ax.set_yticks(np.arange(0, 101, 20))
        add_grid(ax)
        add_panel_label(ax, f"({'def'[col]})")

        ax = axes[2, col]
        ax.plot(t, c_ref[-1], color=MODEL_COLORS["reference"], linewidth=1.8, label="HYDRUS")
        plot_seed_band(ax, t, c_reg[:, -1, :], MODEL_COLORS["regshift"], "RegShift")
        plot_seed_band(ax, t, c_fnn[:, -1, :], MODEL_COLORS["fnn"], "FNN", "--")
        ax.axhline(0.05, color="#A4ABB5", linewidth=0.75, linestyle=(0, (1.5, 1.5)))
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time [h]")
        add_grid(ax)
        add_panel_label(ax, f"({'ghi'[col]})")

    axes[0, 0].set_ylabel(r"$q_{\mathrm{top}}$ [cm h$^{-1}$]")
    axes[1, 0].set_ylabel("Front depth [cm]\n" + r"($c \geq 0.01$)")
    axes[2, 0].set_ylabel(r"Outlet $c(100\,\mathrm{cm},t)$")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
        handlelength=2.6,
        columnspacing=1.5,
    )
    return save_figure(fig, output_dir, "fig4_process_chain_corrected")


def generate_case_study(data: np.lib.npyio.NpzFile, output_dir: Path) -> list[Path]:
    scenario = "iid"
    seed_position = int(np.where(data["seeds"] == 42)[0][0])
    selected = int(data[f"{scenario}_selected_index"])
    z = data[f"{scenario}_z"]
    t = data[f"{scenario}_t"]
    h_ref = data[f"{scenario}_h_ref"]
    c_ref = data[f"{scenario}_c_ref"]
    h_reg = data[f"{scenario}_regshift_h_pred"][seed_position]
    c_reg = data[f"{scenario}_regshift_c_pred"][seed_position]
    h_fnn = data[f"{scenario}_fnn_h_pred"][seed_position]
    c_fnn = data[f"{scenario}_fnn_c_pred"][seed_position]
    extent = [float(t.min()), float(t.max()), float(z.max()), float(z.min())]

    fig = plt.figure(figsize=(7.25, 4.55), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=(1.0, 1.0, 1.0, 0.045, 1.0, 0.045),
        wspace=0.08,
        hspace=0.08,
    )
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        axes[row, 0] = fig.add_subplot(grid[row, 0])
        axes[row, 1] = fig.add_subplot(grid[row, 1])
        axes[row, 2] = fig.add_subplot(grid[row, 2])
        axes[row, 3] = fig.add_subplot(grid[row, 4])
    state_color_axes = [fig.add_subplot(grid[row, 3]) for row in range(2)]
    difference_color_axes = [fig.add_subplot(grid[row, 5]) for row in range(2)]
    rows = [
        (h_ref, h_reg, h_fnn, "$h$ [cm]", "Pressure head", "cividis"),
        (c_ref, c_reg, c_fnn, "$c$ [-]", "Concentration", "magma"),
    ]
    column_titles = ("HYDRUS", "RegShift", "FNN", "Absolute-error contrast")

    for row, (ref, reg, fnn, colorbar_label, row_label, cmap) in enumerate(rows):
        vmin = float(min(ref.min(), reg.min(), fnn.min()))
        vmax = float(max(ref.max(), reg.max(), fnn.max()))
        fields = ((ref, "HYDRUS"), (reg, "RegShift"), (fnn, "FNN"))
        row_images = []
        for col, (field, label) in enumerate(fields):
            im = axes[row, col].imshow(
                field,
                aspect="auto",
                origin="upper",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                rasterized=True,
            )
            if row == 0:
                axes[row, col].set_title(column_titles[col], pad=6)
            row_images.append(im)
            add_panel_label(axes[row, col], f"({'abcd'[col] if row == 0 else 'efgh'[col]})")

        advantage = np.abs(fnn - ref) - np.abs(reg - ref)
        bound = float(np.max(np.abs(advantage))) or 1.0
        advantage_image = axes[row, 3].imshow(
            advantage,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
            interpolation="nearest",
            rasterized=True,
        )
        if row == 0:
            axes[row, 3].set_title(column_titles[3], pad=6)
        add_panel_label(axes[row, 3], f"({'d' if row == 0 else 'h'})")
        axes[row, 0].set_ylabel(f"{row_label}\nDepth [cm]")
        shared = fig.colorbar(row_images[-1], cax=state_color_axes[row])
        shared.set_label(colorbar_label)
        diff_bar = fig.colorbar(advantage_image, cax=difference_color_axes[row])
        diff_bar.set_label("Error contrast" + (" [cm]" if row == 0 else " [-]"))

    for row in range(2):
        for col in range(4):
            ax = axes[row, col]
            if row == 1:
                ax.set_xlabel("Time [h]")
            else:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)
            ax.set_xlim(float(t.min()), float(t.max()))
            ax.set_ylim(float(z.max()), float(z.min()))
    fig.suptitle(
        f"IID  ·  representative case {selected + 1}  ·  seed 42",
        fontsize=9.4,
        fontweight="semibold",
    )
    return save_figure(fig, output_dir, "fig6_case_study_corrected")


def load_regime_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["n_samples"] = int(row["n_samples"])
        row["c_mae"] = float(row["c_mae"])
    return rows


def paired_gain_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, float]] = defaultdict(dict)
    metadata: dict[tuple, dict] = {}
    for row in rows:
        key = (row["scenario"], row["seed"], row["group_type"], row["group"])
        grouped[key][row["model"]] = row["c_mae"]
        metadata[key] = row
    gains = []
    for key, values in grouped.items():
        if not {"regshift", "shift"}.issubset(values):
            continue
        scenario, seed, group_type, group = key
        gain = 100.0 * (values["shift"] - values["regshift"]) / values["shift"]
        gains.append(
            {
                "scenario": scenario,
                "seed": seed,
                "group_type": group_type,
                "group": group,
                "gain_percent": gain,
                "regshift_c_mae": values["regshift"],
                "shift_c_mae": values["shift"],
                "n_samples": metadata[key]["n_samples"],
            }
        )
    return gains


def summarise_gains(gains: list[dict]) -> list[dict]:
    values: dict[tuple, list[float]] = defaultdict(list)
    sample_counts: dict[tuple, int] = {}
    for row in gains:
        key = (row["scenario"], row["group_type"], row["group"])
        values[key].append(row["gain_percent"])
        sample_counts[key] = row["n_samples"]
    summary = []
    for (scenario, group_type, group), sample in sorted(values.items()):
        arr = np.asarray(sample, dtype=np.float64)
        summary.append(
            {
                "scenario": scenario,
                "group_type": group_type,
                "group": group,
                "n_seeds": arr.size,
                "n_samples_per_seed": sample_counts[(scenario, group_type, group)],
                "gain_mean_percent": float(arr.mean()),
                "gain_sample_sd_percent": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            }
        )
    return summary


def lookup_summary(summary: list[dict], scenario: str, group_type: str, group: str) -> tuple[float, float]:
    for row in summary:
        if (
            row["scenario"] == scenario
            and row["group_type"] == group_type
            and row["group"] == group
        ):
            return row["gain_mean_percent"], row["gain_sample_sd_percent"]
    raise KeyError((scenario, group_type, group))


def plot_gain_forest(
    ax: plt.Axes,
    summary: list[dict],
    group_type: str,
    groups: tuple[str, ...],
    panel: str,
) -> None:
    y = np.arange(len(groups), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    markers = ("o", "s", "^")
    for position, scenario in enumerate(SCENARIOS):
        values, errors = zip(
            *(lookup_summary(summary, scenario, group_type, group) for group in groups)
        )
        ax.errorbar(
            values,
            y + offsets[position],
            xerr=errors,
            fmt=markers[position],
            color=SCENARIO_COLORS[scenario],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.0,
            elinewidth=1.0,
            capsize=2.2,
            capthick=1.0,
            linewidth=0,
            label=SCENARIO_LABELS[scenario],
        )
    ax.axvline(0, color="#172033", linewidth=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlim(-65, 45)
    ax.set_xticks((-60, -40, -20, 0, 20, 40))
    ax.set_ylim(-0.55, len(groups) - 0.45)
    ax.invert_yaxis()
    ax.set_xlabel("Concentration-MAE gain [%]")
    ax.set_title(panel, loc="left", pad=5)
    add_grid(ax, "x")


def generate_regime_figure(
    rows: list[dict],
    heatmaps: np.lib.npyio.NpzFile,
    output_dir: Path,
    evidence_dir: Path,
) -> tuple[list[Path], Path]:
    gains = paired_gain_rows(rows)
    summary = summarise_gains(gains)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "corrected_regime_gain_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.55), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.93, bottom=0.095, top=0.86, wspace=0.28, hspace=0.30)
    plot_gain_forest(
        axes[0, 0],
        summary,
        "depth",
        ("0-20 cm", "20-60 cm", "60-100 cm"),
        "(a) Depth regime",
    )
    plot_gain_forest(
        axes[0, 1],
        summary,
        "time",
        ("0-16 h", "16-32 h", "32-48 h"),
        "(b) Time regime",
    )

    plot_gain_forest(
        axes[1, 0],
        summary,
        "intensity",
        ("Low", "Mid", "High"),
        "(c) Event-intensity regime",
    )

    scenario = "ood_peak_time"
    z = heatmaps[f"{scenario}_z"]
    t = heatmaps[f"{scenario}_t"]
    field = heatmaps[f"{scenario}_shift_minus_regshift_abs_error"].mean(axis=0)
    bound = float(np.percentile(np.abs(field), 99.0)) or 1.0
    image = axes[1, 1].imshow(
        field,
        aspect="auto",
        origin="upper",
        extent=[float(t.min()), float(t.max()), float(z.max()), float(z.min())],
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        interpolation="nearest",
        rasterized=True,
    )
    axes[1, 1].set_xlabel("Time [h]")
    axes[1, 1].set_ylabel("Depth [cm]")
    axes[1, 1].set_title("(d) Peak-time shift: local error difference")
    colorbar = fig.colorbar(image, ax=axes[1, 1], shrink=0.86, pad=0.02, extend="both")
    colorbar.set_label("Shift - RegShift absolute error")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        handletextpad=0.55,
        columnspacing=1.35,
    )
    outputs = save_figure(fig, output_dir, "fig7_regime_corrected")
    return outputs, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()

    process_path = args.data_dir / "corrected_process_and_case_fields.npz"
    regime_path = args.data_dir / "corrected_regime_summary.csv"
    heatmap_path = args.data_dir / "corrected_regime_heatmaps.npz"
    process = np.load(process_path)
    heatmaps = np.load(heatmap_path)
    rows = load_regime_rows(regime_path)

    outputs = []
    outputs.extend(generate_process_chain(process, args.output_dir))
    outputs.extend(generate_case_study(process, args.output_dir))
    regime_outputs, summary_path = generate_regime_figure(
        rows, heatmaps, args.output_dir, args.data_dir
    )
    outputs.extend(regime_outputs)

    manifest = {
        "generator": Path(__file__).name,
        "inputs": {
            path.name: sha256_file(path)
            for path in (process_path, regime_path, heatmap_path)
        },
        "derived_summary": {
            summary_path.name: sha256_file(summary_path),
        },
        "outputs": {path.name: sha256_file(path) for path in outputs},
    }
    manifest_path = args.output_dir / "corrected_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "FIGURES_OK", **manifest}, indent=2))


if __name__ == "__main__":
    main()
