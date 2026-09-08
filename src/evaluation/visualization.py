"""
论文级图表绘制模块

覆盖 B1-B4 全部实验图表, 输出到 figures/ 目录
风格: Matplotlib + Seaborn, 适用于 Journal of Hydrology
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# --- 论文图表全局样式 ---
PAPER_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

MODEL_COLORS = {
    "m1": "#2196F3",
    "m2": "#E53935",
    "m2_data": "#E53935",
    "m2_score": "#FB8C00",
}
MODEL_LABELS = {
    "m1": "M1 (Data only)",
    "m2": "M2 (Flux-constrained)",
    "m2_data": "M2 (Best data)",
    "m2_score": "M2 (Best score)",
}


def get_model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.upper())


def get_model_color(name: str, index: int = 0) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    cmap = plt.get_cmap("tab10")
    return cmap(index % 10)


def get_result_model_names(results: dict) -> list[str]:
    return [name for name in results.keys() if name != "hydrus_reference"]


def get_npz_model_names(exp_dir: Path, prefix: str, suffix: str) -> list[str]:
    model_names = []
    for path in sorted(exp_dir.glob(f"{prefix}_*_{suffix}.npz")):
        stem = path.stem
        model_names.append(stem.removeprefix(f"{prefix}_").removesuffix(f"_{suffix}"))
    return model_names


def apply_style():
    mpl.rcParams.update(PAPER_RC)


# -----------------------------------------------------------------------
# B1: 精度箱线图 + 时空场对比
# -----------------------------------------------------------------------

def plot_b1_boxplot(results: dict, save_path: str):
    """B1 精度箱线图: M1 vs M2 的 rel_L2 分布 (h 和 c)。"""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    model_names = get_result_model_names(results)

    for ax, var, title in zip(axes, ["h", "c"], ["Pressure head $h$", "Concentration $c$"]):
        data, labels, colors = [], [], []
        for idx, name in enumerate(model_names):
            if name in results:
                vals = results[name][var]["rel_l2"]["values"]
                data.append(vals)
                labels.append(get_model_label(name))
                colors.append(get_model_color(name, idx))

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel("Relative $L_2$ error")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_b1_spatiotemporal(h_pred, c_pred, h_ref, c_ref, z, t, idx,
                           model_name, save_path):
    """B1 单个案例的时空场对比 (h 和 c)。"""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    zz, tt = np.meshgrid(t, z)
    fields = [
        (h_ref[idx], "HYDRUS (ref)", axes[0, 0]),
        (h_pred[idx], f"{get_model_label(model_name)} (pred)", axes[0, 1]),
        (np.abs(h_pred[idx] - h_ref[idx]), "Absolute error", axes[0, 2]),
        (c_ref[idx], "HYDRUS (ref)", axes[1, 0]),
        (c_pred[idx], f"{get_model_label(model_name)} (pred)", axes[1, 1]),
        (np.abs(c_pred[idx] - c_ref[idx]), "Absolute error", axes[1, 2]),
    ]

    for row_idx, (field, title, ax) in enumerate(fields):
        im = ax.pcolormesh(zz, tt, field, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Depth [cm]")

    axes[0, 0].set_ylabel("$h$ [cm]\nDepth [cm]")
    axes[1, 0].set_ylabel("$c$ [mg/L]\nDepth [cm]")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------
# B2: 守恒性对比
# -----------------------------------------------------------------------

def plot_b2_mbe(exp_dir: Path, save_path: str):
    """B2 质量平衡误差随时间演化: M1 vs M2。"""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, name in enumerate(get_npz_model_names(exp_dir, "B2", "mbe")):
        fpath = exp_dir / f"B2_{name}_mbe.npz"
        if not fpath.exists():
            continue
        data = np.load(fpath)
        t = data["t"]
        mbe_w = data["mbe_w"]  # (N, n_t)
        mbe_c = data["mbe_c"]

        color = get_model_color(name, idx)
        label = get_model_label(name)

        for ax, mbe, title in zip(axes, [mbe_w, mbe_c],
                                   ["Water MBE", "Solute MBE"]):
            median = np.nanmedian(mbe, axis=0)
            p25 = np.nanpercentile(mbe, 25, axis=0)
            p75 = np.nanpercentile(mbe, 75, axis=0)

            ax.plot(t, median, color=color, label=label, linewidth=1.5)
            ax.fill_between(t, p25, p75, color=color, alpha=0.15)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Mass balance error [%]")
            ax.set_title(title)

    for ax in axes:
        ax.legend()
        ax.axhline(y=5, color="gray", linestyle="--", linewidth=0.8, label="5% threshold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------
# B3: 水文学过程指标
# -----------------------------------------------------------------------

def plot_b3_breakthrough(exp_dir: Path, save_path: str):
    """B3 穿透时间 pred vs ref 散点图。"""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    for idx, name in enumerate(get_npz_model_names(exp_dir, "B3", "hydro")):
        fpath = exp_dir / f"B3_{name}_hydro.npz"
        if not fpath.exists():
            continue
        data = np.load(fpath)
        bt_pred = data["bt_pred"]
        bt_ref = data["bt_ref"]
        valid = np.isfinite(bt_pred) & np.isfinite(bt_ref)

        ax.scatter(bt_ref[valid], bt_pred[valid], s=25, alpha=0.6,
                   color=get_model_color(name, idx), label=get_model_label(name),
                   edgecolors="white", linewidth=0.3)

    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", linewidth=0.8, label="1:1 line")
    ax.set_xlabel("Breakthrough time (HYDRUS) [h]")
    ax.set_ylabel("Breakthrough time (DeepONet) [h]")
    ax.set_title("Breakthrough time prediction")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------
# B4: 计算效率
# -----------------------------------------------------------------------

def plot_b4_cost(results: dict, save_path: str):
    """B4 总计算时间 vs 场景数量曲线。"""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    hydrus_plotted = False
    for idx, name in enumerate(get_result_model_names(results)):
        if name not in results or "cost_curves" not in results[name]:
            continue
        curves = results[name]["cost_curves"]
        n = np.array(curves["n_range"])

        if not hydrus_plotted:
            ax.plot(n, np.array(curves["cost_hydrus"]) / 3600,
                    "k-", linewidth=2, label="HYDRUS-1D")
            hydrus_plotted = True

        ax.plot(n, np.array(curves["cost_deeponet"]) / 3600,
                color=get_model_color(name, idx), linewidth=1.5,
                label=get_model_label(name))

        n_cross = results[name]["n_cross"]
        ax.axvline(x=n_cross, color=get_model_color(name, idx),
                   linestyle=":", linewidth=0.8)
        ax.annotate(f"$N_{{cross}}$={int(n_cross)}",
                    xy=(n_cross, 0), fontsize=8,
                    color=get_model_color(name, idx))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of scenarios $N$")
    ax.set_ylabel("Total computation time [h]")
    ax.set_title("Computational cost comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------
# 训练曲线
# -----------------------------------------------------------------------

def plot_training_curves(hist_paths: dict, save_path: str):
    """训练损失曲线: M1 vs M2。"""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for name, path in hist_paths.items():
        if not Path(path).exists():
            continue
        with open(path) as f:
            hist = json.load(f)["history"]

        color = get_model_color(name)
        label = get_model_label(name)

        train_loss = hist["train_loss"]
        epochs = range(1, len(train_loss) + 1)

        axes[0].semilogy(epochs, train_loss, color=color, alpha=0.7,
                         linewidth=0.8, label=label)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training loss")
        axes[0].set_title("Training loss")

        if "h_rel_l2" in hist and hist["h_rel_l2"]:
            x = hist.get("val_epoch") or np.linspace(1, len(train_loss), len(hist["h_rel_l2"]))
            axes[1].plot(x, hist["h_rel_l2"], color=color, linewidth=1.2,
                         label=f"{label} ($h$)")
            axes[1].plot(x, hist["c_rel_l2"], color=color, linewidth=1.2,
                         linestyle="--", label=f"{label} ($c$)")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative $L_2$ error")
    axes[1].set_title("Validation error")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
