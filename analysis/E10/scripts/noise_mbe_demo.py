#!/usr/bin/env python3
"""
E10 步骤 1 — MBE 溯源受控噪声演示（回应 R1-7 的"溯源"半问）。

原理链条（全部用主仓库原公式，见 mbe_common.py 的出处注释）：
    给 HYDRUS 参考 h 场加已知幅度的相对噪声
        -> theta = vG(h)（van Genuchten 非线性映射，陡峭段 dθ/dh 大，放大 h 误差）
        -> 储量 S(t) = ∫θ dz（空间积分：白噪声相消，相关误差累积）
        -> MBE = |ΔS - 累计边界通量| / max(|累计边界通量|,1) × 100%（B2 原始定义）
    从而定量回答：0.08–0.2 量级的场相对误差是否足以解释论文表 5 中 5–17% 的水量 MBE。

三种噪声结构（真实代理模型误差介于 smooth 与 bias 之间）：
    white  : 逐点独立相对噪声 h·(1+ε·ξ)，空间积分大量相消 —— MBE 放大最弱（下界）；
    smooth : 空间/时间相关的平滑噪声（盒式核平滑后归一化）—— 接近代理模型的场误差结构；
    bias   : 全场同号相对偏差 h·(1±ε) —— 完全相关（上界）。

运行位置：集群，或本机在 test.npz/scaler.npz 物化（brctl download）之后。
numpy-only；matplotlib 可选（缺失自动降级为只出 CSV）。

用法示例：
    python3 noise_mbe_demo.py \
        --run-dir "$REPO/experiments/qtop_func/runs/iid_medium_v1" \
        --scenario iid --out-dir <包目录>/outputs
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbe_common import (  # noqa: E402
    get_raw_params,
    mass_balance_error_water,
    relative_l2,
    safe_load_npz,
    trapz,
    vg_theta,
    write_csv,
)


def make_noise(mode, shape, rng):
    """单位标准差的噪声场 (n_z, n_t)。"""
    n_z, n_t = shape
    if mode == "white":
        return rng.standard_normal(shape)
    if mode == "smooth":
        xi = rng.standard_normal(shape)
        # 盒式核平滑（numpy-only）：z 方向窗宽 ~n_z/4，t 方向 ~n_t/4
        wz = max(3, n_z // 4)
        wt = max(3, n_t // 4)
        kz = np.ones(wz) / wz
        kt = np.ones(wt) / wt
        xi = np.apply_along_axis(lambda v: np.convolve(v, kz, mode="same"), 0, xi)
        xi = np.apply_along_axis(lambda v: np.convolve(v, kt, mode="same"), 1, xi)
        std = xi.std()
        return xi / (std + 1e-12)
    if mode == "bias":
        # 每个样本一个随机符号的全场常数偏差
        return np.full(shape, rng.choice([-1.0, 1.0]))
    raise ValueError(f"未知噪声模式: {mode}")


def main():
    ap = argparse.ArgumentParser(description="E10 受控噪声 -> MBE 放大演示")
    ap.add_argument("--run-dir", required=True,
                    help="数据源 run 目录（其下 data/processed/{test,scaler}.npz）")
    ap.add_argument("--scenario", default="", help="场景标签，仅写入输出 CSV")
    ap.add_argument("--levels", default="0.005,0.01,0.02",
                    help="相对噪声水平列表（0.005=0.5%%）")
    ap.add_argument("--modes", default="white,smooth,bias")
    ap.add_argument("--n-samples", type=int, default=0, help="限制样本数（0=全部）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    proc = os.path.join(args.run_dir, "data", "processed")
    test_path = os.path.join(proc, "test.npz")
    scaler_path = os.path.join(proc, "scaler.npz")

    test_data, err = safe_load_npz(test_path)
    if err:
        sys.exit(f"[E10] 无法读取 {test_path}: {err}")
    scaler, err = safe_load_npz(scaler_path)
    if err:
        sys.exit(f"[E10] 无法读取 {scaler_path}: {err}")

    z = test_data["z"]
    t = test_data["t"]
    n_z, n_t = len(z), len(t)
    dz = float(z[1] - z[0])

    h_raw = test_data["h_raw"]  # (N, n_z*n_t)
    N = h_raw.shape[0]
    if args.n_samples > 0:
        N = min(N, args.n_samples)
    h_ref = h_raw[:N].reshape(N, n_z, n_t)
    water_cum = test_data["water_cum_net"][:N]  # (N, n_t) HYDRUS 累计净边界通量

    params = get_raw_params(test_data, scaler)
    need = ["theta_r", "theta_s", "alpha", "n_vg"]
    missing = [k for k in need if k not in params]
    if missing:
        sys.exit(f"[E10] branch_keys 中缺少 vG 参数列: {missing}")

    rng = np.random.default_rng(args.seed)
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    # --- 参考基线：HYDRUS 自身场对同一 MBE 公式的响应（应接近其 QC 水平） ---
    mbe_ref_final = []
    theta_ref_all = []
    for i in range(N):
        th = vg_theta(h_ref[i], params["theta_r"][i], params["theta_s"][i],
                      params["alpha"][i], params["n_vg"][i])
        theta_ref_all.append(th)
        mbe_ref_final.append(mass_balance_error_water(th, water_cum[i], dz=dz)[-1])
    mbe_ref_final = np.array(mbe_ref_final)

    header = ["scenario", "mode", "eps_rel", "h_rel_l2_mean", "theta_rel_l2_mean",
              "theta_gain", "dstorage_rel_err_mean_pct",
              "mbe_w_final_mean_pct", "mbe_w_final_median_pct",
              "mbe_minus_ref_mean_pct", "amplification_mbe_per_h_l2"]
    rows = []
    # 基线行
    rows.append([args.scenario, "reference(HYDRUS)", 0.0, 0.0, 0.0, "",
                 0.0, f"{np.nanmean(mbe_ref_final):.4f}",
                 f"{np.nanmedian(mbe_ref_final):.4f}", 0.0, ""])

    print(f"[E10] N={N}, n_z={n_z}, n_t={n_t}, dz={dz:.3f}")
    print(f"[E10] HYDRUS 参考场自身 MBE_w(终点) 均值 = {np.nanmean(mbe_ref_final):.3f}%")

    for mode in modes:
        for eps in levels:
            h_errs, th_errs, ds_errs, mbe_fin = [], [], [], []
            for i in range(N):
                xi = make_noise(mode, (n_z, n_t), rng)
                h_noisy = h_ref[i] * (1.0 + eps * xi)
                th_noisy = vg_theta(h_noisy, params["theta_r"][i],
                                    params["theta_s"][i], params["alpha"][i],
                                    params["n_vg"][i])
                th_ref = theta_ref_all[i]

                h_errs.append(relative_l2(h_noisy, h_ref[i]))
                th_errs.append(relative_l2(th_noisy, th_ref))

                # 储量链条: S(t)=∫θdz（与 mass_balance_error_water 内部一致）
                s_noisy = trapz(th_noisy, dx=dz, axis=0)
                s_ref = trapz(th_ref, dx=dz, axis=0)
                ds_noisy = s_noisy - s_noisy[0]
                ds_ref = s_ref - s_ref[0]
                denom = max(abs(ds_ref[-1]), 1.0)
                ds_errs.append(abs(ds_noisy[-1] - ds_ref[-1]) / denom * 100.0)

                mbe = mass_balance_error_water(th_noisy, water_cum[i], dz=dz)
                mbe_fin.append(mbe[-1])

            h_l2 = float(np.mean(h_errs))
            th_l2 = float(np.mean(th_errs))
            mbe_mean = float(np.nanmean(mbe_fin))
            mbe_med = float(np.nanmedian(mbe_fin))
            excess = mbe_mean - float(np.nanmean(mbe_ref_final))
            amp = excess / (h_l2 * 100.0) if h_l2 > 0 else float("nan")
            gain = th_l2 / h_l2 if h_l2 > 0 else float("nan")
            rows.append([args.scenario, mode, eps, f"{h_l2:.5f}", f"{th_l2:.5f}",
                         f"{gain:.3f}", f"{np.mean(ds_errs):.4f}",
                         f"{mbe_mean:.4f}", f"{mbe_med:.4f}",
                         f"{excess:.4f}", f"{amp:.3f}"])
            print(f"[E10] mode={mode:>6s} eps={eps:.3%}: h-L2={h_l2:.4f}, "
                  f"theta-L2={th_l2:.4f} (gain x{gain:.2f}), "
                  f"MBE_w(final) mean={mbe_mean:.2f}% (超出参考 {excess:+.2f}%)")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.scenario or os.path.basename(os.path.normpath(args.run_dir))
    out_csv = os.path.join(args.out_dir, f"noise_mbe_demo_{tag}.csv")
    write_csv(out_csv, header, rows)
    print(f"[E10] 写出 {out_csv}")

    # 可选画图（matplotlib 缺失则跳过）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for mode in modes:
            xs = [float(r[2]) for r in rows if r[1] == mode]
            ys = [float(r[7]) for r in rows if r[1] == mode]
            ax.plot(np.array(xs) * 100, ys, "o-", label=mode)
        ax.axhspan(5, 17, alpha=0.15, color="grey", label="paper MBE 5-17%")
        ax.set_xlabel("relative h-noise level [%]")
        ax.set_ylabel("terminal water MBE [%] (mean)")
        ax.set_title(f"MBE amplification under controlled noise ({tag})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, f"noise_mbe_demo_{tag}.png")
        fig.savefig(out_png, dpi=200)
        print(f"[E10] 写出 {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[E10] matplotlib 不可用或画图失败（{e}），仅输出 CSV")


if __name__ == "__main__":
    main()
