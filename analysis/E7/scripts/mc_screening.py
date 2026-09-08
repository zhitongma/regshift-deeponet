#!/usr/bin/env python3
"""E7 步骤 1 — Monte Carlo / MAR 入渗调度情景筛选 (⚠️ 集群执行, 需 torch + scipy).

回应 R1-8 / R4-8: 用 RegShift m2 代理模型对 10^4 个候选 MAR 入渗调度 q_top(t)
做一次完整的情景筛选 + 风险分析演示, 并为 HYDRUS 验证抽取分层子集。

流程
----
1. 固定一组代表性壤土 (loam) 参数 (Carsel & Parrish 1988 均值, 写死于 LOAM_PARAMS);
   可选 --param-jitter 在均值 ±10% 内做 LHS 抖动 (联合 forcing+参数不确定性)。
2. 生成 N=10^4 个候选调度: 6 控制点在 [min_flux, max_flux] 均匀 LHS 采样,
   线性插值到 48 个小时步 —— 与训练数据完全同一生成逻辑
   (src/data_generation/sampling.py:160-166 `_interpolate_q_top_series`,
    直接 import 复用; 控制点采样对应 sampling.py:221-230)。
3. 加载 RegShift m2 checkpoint, 用 checkpoint run 自己的 config 快照构建模型
   (与 pipeline/06_evaluate.py:37-44,149-153 完全一致的构建/加载路径),
   branch 输入按该 run 的 data/processed/scaler.npz 做 min-max 归一化
   (06_evaluate.py:126,247-259 predict() 的口径), 批量推理得 c(z,t)/h(z,t)。
4. 决策量:
   - 突破时间 t_b: c(L,t) 首次 >= 0.05*c_top (同 src/evaluation/metrics.py:180-201
     breakthrough_time 的定义, c_th_frac=0.05);
   - 48h 内底部峰值浓度 max_t c(L,t);
   - 风险指标 P(t_b < 36h);
   - 代理模型自诊断 MBE: 预测场储量变化 vs "已知/可诊断" 累计边界通量,
     公式同 src/evaluation/metrics.py:82-109 (_mass_balance_error, B2 口径,
     亦即 src/models/losses.py:54-63 _relative_balance_error 的百分数版):
       MBE(t) = |ΔS_pred(t) - cum_net(t)| / max(|cum_net(t)|, 1) * 100
     其中顶部通量 = 设计调度 q_top(t) (精确已知);
     底部通量按自由排水边界估计 q_bot(t) = K(h_pred(L,t))
     (HYDRUS 配置 bot_bc=4 自由排水, src/data_generation/hydrus_runner.py:138-139;
      单位梯度下 q = K(h); K 用 Mualem-VG, 与 src/utils/normalization.py:35-42
      vg_K_torch 同公式的 numpy 实现)。
     溶质: 顶部 = cum_top*c_top (Cauchy 入流, tpulse=T, hydrus_runner.py:142-145);
     底部 = ∫ q_bot * c_pred(L,t) dt (零浓度梯度 → 纯对流出流)。
     自诊断 MBE = max(MBE_w(48h), MBE_c(48h))。全程只用预测场+设计边界,
     不依赖 HYDRUS —— 这正是 R1-7 自诊断准则的应用形态 (与 E10 交叉引用)。
5. 排序任务: 在 t_b > 36h 的可行方案中按总入渗量 q_top_total 降序取 top-50。
6. 分层抽 150 个方案写 HYDRUS 验证参数表 CSV (头部 50 + 中部 40 + 尾部 30 +
   高自诊断 MBE 30, 去重后随机补齐), 格式满足 02_run_hydrus.py:60-67
   (index 列名 sample_id; q_top_00..q_top_47 共 48 列, hydrus_runner.py:172-176)。

用法 (集群):
    python mc_screening.py --repo /path/to/终极版论文项目_lx \
        --ckpt-run-dir $REPO/experiments/qtop_func/runs/g4_regshift_iid_s42 \
        --out-dir $REPO/experiments/qtop_func/runs_revision/e7_mc_screening_v1 \
        [--param-jitter] [--expect-scale 0.3 --expect-shift 0.2 | --expect-unbounded]

⚠️ 有效界限一致性: 历史 g4_regshift_* 的 config 快照把 transform_bound_* 写在
顶层 shift_deeponet: 块 (E0 已核查的历史 bug), 训练时实际不生效 (= 无界)。
本脚本按快照原样构建 → 推理与训练自洽。若 --expect-* 与快照有效值不符则报错退出。
"""

import os
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# 代表性壤土: Carsel & Parrish (1988) USDA "Loam" 均值
#   theta_r=0.078, theta_s=0.43, alpha=0.036 1/cm, n=1.56,
#   K_s=24.96 cm/day = 1.04 cm/h (本仓库单位 cm/h, 见 K_s 上界 29.7 = C&P 砂土
#   712.8 cm/d / 24; configs/qtop_func/qtop_func_iid_medium.yaml:30-33)
#   D_L 非 C&P 参数, 取训练范围 [1,20] cm 的中值 10 cm (README 已声明该假设)
#   c_top=1.0 (相对浓度, 训练范围 [0.1,1.0] 上端点, 在训练支持内)
# 全部落在训练采样范围内 → 纯 forcing 外推为零, 只考察调度空间的筛选能力。
# ---------------------------------------------------------------------------
LOAM_PARAMS = {
    "theta_r": 0.078,
    "theta_s": 0.43,
    "alpha": 0.036,
    "n_vg": 1.56,
    "K_s": 1.04,
    "D_L": 10.0,
}
C_TOP = 1.0

TRAIN_RANGES = {  # 训练 LHS 范围 (qtop_func_iid_medium.yaml parameters 段), 抖动后裁剪用
    "theta_r": (0.034, 0.098),
    "theta_s": (0.36, 0.51),
    "alpha": (0.005, 0.145),
    "n_vg": (1.09, 2.68),
    "K_s": (0.25, 29.7),
    "D_L": (1.0, 20.0),
}

SOIL_KEYS = ["theta_r", "theta_s", "alpha", "n_vg", "K_s", "D_L"]


def parse_args():
    ap = argparse.ArgumentParser(description="E7 MC/MAR 情景筛选 (集群, 需 torch)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="主仓库根目录 (含 src/)")
    ap.add_argument("--ckpt-run-dir", required=True,
                    help="RegShift 训练 run 目录 (含 results/checkpoints/m2_best.pt, "
                         "data/processed/scaler.npz, metadata/config.yaml)")
    ap.add_argument("--ckpt-name", default="m2_best.pt")
    ap.add_argument("--ckpt-config", default=None,
                    help="模型构建用 config (默认 <ckpt-run-dir>/metadata/config.yaml, "
                         "保证与训练时构建一致)")
    ap.add_argument("--out-dir", required=True,
                    help="输出目录 (建议 $REPO/experiments/qtop_func/runs_revision/e7_mc_screening_v1)")
    ap.add_argument("--n-scenarios", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260716, help="MC 场景生成种子")
    ap.add_argument("--param-jitter", action="store_true",
                    help="开启土壤参数 ±jitter-frac LHS 抖动 (默认关闭=固定壤土均值)")
    ap.add_argument("--jitter-frac", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tb-threshold-hours", type=float, default=36.0)
    ap.add_argument("--bt-frac", type=float, default=0.05,
                    help="突破阈值 c_th = bt_frac * c_top (同 metrics.breakthrough_time)")
    ap.add_argument("--n-top", type=int, default=50)
    ap.add_argument("--n-validate", type=int, default=150)
    # 有效界限断言 (三选一; 不给则只打印)
    ap.add_argument("--expect-scale", type=float, default=None)
    ap.add_argument("--expect-shift", type=float, default=None)
    ap.add_argument("--expect-unbounded", action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# numpy 版 Mualem-van Genuchten K(h) — 与 src/utils/normalization.py:25-42
# (vg_Se_torch / vg_K_torch) 逐式对应
# ---------------------------------------------------------------------------
def vg_K_np(h, K_s, alpha, n):
    m = 1.0 - 1.0 / n
    abs_ah = np.abs(alpha * h)
    Se = np.where(h >= 0, 1.0, 1.0 / (1.0 + abs_ah ** n) ** m)
    Se_safe = np.clip(Se, 1e-8, 1.0)
    inner = 1.0 - (1.0 - Se_safe ** (1.0 / m)) ** m
    return K_s * np.sqrt(Se_safe) * inner ** 2


def mass_balance_error_pct(delta_storage, cum_target):
    """与 src/evaluation/metrics.py:82-87 _mass_balance_error 完全同式 (批量版)。"""
    denom = np.maximum(np.abs(cum_target), 1.0)
    mbe = np.abs(delta_storage - cum_target) / denom * 100.0
    mbe[..., 0] = 0.0
    return mbe


def cum_forcing_at(q_series, dt_forcing, t_out):
    """分段常数 forcing 的累计积分, 在输出时刻 t_out 取值 (逐样本)。

    q_series: (N, n_steps); 返回 (N, n_t)。分段常数的累计量是分段线性 → interp 精确。
    """
    n_steps = q_series.shape[1]
    edges = np.arange(n_steps + 1) * dt_forcing
    cum_edges = np.concatenate(
        [np.zeros((q_series.shape[0], 1)), np.cumsum(q_series, axis=1) * dt_forcing],
        axis=1,
    )
    return np.vstack([np.interp(t_out, edges, row) for row in cum_edges])


def cumtrapz_axis1(y, t):
    """手写 cumulative trapezoid (N, n_t) 沿时间轴, 首点为 0。"""
    dt = np.diff(t)[None, :]
    inc = 0.5 * (y[:, 1:] + y[:, :-1]) * dt
    return np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)


def breakthrough_times(c_bottom, t, c_th):
    """c_bottom: (N, n_t)。返回 (N,), 未突破为 np.inf。
    同 src/evaluation/metrics.py:180-201 breakthrough_time 的向量化版。"""
    exceed = c_bottom >= c_th
    any_ex = exceed.any(axis=1)
    first_idx = exceed.argmax(axis=1)
    tb = np.where(any_ex, t[np.clip(first_idx, 0, len(t) - 1)], np.inf)
    return tb


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import torch  # noqa: E402  (集群环境)
    import yaml  # noqa: E402
    from src.models.shift_deeponet import build_shift_deeponet  # noqa: E402
    from src.data_generation.postprocess import DataScaler  # noqa: E402
    from src.data_generation.sampling import _interpolate_q_top_series  # noqa: E402
    from src.utils.normalization import vg_theta  # noqa: E402
    from scipy.stats import qmc  # noqa: E402  (与训练同源的 LHS)

    ckpt_run = Path(args.ckpt_run_dir).resolve()
    cfg_path = Path(args.ckpt_config) if args.ckpt_config else ckpt_run / "metadata" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    # ---------------- 模型构建 + 有效界限自检 ----------------
    model_cfg = cfg["model"]
    assert model_cfg.get("arch") == "shift_deeponet", \
        f"期望 shift_deeponet, 得到 {model_cfg.get('arch')!r}"
    model = build_shift_deeponet(model_cfg)
    eff_scale = float(model.transform_bound_scale)
    eff_shift = float(model.transform_bound_shift)
    print(f"[preflight] 有效界限: scale={eff_scale}, shift={eff_shift} (config={cfg_path})")
    top_sd = cfg.get("shift_deeponet") or {}
    if any(str(k).startswith("transform_bound") for k in top_sd):
        print("[preflight][warn] 该 config 快照在顶层 shift_deeponet: 块写了 "
              "transform_bound_* —— 训练时不生效 (E0 历史 bug), checkpoint 实为无界 "
              "Shift-DeepONet; 本脚本按快照原样构建, 推理与训练自洽, 但论文表述须与 E0 结论一致。")
    if args.expect_unbounded:
        assert eff_scale == 0.0 and eff_shift == 0.0, \
            f"期望无界, 实际 scale={eff_scale}, shift={eff_shift}"
    elif args.expect_scale is not None or args.expect_shift is not None:
        assert args.expect_scale is not None and abs(eff_scale - args.expect_scale) < 1e-9, \
            f"transform_bound_scale={eff_scale} != 期望 {args.expect_scale}"
        assert args.expect_shift is not None and abs(eff_shift - args.expect_shift) < 1e-9, \
            f"transform_bound_shift={eff_shift} != 期望 {args.expect_shift}"
    print("[preflight] OK")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = ckpt_run / "results" / "checkpoints" / args.ckpt_name
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device).eval()
    print(f"[model] 加载 {ckpt_path} -> {device}")

    scaler = DataScaler.load(ckpt_run / "data" / "processed" / "scaler.npz")

    # ---------------- 物理网格与 branch 键序 ----------------
    phys = cfg["physics"]
    L = float(phys["domain_length"]); T = float(phys["simulation_time"])
    n_z = int(phys["n_spatial_nodes"]); n_t = int(phys["n_time_steps"])
    z = np.linspace(0.0, L, n_z)
    t = np.linspace(0.0, T, n_t)
    dz = float(z[1] - z[0])

    qcfg = cfg["q_top_function"]
    n_steps = int(qcfg["n_steps"]); n_ctrl = int(qcfg["n_control_points"])
    q_min = float(qcfg["min_flux"]); q_max = float(qcfg["max_flux"])
    dt_forcing = T / n_steps

    # branch 键序 = param_keys + [h_init, c_init]
    # (src/data_generation/postprocess.py:226-263 get_param_keys/get_branch_keys)
    q_cols = [f"q_top_{i:02d}" for i in range(n_steps)]
    branch_keys = SOIL_KEYS + q_cols + ["c_top", "h_init", "c_init"]
    assert len(branch_keys) == int(model_cfg["branch_input_dim"]) == scaler.input_min.shape[0], \
        (len(branch_keys), model_cfg["branch_input_dim"], scaler.input_min.shape)

    # ---------------- 场景生成 (同训练逻辑) ----------------
    N = args.n_scenarios
    n_dims = n_ctrl + (len(SOIL_KEYS) if args.param_jitter else 0)
    sampler = qmc.LatinHypercube(d=n_dims, seed=args.seed)
    unit = sampler.random(n=N)

    ctrl_values = q_min + unit[:, :n_ctrl] * (q_max - q_min)
    q_series = _interpolate_q_top_series(ctrl_values, n_steps=n_steps)  # (N, 48)
    q_total = q_series.sum(axis=1) * dt_forcing  # cm (总入渗量)

    soil = np.tile(np.array([LOAM_PARAMS[k] for k in SOIL_KEYS]), (N, 1))
    if args.param_jitter:
        f = args.jitter_frac
        mult = 1.0 - f + 2.0 * f * unit[:, n_ctrl:]
        soil = soil * mult
        for j, k in enumerate(SOIL_KEYS):
            lo, hi = TRAIN_RANGES[k]
            soil[:, j] = np.clip(soil[:, j], lo, hi)

    # branch 原始向量 → scaler min-max 归一化 (同 postprocess.build_dataset:337)
    # h_init/c_init 在训练数据中为 h(:,0).mean()/c(:,0).mean() ≈ 初始条件常数
    branch_raw = np.concatenate(
        [soil, q_series,
         np.full((N, 1), C_TOP),
         np.full((N, 1), float(phys["initial_head"])),
         np.full((N, 1), float(phys["initial_conc"]))],
        axis=1,
    )
    branch_norm = scaler.transform_input(branch_raw).astype(np.float32)

    # trunk: (z,t) 笛卡尔积, 归一化 z/L, t/T (postprocess.build_dataset:327-344)
    zz, tt = np.meshgrid(z, t, indexing="ij")
    trunk = np.column_stack([zz.ravel() / z.max(), tt.ravel() / t.max()]).astype(np.float32)
    trunk_t = torch.from_numpy(trunk).to(device)

    # ---------------- 批量推理 + 决策量 ----------------
    c_th = args.bt_frac * C_TOP
    cum_top_w = cum_forcing_at(q_series, dt_forcing, t)          # (N, n_t) 顶部累计水量
    cum_top_s = cum_top_w * C_TOP                                 # 顶部累计溶质 (c_top 常数)

    tb = np.empty(N); peak_c = np.empty(N)
    mbe_w_final = np.empty(N); mbe_c_final = np.empty(N)
    mbe_w_tmean = np.empty(N); mbe_c_tmean = np.empty(N)
    c_bottom_all = np.empty((N, n_t), dtype=np.float32)

    def predict_batch(sl):
        """镜像 06_evaluate.py predict(): 前向→反归一化→c 截非负→(B,n_z,n_t)。"""
        b = torch.from_numpy(branch_norm[sl]).to(device)
        with torch.no_grad():
            pred = model(b, trunk_t)
        h = scaler.inverse_h(pred[..., 0].cpu().numpy())
        c = np.clip(scaler.inverse_c(pred[..., 1].cpu().numpy()), 0, None)
        B = h.shape[0]
        return h.reshape(B, n_z, n_t), c.reshape(B, n_z, n_t)

    bs = args.batch_size
    for start in range(0, N, bs):
        sl = slice(start, min(start + bs, N))
        h_phys, c_phys = predict_batch(sl)
        B = h_phys.shape[0]
        p = {k: soil[sl, j].reshape(B, 1, 1) for j, k in enumerate(SOIL_KEYS)}

        c_bot = c_phys[:, -1, :]                                  # z=L 底部 (metrics.py:196)
        c_bottom_all[sl] = c_bot
        tb[sl] = breakthrough_times(c_bot, t, c_th)
        peak_c[sl] = c_bot.max(axis=1)

        # --- 自诊断 MBE (B2 口径, 见模块 docstring 第 4 点) ---
        theta = vg_theta(h_phys, p["theta_r"], p["theta_s"], p["alpha"], p["n_vg"])
        storage_w = np.trapz(theta, dx=dz, axis=1)                # (B, n_t)
        delta_sw = storage_w - storage_w[:, :1]
        q_bot = vg_K_np(h_phys[:, -1, :],
                        p["K_s"][:, 0, :], p["alpha"][:, 0, :], p["n_vg"][:, 0, :])
        cum_bot_w = cumtrapz_axis1(q_bot, t)
        mbe_w = mass_balance_error_pct(delta_sw, cum_top_w[sl] - cum_bot_w)

        storage_c = np.trapz(theta * c_phys, dx=dz, axis=1)
        delta_sc = storage_c - storage_c[:, :1]
        cum_bot_s = cumtrapz_axis1(q_bot * c_bot, t)
        mbe_c = mass_balance_error_pct(delta_sc, cum_top_s[sl] - cum_bot_s)

        mbe_w_final[sl] = mbe_w[:, -1]; mbe_c_final[sl] = mbe_c[:, -1]
        mbe_w_tmean[sl] = mbe_w[:, 1:].mean(axis=1); mbe_c_tmean[sl] = mbe_c[:, 1:].mean(axis=1)
        if start % (bs * 10) == 0:
            print(f"  推理 {sl.stop}/{N}")

    mbe_self = np.maximum(mbe_w_final, mbe_c_final)  # 自诊断指标 (README 定义)

    # ---------------- 风险与排序 ----------------
    thr = args.tb_threshold_hours
    risk_prob = float(np.mean(tb < thr))              # P(t_b < 36h), 全 ensemble
    feasible = tb > thr

    order_feas = np.argsort(-q_total)
    ranked = np.concatenate([
        order_feas[feasible[order_feas]],             # 可行: 总入渗量降序
        order_feas[~feasible[order_feas]],            # 不可行垫底 (同样按入渗量降序)
    ])
    top_ids = ranked[: args.n_top]

    # ---------------- 分层抽 150 个验证方案 ----------------
    rng = np.random.default_rng(args.seed + 1)
    selected, strata = [], []

    def add(ids, name, k):
        n_added = 0
        for i in ids:
            if n_added >= k:
                break
            if i not in seen:
                seen.add(int(i)); selected.append(int(i)); strata.append(name)
                n_added += 1

    seen: set[int] = set()
    add(top_ids, "head", args.n_top)                                       # 头部 50
    mid = ranked[N // 3: 2 * N // 3]
    add(mid[np.linspace(0, len(mid) - 1, 60).astype(int)], "mid", 40)      # 中部 40
    tail = ranked[2 * N // 3:]
    add(tail[np.linspace(0, len(tail) - 1, 45).astype(int)], "tail", 30)   # 尾部 30
    add(np.argsort(-mbe_self), "high_mbe", 30)                             # 高 MBE 30
    remaining = np.array([i for i in range(N) if i not in seen])
    if len(selected) < args.n_validate:
        add(rng.permutation(remaining), "fill", args.n_validate - len(selected))
    selected = np.array(selected[: args.n_validate])
    strata = strata[: args.n_validate]

    # ---------------- 输出 ----------------
    np.savez_compressed(
        out_dir / "results" / "mc_scenarios.npz",
        q_series=q_series.astype(np.float32), soil=soil.astype(np.float32),
        soil_keys=np.asarray(SOIL_KEYS, dtype=str),
        q_total=q_total, tb_hours=tb, peak_c_bottom=peak_c,
        c_bottom=c_bottom_all,
        mbe_w_final=mbe_w_final, mbe_c_final=mbe_c_final,
        mbe_w_tmean=mbe_w_tmean, mbe_c_tmean=mbe_c_tmean, mbe_self=mbe_self,
        feasible=feasible, ranked_ids=ranked, top_ids=top_ids,
        selected_ids=selected, selected_strata=np.asarray(strata, dtype=str),
        z=z, t=t,
    )

    # 为验证子集保存完整预测场 (画剖面图用)
    h_sel = np.empty((len(selected), n_z, n_t), dtype=np.float32)
    c_sel = np.empty((len(selected), n_z, n_t), dtype=np.float32)
    for k0 in range(0, len(selected), bs):
        ids = selected[k0: k0 + bs]
        hh, cc = predict_batch(ids)
        h_sel[k0: k0 + len(ids)] = hh; c_sel[k0: k0 + len(ids)] = cc
    np.savez_compressed(out_dir / "results" / "selected_fields.npz",
                        scenario_ids=selected, h_pred=h_sel, c_pred=c_sel, z=z, t=t)

    # HYDRUS 验证参数表 (02_run_hydrus.py:60-67 要求 index 列 sample_id)
    csv_path = out_dir / "results" / "validation_params.csv"
    header = (["sample_id"] + SOIL_KEYS + q_cols
              + ["c_top", "scenario_id", "stratum", "surrogate_tb_hours",
                 "surrogate_peak_c", "surrogate_mbe_pct", "q_top_total"])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row_i, (sc, st) in enumerate(zip(selected, strata)):
            w.writerow(
                [row_i]
                + [f"{v:.8g}" for v in soil[sc]]
                + [f"{v:.8g}" for v in q_series[sc]]
                + [f"{C_TOP:.8g}", int(sc), st,
                   ("inf" if np.isinf(tb[sc]) else f"{tb[sc]:.4g}"),
                   f"{peak_c[sc]:.6g}", f"{mbe_self[sc]:.6g}", f"{q_total[sc]:.6g}"]
            )

    summary = {
        "ckpt_run_dir": str(ckpt_run), "ckpt": str(ckpt_path),
        "config_used": str(cfg_path),
        "effective_transform_bound_scale": eff_scale,
        "effective_transform_bound_shift": eff_shift,
        "n_scenarios": N, "seed": args.seed,
        "param_jitter": bool(args.param_jitter), "jitter_frac": args.jitter_frac,
        "loam_params": LOAM_PARAMS, "c_top": C_TOP,
        "tb_threshold_hours": thr, "bt_frac": args.bt_frac,
        "risk_prob_tb_lt_threshold": risk_prob,
        "n_feasible": int(feasible.sum()),
        "tb_hours": {
            "median_finite": float(np.median(tb[np.isfinite(tb)])) if np.isfinite(tb).any() else None,
            "n_no_breakthrough": int(np.isinf(tb).sum()),
        },
        "q_total_cm": {"min": float(q_total.min()), "max": float(q_total.max()),
                       "top50_min": float(q_total[top_ids].min()) if len(top_ids) else None},
        "mbe_self_pct": {"median": float(np.median(mbe_self)),
                         "p90": float(np.percentile(mbe_self, 90)),
                         "frac_gt_10pct": float(np.mean(mbe_self > 10.0))},
        "top_ids": top_ids.tolist(),
        "n_validate": int(len(selected)),
        "strata_counts": {s: strata.count(s) for s in sorted(set(strata))},
        "outputs": {"scenarios_npz": str(out_dir / "results" / "mc_scenarios.npz"),
                    "validation_csv": str(csv_path),
                    "selected_fields_npz": str(out_dir / "results" / "selected_fields.npz")},
    }
    with open(out_dir / "results" / "mc_screening_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: summary[k] for k in
                      ["risk_prob_tb_lt_threshold", "n_feasible", "mbe_self_pct",
                       "strata_counts"]}, indent=2, ensure_ascii=False))
    print(f"[done] 结果写入 {out_dir / 'results'}")


if __name__ == "__main__":
    main()
