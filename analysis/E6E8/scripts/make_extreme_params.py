#!/usr/bin/env python3
"""E6-B/E8: 生成三组新外推参数表（本地可跑, numpy-only, 零仿真）。

输出 <包目录>/outputs/params/<set_key>/lhs_params.csv，随后由 run_B_cluster.sh
复制进 $REPO/experiments/qtop_func/runs_revision/<set_key>/data/parameters/，
供 pipeline/02_run_hydrus.py 直接消费（build_sample_params 会把
q_top_00..q_top_47 读成 q_top_series；hydrus_runner 只校验长度=48，无幅值上界）。

三组集合:
  (1) E6_extreme_peak_v1   96 样本: 越界峰值 max_flux -> 5.5 / 6.5 / 8.0 cm/h 各 32
      （训练生成上界 5.0）。q_top 构造: 6 个整数时刻节点的分段线性序列（与训练
      数据同族——sampling.py 的 6 控制点线性插值），其中 1 个内部节点取目标峰值，
      其余节点 U(0.1, 5.0)，保证 max(series) 恰等于目标值。
  (2) E6_extreme_peaktime_v1  32 样本: 峰值时刻压在最后 4h（44/45/46/47h 各 8），
      峰值幅值 U(4.0, 5.0) 保持训练幅值范围内，隔离峰时轴。
  (3) E8_joint_shift_v1  32 样本: 高峰值(top-10% 风格, U(4.5,5.0))
      x K_s ∈ {0.15, 40.0} cm/h 各 16（训练范围 0.25-29.7, log_uniform 采样，
      两值均越界）。
土壤其余参数一律在训练范围内做分层 LHS（脚本内简易实现，log 参数在 log 空间分层）。
派生列与主仓库 sampling.generate_lhs_samples 相同:
  q_top_mean / q_top_peak / q_top_std / q_top_total / q_top_peak_time (dt=1h)。

元数据列约定: ood_quantile 留空（NA）——本三组均为**构造性越界/极端集**（目标值
由脚本直接指定，不经分位数切分产生），分位阈值概念不适用；sim 配置里
data.ood_quantile: 0.9 仅是沿用场景基版的记录字段（01_generate_params 不会被
调用），两者不构成矛盾。该列无下游消费者，仅作档案记录。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

N_STEPS = 48
DT_HOURS = 1.0  # simulation_time 48 h / n_steps 48
GEN_MIN_FLUX, GEN_MAX_FLUX = 0.1, 5.0  # 训练期 q_top_function.min_flux/max_flux

# 训练参数范围（与 configs/qtop_func/qtop_func_*_medium.yaml parameters 段一致）
SOIL_BOUNDS = {
    "theta_r": (0.034, 0.098, "uniform"),
    "theta_s": (0.36, 0.51, "uniform"),
    "alpha":   (0.005, 0.145, "log_uniform"),
    "n_vg":    (1.09, 2.68, "uniform"),
    "K_s":     (0.25, 29.7, "log_uniform"),
    "D_L":     (1.0, 20.0, "uniform"),
}
C_TOP_BOUNDS = (0.1, 1.0)
KS_TRAIN_RANGE = (0.25, 29.7)


def lhs_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    """一维分层 LHS: n 个等宽层各取一点后打乱。"""
    return rng.permutation((np.arange(n) + rng.random(n)) / n)


def sample_soil(n: int, rng: np.random.Generator, skip: tuple = ()) -> dict:
    out = {}
    for name, (lo, hi, dist) in SOIL_BOUNDS.items():
        if name in skip:
            continue
        u = lhs_unit(n, rng)
        if dist == "log_uniform":
            out[name] = 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
        else:
            out[name] = lo + u * (hi - lo)
    return out


def series_from_knots(pos: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """整数时刻节点的分段线性插值 -> 48 步序列（节点值在网格上精确命中）。"""
    order = np.argsort(pos)
    return np.interp(np.arange(N_STEPS, dtype=float), pos[order].astype(float),
                     vals[order])


def pick_interior(rng, k, lo, hi, exclude=()):
    """在 [lo, hi] 内取 k 个互异整数（避开 exclude）。"""
    pool = [i for i in range(lo, hi + 1) if i not in exclude]
    return np.array(sorted(rng.choice(pool, size=k, replace=False)))


def build_peak_series(rng, peak_value, peak_lo=6, peak_hi=42, others_hi=None):
    """6 节点序列: 端点 0/47 + 3 随机内部节点 + 1 峰值节点(=peak_value)。"""
    if others_hi is None:
        others_hi = GEN_MAX_FLUX
    t_pk = int(rng.integers(peak_lo, peak_hi + 1))
    others_pos = pick_interior(rng, 3, 2, 45, exclude={t_pk})
    pos = np.concatenate([[0], others_pos, [t_pk], [47]])
    vals = np.concatenate([
        rng.uniform(GEN_MIN_FLUX, others_hi, size=4),  # 0 端点 + 3 内部
        [peak_value],
        rng.uniform(GEN_MIN_FLUX, others_hi, size=1),  # 47 端点
    ])
    # 保证峰值节点严格最大
    vals = np.where(np.arange(6) == 4, vals, np.minimum(vals, peak_value * 0.95))
    s = series_from_knots(pos, vals)
    assert abs(s.max() - peak_value) < 1e-9, "max(series) 必须恰为目标峰值"
    return s


def build_peaktime_series(rng, t_p):
    """峰值时刻 = t_p ∈ {44..47}, 峰值幅值 U(4.0,5.0)（训练幅值范围内）。"""
    peak = rng.uniform(4.0, 5.0)
    low_hi = 0.5 * peak
    if t_p == 47:
        others_pos = pick_interior(rng, 4, 2, 43)
        pos = np.concatenate([[0], others_pos, [47]])
        vals = np.concatenate([rng.uniform(GEN_MIN_FLUX, low_hi, size=5), [peak]])
    else:
        others_pos = pick_interior(rng, 3, 2, t_p - 3)
        pos = np.concatenate([[0], others_pos, [t_p], [47]])
        vals = np.concatenate([
            rng.uniform(GEN_MIN_FLUX, low_hi, size=4),
            [peak],
            rng.uniform(GEN_MIN_FLUX, low_hi, size=1),
        ])
    s = series_from_knots(pos, vals)
    assert int(s.argmax()) == t_p, f"argmax(series) 必须为 {t_p}"
    assert s.argmax() * DT_HOURS >= 44.0
    return s


def derived(q: np.ndarray) -> dict:
    return {
        "q_top_mean": float(q.mean()),
        "q_top_peak": float(q.max()),
        "q_top_std": float(q.std()),
        "q_top_total": float(q.sum() * DT_HOURS),
        "q_top_peak_time": float(q.argmax() * DT_HOURS),
    }


def make_set_extreme_peak(rng):
    targets = [5.5, 6.5, 8.0]
    n_per = 32
    n = n_per * len(targets)
    soil = sample_soil(n, rng)
    c_top = C_TOP_BOUNDS[0] + lhs_unit(n, rng) * (C_TOP_BOUNDS[1] - C_TOP_BOUNDS[0])
    rows = []
    i = 0
    for tgt in targets:
        for _ in range(n_per):
            q = build_peak_series(rng, tgt)
            rows.append(dict(
                soil={k: soil[k][i] for k in soil}, c_top=c_top[i], q=q,
                severity_group=f"peak_{tgt}", target_value=tgt,
                ood_parameter="q_top_peak"))
            i += 1
    return rows


def make_set_extreme_peaktime(rng):
    n_per = 8
    rows = []
    i = 0
    n = n_per * 4
    soil = sample_soil(n, rng)
    c_top = C_TOP_BOUNDS[0] + lhs_unit(n, rng) * (C_TOP_BOUNDS[1] - C_TOP_BOUNDS[0])
    for t_p in (44, 45, 46, 47):
        for _ in range(n_per):
            q = build_peaktime_series(rng, t_p)
            rows.append(dict(
                soil={k: soil[k][i] for k in soil}, c_top=c_top[i], q=q,
                severity_group=f"peaktime_{t_p}h", target_value=float(t_p),
                ood_parameter="q_top_peak_time"))
            i += 1
    return rows


def make_set_joint(rng):
    n_per = 16
    ks_values = [0.15, 40.0]  # 训练范围 0.25-29.7 cm/h, 两值均越界
    n = n_per * len(ks_values)
    soil = sample_soil(n, rng, skip=("K_s",))
    c_top = C_TOP_BOUNDS[0] + lhs_unit(n, rng) * (C_TOP_BOUNDS[1] - C_TOP_BOUNDS[0])
    rows = []
    i = 0
    for ks in ks_values:
        for _ in range(n_per):
            peak = rng.uniform(4.5, 5.0)  # top-10% 风格高峰值, 仍在生成上界内
            q = build_peak_series(rng, peak, others_hi=0.8 * peak)
            s = {k: soil[k][i] for k in soil}
            s["K_s"] = ks
            tag = "joint_Ks0.15" if ks < 1 else "joint_Ks40"
            rows.append(dict(
                soil=s, c_top=c_top[i], q=q,
                severity_group=tag, target_value=ks,
                ood_parameter="K_s_x_q_top_peak"))
            i += 1
    return rows


SOIL_COLS = ["theta_r", "theta_s", "alpha", "n_vg", "K_s", "D_L"]
Q_COLS = [f"q_top_{i:02d}" for i in range(N_STEPS)]
DERIVED_COLS = ["q_top_mean", "q_top_peak", "q_top_std", "q_top_total", "q_top_peak_time"]
META_COLS = ["split", "split_mode", "is_ood", "ood_parameter", "ood_quantile",
             "ood_tail", "severity_group", "target_value"]


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id"] + SOIL_COLS + Q_COLS + ["c_top"] + DERIVED_COLS + META_COLS
    fmt = lambda x: format(float(x), ".12g")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sid, r in enumerate(rows):
            d = derived(r["q"])
            line = [sid]
            line += [fmt(r["soil"][k]) for k in SOIL_COLS]
            line += [fmt(v) for v in r["q"]]
            line += [fmt(r["c_top"])]
            line += [fmt(d[k]) for k in DERIVED_COLS]
            # ood_quantile 留空（NA）: 构造性越界集不由分位数切分产生（见模块 docstring）
            line += ["test", "ood", "True", r["ood_parameter"], "", "high",
                     r["severity_group"], fmt(r["target_value"])]
            w.writerow(line)


def main():
    pkg = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=str(pkg / "outputs" / "params"))
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sets = {
        "E6_extreme_peak_v1": make_set_extreme_peak(rng),
        "E6_extreme_peaktime_v1": make_set_extreme_peaktime(rng),
        "E8_joint_shift_v1": make_set_joint(rng),
    }

    summary = {"seed": args.seed, "ks_train_range": list(KS_TRAIN_RANGE),
               "gen_flux_range": [GEN_MIN_FLUX, GEN_MAX_FLUX], "sets": {}}
    out_root = Path(args.out_root)
    for key, rows in sets.items():
        out_csv = out_root / key / "lhs_params.csv"
        write_csv(rows, out_csv)
        groups = {}
        for r in rows:
            g = r["severity_group"]
            groups.setdefault(g, {"n": 0, "peak_min": np.inf, "peak_max": -np.inf,
                                  "peak_time_min": np.inf, "peak_time_max": -np.inf})
            d = derived(r["q"])
            groups[g]["n"] += 1
            groups[g]["peak_min"] = min(groups[g]["peak_min"], d["q_top_peak"])
            groups[g]["peak_max"] = max(groups[g]["peak_max"], d["q_top_peak"])
            groups[g]["peak_time_min"] = min(groups[g]["peak_time_min"], d["q_top_peak_time"])
            groups[g]["peak_time_max"] = max(groups[g]["peak_time_max"], d["q_top_peak_time"])
        summary["sets"][key] = {"n_samples": len(rows), "csv": str(out_csv),
                                "groups": groups}
        print(f"[OK] {key}: {len(rows)} 样本 -> {out_csv}")
        for g, s in groups.items():
            print(f"     {g}: n={s['n']}, peak=[{s['peak_min']:.3f},{s['peak_max']:.3f}] cm/h, "
                  f"peak_time=[{s['peak_time_min']:.0f},{s['peak_time_max']:.0f}] h")

    with open(out_root / "sets_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"输出: {out_root / 'sets_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
