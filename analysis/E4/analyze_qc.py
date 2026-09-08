#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E4: QC 前后分布分析（回应审稿意见 R1-5 / R4-3；纯分析、零训练）。

依赖：Python 3.9+ 与 numpy（标准库 csv/json；无 pandas/scipy）。
matplotlib 为可选：有则额外输出直方图 png，无则仅输出直方图数据 CSV。

事实基础（主仓库代码，见 REPO_FACTS.md §3）：
  - 划分在 step01（仿真前）写入 lhs_params.csv 的 split 列
    （pipeline/01_generate_params.py + src/data_generation/sampling.py::split_dataset）；
  - QC 在 step03 按既定 split 内剔除（pipeline/03_process_data.py::run_quality_control），
    剔除清单写 metadata/qc_summary.json（n_raw/n_passed/n_failed, failed_sample_ids, failed_reasons）。
  - HYDRUS 失败样本不落 raw npz（pre-QC 样本量 = data/raw/sample_XXXX.npz 计数，按文件名统计，
    不读取二进制内容）。

输入（全部经 CLI 传入，路径应已由 run.sh 用 readlink -f 解析为真实路径）：
  --scene NAME PARAMS_CSV QC_JSON RAW_DIR OOD_PARAM   （可重复；QC_JSON/RAW_DIR 可填 NONE 降级）

输出（--out-dir，默认包内 output/）：
  (1) qc_counts.csv / qc_counts.md         每场景×子集 pre/post-QC 样本量与剔除率
      qc_reasons.csv                       QC 剔除原因分类计数
  (2) ks_removed_vs_retained.csv / .md     剔除 vs 保留在 5 个变量上的 KS 统计量 + 置换 p 值
  (3) ood_quantile_distance.csv / .md      post-QC（对照 pre-QC）OOD 测试集对训练集的分位距离
      extreme_removal.csv                  划分变量极端区（>= pre-QC 池 P90）剔除集中度
  (4) hist_<scene>_<var>.csv               直方图数据（可选 figs/hist_<scene>_<var>.png）
  summary.json / SUMMARY.md                自动判据汇总（最终结论仍需人工复核）
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

VARS = ["q_top_peak", "q_top_peak_time", "K_s", "n_vg", "alpha"]
SPLITS = ["train", "val", "test"]

# ---- 自动判据阈值（README 验收标准同步修改） --------------------------------
TH_REMOVAL_MAX = 0.15      # 每子集 QC 剔除率上限
TH_REMOVAL_SPREAD = 0.05   # 同场景子集间剔除率极差上限
TH_ALPHA = 0.05            # KS 置换检验显著性水平
TH_EXTREME_RATIO = 2.0     # 极端区剔除率 / 全体剔除率 的告警阈值


# ============================ 基础工具 ============================

def read_params_csv(path: Path) -> dict[int, dict]:
    """读取 lhs_params.csv -> {sample_id: {列名: 字符串值}}。只保留分析所需列以省内存。"""
    keep = set(VARS) | {"split", "is_ood", "split_mode"}
    rows: dict[int, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"{path} 缺少 sample_id 列（应由 01_generate_params.py 以 index_label 写出）")
        missing = [v for v in VARS if v not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} 缺少分析变量列: {missing}")
        for r in reader:
            sid = int(r["sample_id"])
            rows[sid] = {k: r[k] for k in keep if k in r}
    return rows


def list_raw_sample_ids(raw_dir: Path) -> set[int] | None:
    """仅按文件名统计 raw npz（不读内容，dataless 安全）。"""
    if raw_dir is None or not raw_dir.is_dir():
        return None
    pat = re.compile(r"^sample_(\d+)\.npz$")
    ids = set()
    for p in raw_dir.iterdir():
        m = pat.match(p.name)
        if m:
            ids.add(int(m.group(1)))
    return ids if ids else None


def ks_2samp_stat(x: np.ndarray, y: np.ndarray) -> float:
    """双样本 KS 统计量 D = sup|F_x - F_y|（手写，无 scipy；对并列值/离散变量按 ECDF 定义处理）。"""
    xs, ys = np.sort(x), np.sort(y)
    v = np.concatenate([xs, ys])
    cx = np.searchsorted(xs, v, side="right") / xs.size
    cy = np.searchsorted(ys, v, side="right") / ys.size
    return float(np.max(np.abs(cx - cy)))


def ks_perm_test(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator):
    """置换检验 p 值：p = (1 + #{D_perm >= D_obs}) / (n_perm + 1)。离散变量下仍精确有效。"""
    d_obs = ks_2samp_stat(x, y)
    pooled = np.concatenate([x, y])
    n = x.size
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pooled)
        if ks_2samp_stat(p[:n], p[n:]) >= d_obs - 1e-12:
            hits += 1
    return d_obs, (hits + 1) / (n_perm + 1)


def ecdf_position(ref: np.ndarray, value: float) -> float:
    """value 在 ref 经验分布中的分位位置（0~1）。"""
    return float(np.searchsorted(np.sort(ref), value, side="right")) / ref.size


def classify_reason(reason: str) -> str:
    r = reason.lower()
    if r.startswith("contains nan"):
        return "nan_h_or_c"
    if r.startswith("theta contains nan"):
        return "nan_theta"
    if r.startswith("theta out of range"):
        return "theta_out_of_range"
    if r.startswith("negative concentration"):
        return "negative_concentration"
    if r.startswith("water balance"):
        return "water_balance"
    if r.startswith("solute balance"):
        return "solute_balance"
    return "other"


def write_csv(path: Path, header: list[str], rows: list[list]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# ============================ 单场景分析 ============================

def analyze_scene(scene: dict, n_perm: int, n_bins: int, rng: np.random.Generator):
    """返回该场景的全部分析结果字典。scene 键：name/params_csv/qc_json/raw_dir/ood_param。"""
    name = scene["name"]
    rows = read_params_csv(scene["params_csv"])
    all_ids = np.array(sorted(rows.keys()))

    # --- QC 剔除清单（缺失则降级：只做 CSV 可得信息） ---
    qc = None
    if scene["qc_json"] is not None and scene["qc_json"].is_file():
        with open(scene["qc_json"]) as f:
            qc = json.load(f)
    failed_ids = set(int(i) for i in qc["failed_sample_ids"]) if qc else set()
    degraded = qc is None

    # --- pre-QC = 仿真成功（raw npz 存在）；raw 目录不可用则退化为 CSV 全体 ---
    raw_ids = list_raw_sample_ids(scene["raw_dir"])
    raw_available = raw_ids is not None
    sim_ok_ids = set(int(i) for i in all_ids) & raw_ids if raw_available else set(int(i) for i in all_ids)
    # qc_summary 的 failed_ids 定义在仿真成功样本上；防御性求交
    failed_ids &= sim_ok_ids
    post_ids = sim_ok_ids - failed_ids

    if qc is not None and raw_available and len(sim_ok_ids) != int(qc.get("n_raw", -1)):
        print(f"[warn] {name}: raw npz 计数 {len(sim_ok_ids)} != qc_summary.n_raw {qc.get('n_raw')}"
              f"（raw 目录可能被清理或补跑过，以 qc_summary 口径为准解读）", file=sys.stderr)

    def ids_of_split(s):
        return [i for i in all_ids if rows[i]["split"] == s]

    # (1) 计数表
    counts = []
    for s in SPLITS:
        ids_s = ids_of_split(s)
        n_csv = len(ids_s)
        n_sim = sum(1 for i in ids_s if i in sim_ok_ids)
        n_qcf = sum(1 for i in ids_s if i in failed_ids)
        n_post = sum(1 for i in ids_s if i in post_ids)
        counts.append({
            "scene": name, "subset": s,
            "n_csv": n_csv,
            "n_sim_ok(pre_qc)": n_sim if raw_available else f"{n_sim}(=CSV,raw不可用)",
            "n_sim_failed": (n_csv - n_sim) if raw_available else "NA",
            "n_qc_failed": n_qcf if not degraded else "NA",
            "n_post_qc": n_post if not degraded else "NA",
            "qc_removal_rate": (n_qcf / n_sim) if (not degraded and n_sim > 0) else None,
            "total_attrition_rate": (1.0 - n_post / n_csv) if (not degraded and n_csv > 0) else None,
        })

    # 剔除原因分类
    reasons = {}
    if qc is not None:
        for sid, reason in qc.get("failed_reasons", {}).items():
            if int(sid) in sim_ok_ids or not raw_available:
                cat = classify_reason(str(reason))
                reasons[cat] = reasons.get(cat, 0) + 1

    def values(ids, var):
        return np.array([float(rows[i][var]) for i in ids], dtype=np.float64)

    # (2) 剔除 vs 保留 KS（场景内合并三个子集；KS 对单调变换不变，K_s/alpha 无需取 log）
    ks_rows = []
    if not degraded and len(failed_ids) >= 5:
        rem = sorted(failed_ids)
        ret = sorted(post_ids)
        for var in VARS:
            x, y = values(rem, var), values(ret, var)
            d, p = ks_perm_test(x, y, n_perm, rng)
            ks_rows.append({
                "scene": name, "variable": var,
                "n_removed": x.size, "n_retained": y.size,
                "ks_D": d, "p_perm": p,
                "median_removed": float(np.median(x)),
                "median_retained": float(np.median(y)),
            })

    # (3) OOD 测试集 vs 训练集分位距离（pre/post 对照）
    var = scene["ood_param"]
    dist_rows = []
    for stage, pool in [("pre_qc", sim_ok_ids), ("post_qc", post_ids)]:
        if degraded and stage == "post_qc":
            continue
        tr = values([i for i in ids_of_split("train") if i in pool], var)
        te = values([i for i in ids_of_split("test") if i in pool], var)
        if tr.size == 0 or te.size == 0:
            continue
        tr_p90 = float(np.quantile(tr, 0.9))
        te_p50 = float(np.median(te))
        dist_rows.append({
            "scene": name, "variable": var, "stage": stage,
            "train_p50": float(np.median(tr)),
            "train_p90": tr_p90,
            "train_max": float(np.max(tr)),
            "test_p10": float(np.quantile(te, 0.1)),
            "test_p50": te_p50,
            "test_median_quantile_in_train": ecdf_position(tr, te_p50),
            "frac_test_ge_train_p90": float(np.mean(te >= tr_p90)),
            "frac_test_gt_train_max": float(np.mean(te > float(np.max(tr)))),
            "gap_testP50_minus_trainP90": te_p50 - tr_p90,
        })

    # 极端区剔除集中度：pre-QC 池上 var >= P90 的样本，其剔除率 / 全体剔除率
    extreme = None
    if not degraded and len(sim_ok_ids) > 0:
        pool = sorted(sim_ok_ids)
        v = values(pool, var)
        thr = float(np.quantile(v, 0.9))
        in_ext = v >= thr
        failed_mask = np.array([i in failed_ids for i in pool])
        overall = float(np.mean(failed_mask))
        ext_rate = float(np.mean(failed_mask[in_ext])) if in_ext.sum() > 0 else float("nan")
        extreme = {
            "scene": name, "variable": var, "pre_qc_p90_threshold": thr,
            "n_extreme": int(in_ext.sum()),
            "removal_rate_overall": overall,
            "removal_rate_extreme": ext_rate,
            "concentration_ratio": (ext_rate / overall) if overall > 0 else float("nan"),
        }

    # (4) 直方图数据（公共分箱 = pre-QC 池范围）
    hists = {}
    for hvar in VARS:
        pool = sorted(sim_ok_ids)
        v_all = values(pool, hvar)
        if v_all.size == 0:
            continue
        lo, hi = float(v_all.min()), float(v_all.max())
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, n_bins + 1)
        groups = {
            "retained": values(sorted(post_ids), hvar) if not degraded else None,
            "removed": values(sorted(failed_ids), hvar) if not degraded else None,
            "train_post": values([i for i in ids_of_split("train") if i in post_ids], hvar) if not degraded else None,
            "test_post": values([i for i in ids_of_split("test") if i in post_ids], hvar) if not degraded else None,
            "pre_qc_all": v_all,
        }
        hcounts = {}
        for g, arr in groups.items():
            if arr is None or arr.size == 0:
                hcounts[g] = np.zeros(n_bins, dtype=int)
            else:
                hcounts[g], _ = np.histogram(arr, bins=bins)
        hists[hvar] = {"bins": bins, "counts": hcounts}

    return {
        "name": name, "degraded": degraded, "raw_available": raw_available,
        "qc_meta": {k: qc[k] for k in ("n_raw", "n_passed", "n_failed", "threshold_fraction") if qc and k in qc} if qc else None,
        "counts": counts, "reasons": reasons, "ks": ks_rows,
        "distance": dist_rows, "extreme": extreme, "hists": hists,
        "ood": scene["ood"], "ood_param": var,
    }


# ============================ 汇总与判据 ============================

def build_verdict(results: list[dict]) -> dict:
    checks = {}
    for r in results:
        name = r["name"]
        c = {}
        if r["degraded"]:
            c["status"] = "DEGRADED(qc_summary.json 缺失，仅 CSV 信息)"
            checks[name] = c
            continue
        rates = [row["qc_removal_rate"] for row in r["counts"] if row["qc_removal_rate"] is not None]
        c["max_subset_removal_rate"] = max(rates) if rates else None
        c["removal_rate_spread"] = (max(rates) - min(rates)) if rates else None
        c["pass_removal_low"] = bool(rates and max(rates) <= TH_REMOVAL_MAX)
        c["pass_removal_uniform"] = bool(rates and (max(rates) - min(rates)) <= TH_REMOVAL_SPREAD)
        pvals = {row["variable"]: row["p_perm"] for row in r["ks"]}
        c["ks_p_values"] = pvals
        c["pass_ks_all"] = bool(pvals) and all(p > TH_ALPHA for p in pvals.values())
        c["ks_significant_vars"] = [v for v, p in pvals.items() if p <= TH_ALPHA]
        if r["ood"]:
            post = [d for d in r["distance"] if d["stage"] == "post_qc"]
            if post:
                d = post[0]
                c["pass_ood_gap_preserved"] = bool(d["test_p50"] >= d["train_p90"])
                c["post_qc_test_median_quantile_in_train"] = d["test_median_quantile_in_train"]
        if r["extreme"]:
            c["extreme_concentration_ratio"] = r["extreme"]["concentration_ratio"]
            c["pass_no_extreme_concentration"] = bool(
                not np.isfinite(r["extreme"]["concentration_ratio"])
                or r["extreme"]["concentration_ratio"] < TH_EXTREME_RATIO
            )
        keys = [k for k in c if k.startswith("pass_")]
        c["scene_pass"] = all(c[k] for k in keys) if keys else False
        checks[name] = c
    all_pass = all(v.get("scene_pass", False) for v in checks.values())
    return {
        "thresholds": {
            "removal_rate_max": TH_REMOVAL_MAX,
            "removal_rate_spread": TH_REMOVAL_SPREAD,
            "ks_alpha": TH_ALPHA,
            "extreme_concentration_ratio": TH_EXTREME_RATIO,
        },
        "per_scene": checks,
        "overall": "DIRECT_REPLY" if all_pass else "CHECK_CONTINGENCY",
        "note": "DIRECT_REPLY=可直接引用本分析回复审稿人；CHECK_CONTINGENCY=存在剔除率偏高/分布位移显著/极端区集中，"
                "按 README 预案考虑更严数值设置补跑（02_run_hydrus --start/--end）。最终结论需人工复核。",
    }


# ============================ 主流程 ============================

def main():
    ap = argparse.ArgumentParser(description="E4 QC 前后分布分析（numpy-only，本地可跑）")
    ap.add_argument("--scene", nargs=5, action="append", required=True,
                    metavar=("NAME", "PARAMS_CSV", "QC_JSON", "RAW_DIR", "OOD_PARAM"),
                    help="可重复。QC_JSON/RAW_DIR 可填 NONE；OOD_PARAM 形如 q_top_peak / q_top_peak_time，"
                         "iid 场景加后缀 ':iid' 表示无 OOD 划分（如 q_top_peak:iid）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=2000, help="KS 置换次数（默认 2000）")
    ap.add_argument("--n-bins", type=int, default=24, help="直方图分箱数（默认 24）")
    ap.add_argument("--seed", type=int, default=42, help="置换检验随机种子（默认 42）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scenes = []
    for name, pcsv, qjson, rawd, oparam in args.scene:
        ood = not oparam.endswith(":iid")
        scenes.append({
            "name": name,
            "params_csv": Path(pcsv),
            "qc_json": None if qjson.upper() == "NONE" else Path(qjson),
            "raw_dir": None if rawd.upper() == "NONE" else Path(rawd),
            "ood_param": oparam.split(":")[0],
            "ood": ood,
        })
        if not scenes[-1]["params_csv"].is_file():
            sys.exit(f"[error] 找不到参数表: {pcsv}")

    results = [analyze_scene(s, args.n_perm, args.n_bins, rng) for s in scenes]

    # ---- (1) 计数表 ----
    cnt_hdr = ["scene", "subset", "n_csv", "n_sim_ok(pre_qc)", "n_sim_failed",
               "n_qc_failed", "n_post_qc", "qc_removal_rate", "total_attrition_rate"]
    cnt_rows = [[fmt(row[h]) for h in cnt_hdr] for r in results for row in r["counts"]]
    write_csv(out_dir / "qc_counts.csv", cnt_hdr, cnt_rows)
    (out_dir / "qc_counts.md").write_text(
        "# E4 (1) 每场景×子集 pre/post-QC 样本量与剔除率\n\n"
        "口径：n_csv=step01 写入 CSV 的划分数；n_sim_ok=HYDRUS 成功（raw npz 存在，按文件名计数）=pre-QC；\n"
        "n_qc_failed=step03 QC 剔除；n_post_qc=进入 processed npz 的样本；qc_removal_rate=n_qc_failed/n_sim_ok。\n\n"
        + md_table(cnt_hdr, cnt_rows), encoding="utf-8")

    reason_rows = [[r["name"], cat, n] for r in results for cat, n in sorted(r["reasons"].items())]
    write_csv(out_dir / "qc_reasons.csv", ["scene", "reason_category", "count"], reason_rows)

    # ---- (2) KS 表 ----
    ks_hdr = ["scene", "variable", "n_removed", "n_retained", "ks_D", "p_perm",
              "median_removed", "median_retained"]
    ks_rows = [[fmt(row[h]) for h in ks_hdr] for r in results for row in r["ks"]]
    write_csv(out_dir / "ks_removed_vs_retained.csv", ks_hdr, ks_rows)
    (out_dir / "ks_removed_vs_retained.md").write_text(
        "# E4 (2) QC 剔除 vs 保留样本分布对比（双样本 KS + 置换 p 值）\n\n"
        f"手写 KS（无 scipy），p 值为 {args.n_perm} 次置换的 (1+hits)/(n+1)；seed={args.seed}。\n"
        "KS 对单调变换不变，log 分布参数 (K_s, alpha) 无需取对数；q_top_peak_time 为离散变量（1 h 步长），\n"
        "置换 p 值对并列值仍精确有效。p<=0.05 表示剔除样本在该变量上分布与保留样本显著不同。\n\n"
        + md_table(ks_hdr, ks_rows), encoding="utf-8")

    # ---- (3) 分位距离 + 极端区集中度 ----
    d_hdr = ["scene", "variable", "stage", "train_p50", "train_p90", "train_max",
             "test_p10", "test_p50", "test_median_quantile_in_train",
             "frac_test_ge_train_p90", "frac_test_gt_train_max", "gap_testP50_minus_trainP90"]
    d_rows = [[fmt(row[h]) for h in d_hdr] for r in results for row in r["distance"]]
    write_csv(out_dir / "ood_quantile_distance.csv", d_hdr, d_rows)
    (out_dir / "ood_quantile_distance.md").write_text(
        "# E4 (3) 测试集对训练集在划分变量上的分位距离（pre-QC vs post-QC）\n\n"
        "关键判据：OOD 场景 post_qc 行的 test_p50 应仍 >= train_p90（OOD 间隔未被 QC 侵蚀）。\n"
        "test_median_quantile_in_train = 测试集中位数落在训练集 ECDF 的分位位置（OOD 场景应接近 1.0；\n"
        "iid 场景为参考行，应接近 0.5）。\n\n"
        + md_table(d_hdr, d_rows), encoding="utf-8")

    e_hdr = ["scene", "variable", "pre_qc_p90_threshold", "n_extreme",
             "removal_rate_overall", "removal_rate_extreme", "concentration_ratio"]
    e_rows = [[fmt(r["extreme"][h]) for h in e_hdr] for r in results if r["extreme"]]
    write_csv(out_dir / "extreme_removal.csv", e_hdr, e_rows)

    # ---- (4) 直方图 CSV（+ 可选 png） ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        has_mpl = True
        (out_dir / "figs").mkdir(exist_ok=True)
    except Exception:
        has_mpl = False
        print("[info] matplotlib 不可用，降级为仅输出直方图 CSV", file=sys.stderr)

    for r in results:
        for var, h in r["hists"].items():
            bins, hc = h["bins"], h["counts"]
            hdr = ["bin_left", "bin_right"] + list(hc.keys())
            rws = [[fmt(float(bins[i]), 6), fmt(float(bins[i + 1]), 6)] + [int(hc[g][i]) for g in hc]
                   for i in range(len(bins) - 1)]
            write_csv(out_dir / f"hist_{r['name']}_{var}.csv", hdr, rws)
            if has_mpl:
                fig, ax = plt.subplots(figsize=(6, 4))
                mid = (bins[:-1] + bins[1:]) / 2
                n_plotted = 0
                for g, style in [("pre_qc_all", "-"), ("retained", "-"), ("removed", "--"),
                                 ("train_post", ":"), ("test_post", "-.")]:
                    if g in hc and hc[g].sum() > 0:
                        ax.step(mid, hc[g], style, where="mid", label=g,
                                alpha=0.45 if g == "pre_qc_all" else 1.0)
                        n_plotted += 1
                ax.set_xlabel(var); ax.set_ylabel("count")
                ax.set_title(f"{r['name']}: {var} (pre-QC common bins)")
                if n_plotted:
                    ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(out_dir / "figs" / f"hist_{r['name']}_{var}.png", dpi=150)
                plt.close(fig)

    # ---- 汇总 ----
    verdict = build_verdict(results)
    summary = {
        "n_perm": args.n_perm, "seed": args.seed,
        "scenes": [{k: r[k] for k in ("name", "degraded", "raw_available", "qc_meta",
                                      "counts", "reasons", "ks", "distance", "extreme",
                                      "ood", "ood_param")} for r in results],
        "verdict": verdict,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    lines = ["# E4 SUMMARY（自动判据，需人工复核）\n",
             f"- 总体判定：**{verdict['overall']}**",
             f"- 阈值：剔除率<={TH_REMOVAL_MAX}，子集间极差<={TH_REMOVAL_SPREAD}，KS alpha={TH_ALPHA}，"
             f"极端区集中比<{TH_EXTREME_RATIO}\n"]
    for name, c in verdict["per_scene"].items():
        lines.append(f"## {name}")
        for k, v in c.items():
            lines.append(f"- {k}: {fmt(v) if isinstance(v, float) else v}")
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] 输出目录: {out_dir}")
    print(f"[done] 总体判定: {verdict['overall']}（详见 SUMMARY.md / summary.json）")


if __name__ == "__main__":
    main()
