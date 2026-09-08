#!/usr/bin/env python3
"""E5 步骤 1 (本地可跑, 仅需 numpy + 标准库): 从真实小时降雨记录切事件并组装外部测试参数表。

输入
----
用户自行下载的真实小时降雨记录, 两种格式:
  --format csv       通用 CSV: 一列时间戳 + 一列小时降水量 (mm), 列名用
                     --time-col / --precip-col 指定 (默认 time / precip_mm)。
  --format isd-lite  NOAA ISD-Lite 原始文件 (空白分隔, 第 1-4 列为
                     年/月/日/时, 第 11 列为 1 小时降水, 单位 0.1 mm,
                     -9999 = 缺测, -1 = 痕量降水按 0 处理)。

处理
----
1. 扫描全部 48 h 滑动窗口, 剔除含缺测的窗口, 保留总量 >= --min-total-mm 的候选;
2. 贪心去重 (与已选事件重叠 > --max-overlap-hours 的窗口丢弃), 按总量降序;
3. 事件形态分类: front(峰值在前 16 h) / back(峰值在后 16 h) / mid,
   multi-peak = 存在 >=2 个间隔 >= 6 h 且高度 >= 0.3*峰值的局部峰;
4. 选出 --n-events 场事件, 尽量覆盖 front/back/multi-peak 各 >= 2 场;
5. 雨强线性重标定到训练入渗包络 [0.1, 5.0] cm/h (见 README "定位辩护"):
      q(t) = 0.1 + (5.0 - 0.1) * r(t) / r_ref
   r_ref 默认取所选事件池的最大小时雨强 (--scale-mode pool, 保留事件间相对
   强弱; --scale-mode event 则每场事件峰值都拉到 5.0);
   干燥小时 (r=0) 落在包络下限 0.1 cm/h —— 训练池 min_flux 亦为 0.1,
   即训练分布本身不含零通量, 该差异在 stats_compare 中如实报告;
6. 每场事件 × 3 组代表性土壤参数 (Carsel & Parrish, 1988, WRR 24(5):755-769,
   均在训练采样范围内; K_s 由 cm/d 换算为 cm/h):
      sandy_loam: theta_r=0.065, theta_s=0.41, alpha=0.075, n=1.89, K_s=4.42
      loam      : theta_r=0.078, theta_s=0.43, alpha=0.036, n=1.56, K_s=1.04
      clay_loam : theta_r=0.095, theta_s=0.41, alpha=0.019, n=1.31, K_s=0.26
   D_L 统一取 10.0 cm (训练范围 [1,20] 中值), c_top 统一 0.5 (范围 [0.1,1] 中值);
7. 组装主仓库 lhs_params.csv 风格的参数表:
   q_top_00..q_top_47 + 派生列 q_top_mean/peak/std/total/peak_time
   (定义与 src/data_generation/sampling.py:235-239 一致, forcing_dt = 48h/48 = 1h),
   split 全部 = "test", split_mode = "external_real_rain", is_ood = True。

输出 (写入 --out-dir)
----
  lhs_params.csv       外部测试集参数表 (n_events*3 行, sample_id 从 0 起)
  events_raw.csv       每场事件重标定前的原始雨强 (mm/h), 供 stats_compare 与论文附图
  events_meta.json     事件元数据 (起始时刻/类别/原始峰值/缩放系数/数据来源)

用法示例
----
  python3 build_events.py --input 599758-2024.isd-lite --format isd-lite \
      --out-dir <数据run>/data/parameters --station "NOAA-ISD 599758" --n-events 9
  python3 build_events.py --input hourly_rain.csv --format csv \
      --time-col datetime --precip-col precip_mm --out-dir ... --station "CMA 58362"
  python3 build_events.py --demo --out-dir /tmp/e5_demo   # 仅管线自检, 禁止用于论文!
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

N_STEPS = 48                 # 与 q_top_function.n_steps 一致
FORCING_DT = 48.0 / N_STEPS  # = 1.0 h, simulation_time / n_steps
Q_MIN, Q_MAX = 0.1, 5.0      # 训练入渗包络 (q_top_function.min_flux / max_flux)

# Carsel & Parrish (1988) 类别均值, K_s 换算 cm/d -> cm/h, 全部位于训练采样范围:
#   theta_r [0.034,0.098], theta_s [0.36,0.51], alpha [0.005,0.145],
#   n_vg [1.09,2.68], K_s [0.25,29.7] cm/h, D_L [1,20] cm, c_top [0.1,1.0]
SOILS = {
    "sandy_loam": dict(theta_r=0.065, theta_s=0.41, alpha=0.075, n_vg=1.89, K_s=4.42),
    "loam":       dict(theta_r=0.078, theta_s=0.43, alpha=0.036, n_vg=1.56, K_s=1.04),
    "clay_loam":  dict(theta_r=0.095, theta_s=0.41, alpha=0.019, n_vg=1.31, K_s=0.26),
}
D_L = 10.0
C_TOP = 0.5


def read_generic_csv(path: Path, time_col: str, precip_col: str):
    times, precip = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or precip_col not in reader.fieldnames:
            sys.exit(f"[FAIL] CSV 缺少列 '{precip_col}' (现有列: {reader.fieldnames})")
        for row in reader:
            times.append(row.get(time_col, ""))
            val = (row[precip_col] or "").strip()
            try:
                v = float(val)
            except ValueError:
                v = np.nan
            precip.append(v)
    return times, np.asarray(precip, dtype=float)


def read_isd_lite(path: Path):
    """ISD-Lite: 空白分隔, 字段 = 年 月 日 时 气温 露点 海平面气压 风向 风速 云量 1h降水 6h降水。
    1h 降水单位 0.1 mm; -9999 缺测; -1 痕量 -> 0。"""
    times, precip = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 11:
                continue
            y, m, d, h = parts[0], parts[1], parts[2], parts[3]
            raw = float(parts[10])
            if raw <= -9998:
                v = np.nan
            elif raw < 0:  # trace
                v = 0.0
            else:
                v = raw / 10.0  # 0.1mm -> mm
            times.append(f"{y}-{int(m):02d}-{int(d):02d}T{int(h):02d}:00")
            precip.append(v)
    return times, np.asarray(precip, dtype=float)


def make_demo_series(seed: int = 7):
    """--demo: 合成占位序列, 仅用于管线连通性自检, 严禁作为论文结果!"""
    rng = np.random.default_rng(seed)
    n = 24 * 365
    r = np.zeros(n)
    t0 = 0
    while t0 < n - 60:
        t0 += int(rng.exponential(120)) + 24
        dur = int(rng.integers(6, 40))
        shape = rng.random()
        prof = np.zeros(dur)
        peak_at = int(dur * (0.15 if shape < 0.4 else 0.8 if shape < 0.7 else 0.5))
        for k in range(dur):
            prof[k] = max(0.0, 1.0 - abs(k - peak_at) / max(dur * 0.5, 1))
        prof *= rng.gamma(2.0, 4.0)
        prof *= (rng.random(dur) > 0.25)  # 间歇
        if shape >= 0.7 and dur > 20:     # 双峰
            prof[: dur // 2] += np.roll(prof[: dur // 2], dur // 4)
        r[t0:t0 + dur] += prof[: max(0, min(dur, n - t0))]
    times = [f"demo+{i:05d}h" for i in range(n)]
    return times, r


def count_peaks(w: np.ndarray, min_sep: int = 6, rel_height: float = 0.3) -> int:
    peak = w.max()
    if peak <= 0:
        return 0
    idx = [i for i in range(1, len(w) - 1)
           if w[i] >= w[i - 1] and w[i] > w[i + 1] and w[i] >= rel_height * peak]
    kept = []
    for i in idx:
        if not kept or i - kept[-1] >= min_sep:
            kept.append(i)
        elif w[i] > w[kept[-1]]:
            kept[-1] = i
    return len(kept)


def classify(w: np.ndarray) -> tuple[str, int]:
    pk = int(np.argmax(w))
    if pk < 16:
        pos = "front"
    elif pk >= 32:
        pos = "back"
    else:
        pos = "mid"
    return pos, count_peaks(w)


def select_events(r: np.ndarray, n_events: int, min_total_mm: float,
                  max_overlap_hours: int):
    n = len(r)
    cands = []
    for s in range(0, n - N_STEPS + 1):
        w = r[s:s + N_STEPS]
        if np.any(np.isnan(w)):
            continue
        tot = float(w.sum())
        if tot < min_total_mm:
            continue
        cands.append((tot, s))
    cands.sort(reverse=True)

    picked: list[int] = []
    for _, s in cands:
        if all(abs(s - p) >= N_STEPS - max_overlap_hours for p in picked):
            picked.append(s)
    if len(picked) < n_events:
        print(f"[WARN] 满足条件的不重叠事件仅 {len(picked)} 场 (< 请求的 {n_events});"
              " 可降低 --min-total-mm 或换更长的记录/多站数据。")
    picked = picked[: max(n_events * 3, n_events)]  # 留出形态挑选余量

    info = []
    for s in picked:
        w = r[s:s + N_STEPS]
        pos, npk = classify(w)
        info.append(dict(start=s, total=float(w.sum()), pos=pos, n_peaks=npk))

    # 覆盖性挑选: front/back/multi-peak 各争取 >= 2, 其余按总量补齐
    chosen: list[dict] = []

    def take(pred, k):
        got = 0
        for e in info:
            if got >= k:
                break
            if e not in chosen and pred(e):
                chosen.append(e)
                got += 1

    take(lambda e: e["pos"] == "front", 2)
    take(lambda e: e["pos"] == "back", 2)
    take(lambda e: e["n_peaks"] >= 2, 2)
    take(lambda e: True, n_events - len(chosen))
    chosen = chosen[:n_events]
    chosen.sort(key=lambda e: e["start"])
    return chosen


def main():
    ap = argparse.ArgumentParser(description="E5: 真实降雨 -> 外部测试集参数表")
    ap.add_argument("--input", type=str, default=None, help="小时降雨记录文件")
    ap.add_argument("--format", choices=["csv", "isd-lite"], default="csv")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--precip-col", default="precip_mm")
    ap.add_argument("--station", default="UNKNOWN", help="数据来源/站点标识, 写入元数据")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-events", type=int, default=9, help="事件数 (规格: 8-10)")
    ap.add_argument("--min-total-mm", type=float, default=20.0)
    ap.add_argument("--max-overlap-hours", type=int, default=12)
    ap.add_argument("--scale-mode", choices=["pool", "event"], default="pool")
    ap.add_argument("--demo", action="store_true",
                    help="合成占位数据自检管线 (输出打 DEMO 标记, 严禁用于论文)")
    args = ap.parse_args()

    if not (8 <= args.n_events <= 10):
        print(f"[WARN] 规格建议 8-10 场事件, 当前 --n-events={args.n_events}")

    if args.demo:
        times, r = make_demo_series()
        source = "DEMO_SYNTHETIC_DO_NOT_USE_IN_PAPER"
    else:
        if not args.input:
            sys.exit("[FAIL] 需要 --input (或用 --demo 自检)")
        path = Path(args.input)
        if not path.exists():
            sys.exit(f"[FAIL] 输入文件不存在: {path}")
        if args.format == "isd-lite":
            times, r = read_isd_lite(path)
        else:
            times, r = read_generic_csv(path, args.time_col, args.precip_col)
        source = f"{args.station} ({path.name})"
    n_nan = int(np.isnan(r).sum())
    print(f"读入 {len(r)} 小时记录, 缺测 {n_nan} ({100*n_nan/max(len(r),1):.1f}%), "
          f"来源: {source}")

    events = select_events(r, args.n_events, args.min_total_mm, args.max_overlap_hours)
    if not events:
        sys.exit("[FAIL] 未选出任何事件")

    raw = np.stack([r[e["start"]:e["start"] + N_STEPS] for e in events])  # (E,48) mm/h
    r_ref_pool = raw.max()

    q = np.empty_like(raw)
    for i in range(len(events)):
        r_ref = r_ref_pool if args.scale_mode == "pool" else raw[i].max()
        q[i] = Q_MIN + (Q_MAX - Q_MIN) * raw[i] / r_ref
        events[i]["raw_peak_mm_per_h"] = float(raw[i].max())
        events[i]["scale_factor_cmh_per_mmh"] = float((Q_MAX - Q_MIN) / r_ref)
        events[i]["start_time"] = times[events[i]["start"]] if events[i]["start"] < len(times) else ""
    assert q.min() >= Q_MIN - 1e-12 and q.max() <= Q_MAX + 1e-12, "重标定越出训练包络!"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- events_raw.csv (供 stats_compare / 附图) ---
    with open(out_dir / "events_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "start_time", "pos_class", "n_peaks", "total_mm"]
                   + [f"r_mm_{i:02d}" for i in range(N_STEPS)])
        for i, e in enumerate(events):
            w.writerow([i, e["start_time"], e["pos"], e["n_peaks"],
                        f"{e['total']:.2f}"] + [f"{v:.3f}" for v in raw[i]])

    # --- lhs_params.csv (主仓库风格, 每事件 x 3 土壤) ---
    q_cols = [f"q_top_{i:02d}" for i in range(N_STEPS)]
    header = (["sample_id", "theta_r", "theta_s", "alpha", "n_vg", "K_s", "D_L", "c_top"]
              + q_cols
              + ["q_top_mean", "q_top_peak", "q_top_std", "q_top_total", "q_top_peak_time",
                 "split", "split_mode", "is_ood", "event_id", "soil_class", "event_source"])
    rows = []
    sid = 0
    for i, e in enumerate(events):
        qi = q[i]
        derived = [qi.mean(), qi.max(), qi.std(), qi.sum() * FORCING_DT,
                   float(np.argmax(qi)) * FORCING_DT]  # 与 sampling.py:235-239 定义一致
        for soil_name, sp in SOILS.items():
            rows.append([sid, sp["theta_r"], sp["theta_s"], sp["alpha"], sp["n_vg"],
                         sp["K_s"], D_L, C_TOP]
                        + [f"{v:.6f}" for v in qi]
                        + [f"{d:.6f}" for d in derived]
                        + ["test", "external_real_rain", True, i, soil_name, source])
            sid += 1
    with open(out_dir / "lhs_params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    meta = dict(
        source=source, demo=bool(args.demo), n_events=len(events),
        n_samples=sid, scale_mode=args.scale_mode,
        pool_ref_peak_mm_per_h=float(r_ref_pool),
        envelope_cm_per_h=[Q_MIN, Q_MAX],
        forcing_dt_hours=FORCING_DT,
        soils=SOILS, D_L=D_L, c_top=C_TOP,
        soil_reference="Carsel & Parrish (1988) WRR 24(5):755-769, K_s converted cm/d -> cm/h",
        events=events,
        class_counts=dict(
            front=sum(e["pos"] == "front" for e in events),
            back=sum(e["pos"] == "back" for e in events),
            mid=sum(e["pos"] == "mid" for e in events),
            multi_peak=sum(e["n_peaks"] >= 2 for e in events),
        ),
    )
    with open(out_dir / "events_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] {len(events)} 场事件 x 3 土壤 = {sid} 个外部测试样本")
    print(f"     形态覆盖: {meta['class_counts']}")
    print(f"     输出: {out_dir}/lhs_params.csv, events_raw.csv, events_meta.json")
    if args.demo:
        print("[WARN] DEMO 模式输出仅用于管线自检, 严禁进入论文!")


if __name__ == "__main__":
    main()
