#!/usr/bin/env python3
"""E6-A: 用已有三场景数据源/模型 run 的 lhs_params.csv（纯文本），按
q_top_peak / q_top_peak_time 把测试集重新切成严重度分档
（p00-p90 参考档 / p90-p95 / p95-p98 / p98-p100），输出各分档的
sample_id 清单 + 与 B1_<model>_predictions.npz 行号对齐的 row_indices。

分位阈值口径（输出字段 n_csv_rows）:
  p90/p95/p98 在该 run lhs_params.csv 的**已分配样本**（n_csv_rows 行 =
  concat(train,val,test)）上取分位。
  - IID 场景: CSV = 完整采样池，与 01_generate_params 的口径完全一致。
  - OOD 场景: 01_generate_params.py:86 只保存 concat(train,val,test)，未入选
    的候选样本不写入 CSV（ood_peak 池 1024 → CSV 992 行，ood_peak_time 池
    1152 → CSV 992 行），故本脚本的分位阈值相对 01 在完整池上取的 0.9 分位
    存在轻微偏移，属近似。论文取用时只报**档位相对严重度**，不宣称与 01 的
    OOD 切分阈值完全同口径。

零仿真、numpy-only、本地可跑（前提: 相关 lhs_params.csv / qc_summary.json
已从 iCloud 物化，先跑 scripts/materialize_check.sh）。

行号对齐规则（与 03_process_data.py / postprocess.load_raw_data 一致）:
  test.npz 的第 i 行 = { split=='test' 且 raw npz 存在 且 QC 通过 } 的
  sample_id 升序排列的第 i 个。
  - raw 是否存在: 用 os.listdir(data/raw) 的文件名判断（不读内容，iCloud 安全）。
  - QC 剔除: metadata/qc_summary.json 的 failed_sample_ids；该 run 缺此文件时
    用 manifest 里 qc_fallback_run_dir 的（数据同 seed 生成/复制）。
  - 最终一致性由 eval_subsets.py 以 B1 npz 的 N 断言兜底。

用法:
  python regrade_tails.py \
      --manifest <包目录>/models_manifest.json \
      --out <包目录>/outputs/A_subsets.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

FORCING_DT_HOURS = 1.0  # simulation_time 48h / n_steps 48
N_QTOP_STEPS = 48
BIN_EDGES = (0.90, 0.95, 0.98)
BIN_LABELS = ("p00_p90", "p90_p95", "p95_p98", "p98_p100")


def read_lhs_csv(path: Path):
    """读 lhs_params.csv -> (sample_ids, split, q_series (N,48))，纯 stdlib+numpy。"""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        q_cols = [f"q_top_{i:02d}" for i in range(N_QTOP_STEPS)]
        for col in ("sample_id", "split", *q_cols):
            if col not in reader.fieldnames:
                raise KeyError(f"{path}: 缺少列 {col}")
        ids, splits, rows = [], [], []
        stored_peak, stored_peak_time = [], []
        for row in reader:
            ids.append(int(row["sample_id"]))
            splits.append(row["split"])
            rows.append([float(row[c]) for c in q_cols])
            stored_peak.append(float(row["q_top_peak"]) if row.get("q_top_peak") else np.nan)
            stored_peak_time.append(
                float(row["q_top_peak_time"]) if row.get("q_top_peak_time") else np.nan
            )
    q = np.asarray(rows, dtype=np.float64)
    return (
        np.asarray(ids),
        np.asarray(splits),
        q,
        np.asarray(stored_peak),
        np.asarray(stored_peak_time),
    )


def severity_values(q_series: np.ndarray) -> dict:
    return {
        "q_top_peak": q_series.max(axis=1),
        "q_top_peak_time": q_series.argmax(axis=1) * FORCING_DT_HOURS,
    }


def load_qc_failed_ids(run_dir: Path, fallback_run_dir: Path | None) -> tuple[list[int], str]:
    for d, tag in ((run_dir, "self"), (fallback_run_dir, "fallback")):
        if d is None:
            continue
        p = d / "metadata" / "qc_summary.json"
        if p.exists():
            with open(p) as f:
                qc = json.load(f)
            return [int(x) for x in qc.get("failed_sample_ids", [])], f"{tag}:{p}"
    return [], "missing (assumed 0 failed; eval_subsets.py 会用 B1 npz 的 N 断言)"


def raw_present_ids(run_dir: Path) -> set[int] | None:
    raw_dir = run_dir / "data" / "raw"
    if not raw_dir.is_dir():
        return None
    ids = set()
    for name in os.listdir(raw_dir):
        if name.startswith("sample_") and name.endswith(".npz"):
            try:
                ids.add(int(name[len("sample_"):-len(".npz")]))
            except ValueError:
                pass
    return ids or None


def infer_processed_test_ids(
    run_dir: Path, csv_path: Path
) -> tuple[np.ndarray | None, str]:
    """用 processed branch_inputs 反推其对应的 CSV sample_id 与实际行序。

    一些历史 run 没有保存 qc_summary.json，且 raw 目录仍包含随后被 QC
    排除的样本。此时仅凭 raw 文件存在性会高估 test.npz 的 N。branch_inputs
    是 CSV 输入按 scaler.npz 做 min-max 变换后的确定性结果，因此可以用全部
    非常数输入列做一一匹配，恢复 test.npz/B1 npz 的精确 sample_id 行序。
    """
    test_path = run_dir / "data" / "processed" / "test.npz"
    scaler_path = run_dir / "data" / "processed" / "scaler.npz"
    if not (test_path.exists() and scaler_path.exists()):
        return None, "processed test/scaler missing"

    try:
        test = np.load(test_path, allow_pickle=True)
        scaler = np.load(scaler_path)
        branch = np.asarray(test["branch_inputs"], dtype=np.float64)
        branch_keys = [str(x) for x in test["branch_keys"].tolist()]
        input_min = np.asarray(scaler["input_min"], dtype=np.float64)
        input_max = np.asarray(scaler["input_max"], dtype=np.float64)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            csv_fields = set(reader.fieldnames or [])
            test_rows = [row for row in reader if row["split"] == "test"]

        used = [(j, key) for j, key in enumerate(branch_keys) if key in csv_fields]
        if not used or not test_rows:
            return None, "no shared input columns or test rows"

        col_idx = np.asarray([j for j, _ in used], dtype=int)
        raw = np.asarray(
            [[float(row[key]) for _, key in used] for row in test_rows],
            dtype=np.float64,
        )
        candidate_ids = np.asarray(
            [int(row["sample_id"]) for row in test_rows], dtype=int
        )
        norm = (raw - input_min[col_idx]) / (
            input_max[col_idx] - input_min[col_idx] + 1e-8
        )
        observed = branch[:, col_idx]

        # 数据规模至多百量级；全对全比较能同时验证最近邻距离和一一性。
        max_abs = np.max(
            np.abs(observed[:, None, :] - norm[None, :, :]), axis=2
        )
        best = np.argmin(max_abs, axis=1)
        best_dist = max_abs[np.arange(len(observed)), best]
        if len(set(best.tolist())) != len(best):
            return None, "processed-to-CSV match is not one-to-one"
        if best_dist.size and float(best_dist.max()) > 1e-5:
            return None, f"processed-to-CSV max mismatch={float(best_dist.max()):.3g}"
        return candidate_ids[best], (
            f"branch_inputs/scaler exact match; max_abs="
            f"{float(best_dist.max()) if best_dist.size else 0.0:.3g}"
        )
    except Exception as exc:
        return None, f"processed-to-CSV inference failed: {exc}"


def process_run(entry: dict, repo: Path) -> dict | None:
    run_dir = repo / entry["run_dir"] if not os.path.isabs(entry["run_dir"]) else Path(entry["run_dir"])
    csv_path = run_dir / "data" / "parameters" / "lhs_params.csv"
    if not csv_path.exists():
        print(f"[WARN] 跳过 {entry['key']}: 缺 {csv_path}")
        return None

    ids, splits, q, stored_peak, stored_pt = read_lhs_csv(csv_path)
    sev = severity_values(q)

    # 与 CSV 存储的派生列对账（若存在）
    for name, stored in (("q_top_peak", stored_peak), ("q_top_peak_time", stored_pt)):
        ok = np.isfinite(stored)
        if ok.any() and not np.allclose(sev[name][ok], stored[ok], atol=1e-6):
            n_bad = int((~np.isclose(sev[name][ok], stored[ok], atol=1e-6)).sum())
            print(f"[WARN] {entry['key']}: 重算 {name} 与 CSV 存储列不一致 ({n_bad} 行)，以重算值为准")

    fb = entry.get("qc_fallback_run_dir")
    fb_dir = (repo / fb) if fb else None
    qc_failed, qc_source = load_qc_failed_ids(run_dir, fb_dir)
    qc_failed_set = set(qc_failed)

    test_mask = splits == "test"
    test_ids = ids[test_mask]
    inferred_ids, alignment_source = infer_processed_test_ids(run_dir, csv_path)
    if inferred_ids is not None:
        # 保留 processed test.npz 的实际行序；这也是 B1/B3 npz 的行序。
        kept_ids = inferred_ids
    else:
        present = raw_present_ids(run_dir)
        if present is None and fb_dir is not None:
            present = raw_present_ids(fb_dir)
        keep = np.ones(len(test_ids), dtype=bool)
        if present is not None:
            keep &= np.array([sid in present for sid in test_ids])
        keep &= np.array([sid not in qc_failed_set for sid in test_ids])
        kept_ids = np.sort(test_ids[keep])
        alignment_source = (
            "raw presence + qc_summary; processed inference unavailable: "
            f"{alignment_source}"
        )

    id_to_full = {sid: i for i, sid in enumerate(ids)}
    row_of = {sid: r for r, sid in enumerate(kept_ids)}

    axes_out = {}
    for axis in entry.get("axes", ["q_top_peak", "q_top_peak_time"]):
        full_vals = sev[axis]
        thresholds = {
            f"p{int(qe*100)}": float(np.percentile(full_vals, qe * 100)) for qe in BIN_EDGES
        }
        edges = [thresholds[f"p{int(qe*100)}"] for qe in BIN_EDGES]
        bins = {label: {"sample_ids": [], "row_indices": []} for label in BIN_LABELS}
        for sid in kept_ids:
            v = full_vals[id_to_full[sid]]
            if v < edges[0]:
                label = BIN_LABELS[0]
            elif v < edges[1]:
                label = BIN_LABELS[1]
            elif v < edges[2]:
                label = BIN_LABELS[2]
            else:
                label = BIN_LABELS[3]
            bins[label]["sample_ids"].append(int(sid))
            bins[label]["row_indices"].append(int(row_of[sid]))
        axes_out[axis] = {"thresholds": thresholds, "bins": bins}

    return {
        "key": entry["key"],
        "scenario": entry["scenario"],
        "family": entry["family"],
        "primary_axis": entry.get("primary_axis"),
        "run_dir": str(run_dir),
        "models": entry["models"],
        "n_csv_rows": int(len(ids)),
        "quantile_basis": "csv_assigned_rows(train+val+test); OOD 场景相对完整池分位为近似",
        "n_test_rows": int(len(kept_ids)),
        "row_sample_ids": [int(x) for x in kept_ids],
        "row_alignment_source": alignment_source,
        "qc_failed_source": qc_source,
        "n_qc_failed_in_test": (
            int(len(set(test_ids.tolist()) & qc_failed_set))
            if not qc_source.startswith("missing")
            else None
        ),
        "n_test_excluded_before_processed": int(len(test_ids) - len(kept_ids)),
        "axes": axes_out,
    }


def main():
    pkg_default = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(pkg_default / "models_manifest.json"))
    ap.add_argument("--out", default=str(pkg_default / "outputs" / "A_subsets.json"))
    ap.add_argument("--include-optional", action="store_true",
                    help="包含 manifest 中 optional=true 的 run（多种子 RegShift）")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    repo = Path(manifest["repo"])

    out_runs = {}
    for entry in manifest["part_A_runs"]:
        if entry.get("optional") and not args.include_optional:
            continue
        res = process_run(entry, repo)
        if res is not None:
            out_runs[entry["key"]] = res
            counts = {
                axis: {b: len(v["sample_ids"]) for b, v in ax["bins"].items()}
                for axis, ax in res["axes"].items()
            }
            print(f"[OK] {entry['key']}: n_test_rows={res['n_test_rows']}, bins={counts}")

    out = {
        "kind": "E6_A_severity_subsets",
        "bin_edges": list(BIN_EDGES),
        "bin_labels": list(BIN_LABELS),
        "forcing_dt_hours": FORCING_DT_HOURS,
        "runs": out_runs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n输出: {out_path}  ({len(out_runs)} 个 run)")

    # 分档样本数汇总 CSV
    sum_path = out_path.with_name("A_subset_counts.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_key", "scenario", "family", "axis", "bin", "n_samples"])
        for key, res in out_runs.items():
            for axis, ax in res["axes"].items():
                for label, b in ax["bins"].items():
                    w.writerow([key, res["scenario"], res["family"], axis, label,
                                len(b["sample_ids"])])
    print(f"输出: {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
