#!/usr/bin/env python3
"""E3 汇总: 收集 search / final runs 的 B1_results.json → 排名 CSV.

纯标准库 (json/csv/glob/re), 本地可跑 —— 前提是 runs_revision 结果已从集群
同步回本地且非 iCloud dataless 占位 (若本地 open 卡住说明是 iCloud dataless
占位, 请在 Finder 中下载该目录, 或直接在集群上运行本脚本)。

用法:
  python3 collect_search.py             # 搜索 runs: e3_fno_search_* → search_ranking.csv
  python3 collect_search.py --final     # final runs: e3_fno_final_*  → final_summary.csv
  python3 collect_search.py --runs-root <路径>   # 覆盖默认 runs_revision 根目录

指标来源 (REPO_FACTS.md §4):
  B1_results.json: results["fno"]["h"/"c"]["rel_l2"/"r2"]["mean"/"median"/...]
  fno_history.json: best_epoch / best_val_loss / training_time_min / model_params
排名键: c_rel_l2_mean 升序 (concentration 是主文弱项指标, 与主文表述一致)。
"""

from pathlib import Path
import argparse
import csv
import glob
import json
import os
import re
import sys

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUNS_ROOT = os.path.join(REPO, "experiments", "qtop_func", "runs_revision")

SEARCH_RE = re.compile(
    r"e3_fno_search_fno_w(?P<w>\d+)_m(?P<mz>\d+)x(?P<mt>\d+)_lr(?P<lr>[0-9e.\-]+)_p(?P<p>\d+)$")
FINAL_RE = re.compile(r"e3_fno_final_(?P<sc>iid|ood_peak|ood_peak_time)_s(?P<seed>\d+)$")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"  warn: cannot read {path}: {e}", file=sys.stderr)
        return None


def stat(d, field, key, sub):
    """d[field][key][sub], 缺失返回 ''。"""
    try:
        return round(float(d[field][key][sub]), 6)
    except (KeyError, TypeError, ValueError):
        return ""


def collect_one(run_dir):
    exp = os.path.join(run_dir, "results", "experiments")
    b1 = load_json(os.path.join(exp, "B1_results.json"))
    if not b1 or "fno" not in b1:
        return None
    fno = b1["fno"]
    row = {
        "h_rel_l2_mean": stat(fno, "h", "rel_l2", "mean"),
        "h_rel_l2_median": stat(fno, "h", "rel_l2", "median"),
        "c_rel_l2_mean": stat(fno, "c", "rel_l2", "mean"),
        "c_rel_l2_median": stat(fno, "c", "rel_l2", "median"),
        "h_r2_median": stat(fno, "h", "r2", "median"),
        "c_r2_median": stat(fno, "c", "r2", "median"),
    }
    hist = load_json(os.path.join(exp, "fno_history.json")) or {}
    row["best_epoch"] = hist.get("best_epoch", "")
    row["best_val_loss"] = hist.get("best_val_loss", "")
    row["training_time_min"] = (round(hist["training_time_min"], 1)
                                if "training_time_min" in hist else "")
    row["model_params_M"] = (round(hist["model_params"] / 1e6, 2)
                             if "model_params" in hist else "")
    b3 = load_json(os.path.join(exp, "B3_results.json"))
    if b3 and "fno" in b3:
        fe = b3["fno"].get("front_error_mean_cm")
        row["front_error_mean_cm"] = round(fe, 2) if fe is not None else ""
    else:
        row["front_error_mean_cm"] = ""
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--final", action="store_true",
                    help="汇总 e3_fno_final_* (默认汇总 e3_fno_search_*)")
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    args = ap.parse_args()

    pattern = "e3_fno_final_*" if args.final else "e3_fno_search_*"
    out_path = args.out or os.path.join(
        PKG_DIR, "final_summary.csv" if args.final else "search_ranking.csv")

    rows = []
    run_dirs = sorted(glob.glob(os.path.join(args.runs_root, pattern)))
    if not run_dirs:
        print(f"No runs matched {os.path.join(args.runs_root, pattern)}")
        sys.exit(1)

    for rd in run_dirs:
        name = os.path.basename(rd)
        metrics = collect_one(rd)
        if metrics is None:
            print(f"  skip (no B1 fno results yet): {name}", file=sys.stderr)
            continue
        row = {"run": name}
        m = SEARCH_RE.match(name)
        if m:
            row.update(width=int(m.group("w")), modes_z=int(m.group("mz")),
                       modes_t=int(m.group("mt")), lr=m.group("lr"),
                       padding=int(m.group("p")))
        mf = FINAL_RE.match(name)
        if mf:
            row.update(scenario=mf.group("sc"), seed=int(mf.group("seed")))
        row.update(metrics)
        rows.append(row)

    if not rows:
        print("No completed runs to summarize.")
        sys.exit(1)

    sort_key = (("scenario", "seed") if args.final else ("c_rel_l2_mean",))

    def _k(v):
        """归一化排序值, 避免 str/float 混型比较抛 TypeError.

        数值排前 (按大小), 其余 (缺失 ''、目录名不匹配导致缺键 None、字符串)
        排后 (按字符串序)。"""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return (1, v, "")
        return (2, 0.0, str(v))

    rows.sort(key=lambda r: tuple(_k(r.get(k)) for k in sort_key))
    if not args.final:
        for i, r in enumerate(rows, 1):
            r["rank"] = i

    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")

    # 终端速览 (前 8 行)
    show = rows[:8]
    cols = ([c for c in ("rank", "width", "modes_z", "modes_t", "lr", "padding",
                         "c_rel_l2_mean", "h_rel_l2_mean", "model_params_M")]
            if not args.final else
            [c for c in ("scenario", "seed", "c_rel_l2_mean", "h_rel_l2_mean",
                         "front_error_mean_cm")])
    header = "  ".join(f"{c:>14}" for c in cols)
    print(header)
    for r in show:
        print("  ".join(f"{str(r.get(c, '')):>14}" for c in cols))


if __name__ == "__main__":
    main()
