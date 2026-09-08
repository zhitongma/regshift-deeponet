#!/usr/bin/env python3
"""E1 results collector (numpy-only, runs locally on the Mac).

Reads ONLY small text JSON files (B1_results.json / B3_results.json /
m1_history.json / m2_history.json / fnn_history.json). It NEVER opens .npz/.pt
files -- many binaries in the iCloud-synced repo are dataless placeholders and
reading them hangs forever.

Collected runs:
  1. New E1 runs:        <repo>/experiments/qtop_func/runs_revision/e1_*_s{42,123,456}
  2. Original RegShift:  <repo>/experiments/qtop_func/runs/g4_regshift_{iid,ood_peak,ood_time}_s{42,123,456}
     (= RegShift @ high budget, published)
     CAVEAT: g4 snapshots carry the historical bound bug -- transform_bound_*
     sat at TOP-LEVEL `shift_deeponet:` and is silently ignored by the current
     code path, so this reference was trained as an UNBOUNDED model (E0, H1/H2
     pending). It stays labelled group=orig_regshift for the published numbers.
  2b. E0 bound-FIXED RegShift @ high budget rerun, if present:
     <repo>/experiments/qtop_func/runs_revision/e0_regshift_*_s*
     (group=regshift_high_fixed; preferred RegShift@high reference once E0
     lands; other naming -> pass via --extra)
  3. Original DeepONet-family baselines:
     <repo>/experiments/qtop_func/runs_multiseed/
     {iid_medium_v1,ood_peak_medium_v1,ood_peak_time_medium_v2}/seed_*
     (= baselines @ low budget, published; NOTE seeds there are 42/43/44;
     these dirs contain ONLY m1/m2_data/m2_score/grid_fnn -- no FNN, no
     unbounded Shift)
  3b. Original FNN @ low budget:  <repo>/experiments/qtop_func/runs/
      fnn_baseline_{iid,ood_peak}_v1   (single seed; NO ood_peak_time run)
      Original unbounded Shift @ low budget:  <repo>/experiments/qtop_func/
      runs/boost_A4_{iid,ood}          (single seed; ood == ood_peak scenario;
      NO ood_peak_time run)
  4. Any extra run dirs via --extra <dir> (repeatable).

Outputs (into --out-dir, default this package directory):
  e1_results_long.csv        one row per (run, model)
  e1_budget_comparison.csv   original budget vs parity budget, per scenario/arch
  e1_dataless_files.txt      iCloud placeholder JSONs skipped (if any)
  and a plain-text comparison table on stdout.

Usage:  python3 collect_results.py [--repo <repo>] [--extra <run_dir> ...]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))
PKG = Path(__file__).resolve().parent

SCEN_ALIASES = [  # (token in run path, canonical scenario) -- order matters
    ("ood_peak_time", "ood_peak_time"),
    ("ood_time", "ood_peak_time"),
    ("peak_time_medium", "ood_peak_time"),
    ("ood_peak", "ood_peak"),
    ("iid", "iid"),
]


def infer_scenario(path_str):
    for token, canon in SCEN_ALIASES:
        if token in path_str:
            return canon
    return "unknown"


def infer_seed(path_str):
    name = Path(path_str).name
    for pref in ("_s", "seed_"):
        idx = name.rfind(pref)
        if idx >= 0:
            tail = name[idx + len(pref):]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                return int(digits)
    return None


DATALESS = []  # iCloud placeholder files encountered (never opened)


def is_dataless(path):
    """iCloud 'dataless' placeholder: size > 0 but zero allocated blocks.

    Opening such a file BLOCKS until iCloud downloads it (can hang forever
    offline), so we must stat-check BEFORE any open() -- even for small JSON.
    """
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_size > 0 and getattr(st, "st_blocks", 1) == 0


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    if is_dataless(path):
        DATALESS.append(str(path))
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def training_stats(run_dir, model):
    """(stop_epoch, best_epoch, train_min) from history JSONs (text only)."""
    exp = run_dir / "results" / "experiments"
    if model == "m1":
        h = load_json(exp / "m1_history.json")
        if h:
            hist = h.get("history", {})
            stop = h.get("total_epochs") or len(hist.get("train_loss", []))
            return stop, h.get("best_epoch"), h.get("training_time_min")
    elif model.startswith("m2"):
        h = load_json(exp / "m2_history.json")
        if h:
            hist = h.get("history", {})
            stop = h.get("total_epochs") or len(hist.get("train_loss", []))
            best = h.get("best_data_epoch") or h.get("best_epoch")
            return stop, best, h.get("training_time_min")
    elif model == "fnn":
        h = load_json(exp / "fnn_history.json")
        if h:
            stop = len(h.get("train_loss", [])) or None
            return stop, h.get("best_epoch"), h.get("training_time_min")
    return None, None, None


def collect_run(run_dir, group, budget, models=None, arch=None, scenario=None):
    """Yield one row dict per model found in B1_results.json.

    models   -- optional set of model names to keep (others in the same run
                are ignored, e.g. only "fnn" from fnn_baseline_* runs);
    arch     -- optional architecture label override (stored as arch_hint,
                wins over the name-based heuristic in arch_of());
    scenario -- optional scenario override for run names that the token
                heuristic cannot classify (e.g. boost_A4_ood == ood_peak).
    """
    run_dir = Path(run_dir)
    exp = run_dir / "results" / "experiments"
    b1 = load_json(exp / "B1_results.json")
    if not b1:
        print(f"  [skip] no B1_results.json: {run_dir}", file=sys.stderr)
        return
    b3 = load_json(exp / "B3_results.json") or {}
    scen = scenario or infer_scenario(str(run_dir))
    seed = infer_seed(str(run_dir))
    for model, res in b1.items():
        if models is not None and model not in models:
            continue
        try:
            h_l2 = res["h"]["rel_l2"]["median"]
            c_l2 = res["c"]["rel_l2"]["median"]
            h_l2_mean = res["h"]["rel_l2"]["mean"]
            c_l2_mean = res["c"]["rel_l2"]["mean"]
        except (KeyError, TypeError):
            continue
        hydro = b3.get(model, {}) if isinstance(b3, dict) else {}
        stop, best, tmin = training_stats(run_dir, model)
        yield {
            "group": group,
            "budget": budget,
            "arch_hint": arch,
            "scenario": scen,
            "run": run_dir.name,
            "model": model,
            "seed": seed,
            "h_rel_l2_median": h_l2,
            "c_rel_l2_median": c_l2,
            "h_rel_l2_mean": h_l2_mean,
            "c_rel_l2_mean": c_l2_mean,
            "front_error_mean_cm": hydro.get("front_error_mean_cm"),
            "bt_mae_hours": hydro.get("bt_mae_hours"),
            "stop_epoch": stop,
            "best_epoch": best,
            "train_wallclock_min": tmin,
        }


def e1_budget(run_name):
    return "low" if "lowbudget" in run_name else ("high" if "highbudget" in run_name else "?")


def gather(repo, extras):
    rows = []
    # 1. new E1 runs
    rev = repo / "experiments" / "qtop_func" / "runs_revision"
    for d in sorted(rev.glob("e1_*_s*")) if rev.is_dir() else []:
        if d.is_dir():
            rows += list(collect_run(d, group="E1_parity", budget=e1_budget(d.name)))
    # 2. original RegShift (high budget)
    #    CAVEAT: g4 snapshots carry the historical bound bug (bounds at
    #    top-level `shift_deeponet:`, silently ignored) -> effectively an
    #    unbounded model at high budget; see E0 and the stdout footnote.
    runs = repo / "experiments" / "qtop_func" / "runs"
    for scen in ("iid", "ood_peak", "ood_time"):
        for seed in (42, 123, 456):
            d = runs / f"g4_regshift_{scen}_s{seed}"
            if d.is_dir():
                rows += list(collect_run(d, group="orig_regshift", budget="high"))
    # 2b. E0 bound-FIXED RegShift@high rerun (preferred reference if present)
    for d in sorted(rev.glob("e0_regshift_*_s*")) if rev.is_dir() else []:
        if d.is_dir():
            rows += list(collect_run(d, group="regshift_high_fixed",
                                     budget="high", arch="RegShift"))
    # 3. original DeepONet-family baselines (low budget), seeds 42/43/44
    #    (these dirs contain only m1/m2_data/m2_score/grid_fnn)
    ms = repo / "experiments" / "qtop_func" / "runs_multiseed"
    for scen_dir in ("iid_medium_v1", "ood_peak_medium_v1", "ood_peak_time_medium_v2"):
        base = ms / scen_dir
        for d in sorted(base.glob("seed_*")) if base.is_dir() else []:
            if d.is_dir():
                rows += list(collect_run(d, group="orig_baseline", budget="low"))
    # 3b. original FNN@low and unbounded Shift@low (single seed each; NO
    #     ood_peak_time run exists for either -- see README §2/§6)
    for scen, name in (("iid", "fnn_baseline_iid_v1"),
                       ("ood_peak", "fnn_baseline_ood_peak_v1")):
        d = runs / name
        if d.is_dir():
            rows += list(collect_run(d, group="orig_baseline", budget="low",
                                     models={"fnn"}, arch="FNN", scenario=scen))
        else:
            print(f"  [warn] missing original FNN@low reference: {d}", file=sys.stderr)
    for scen, name in (("iid", "boost_A4_iid"), ("ood_peak", "boost_A4_ood")):
        d = runs / name
        if d.is_dir():
            rows += list(collect_run(d, group="orig_baseline", budget="low",
                                     models={"m1", "m2_data"},
                                     arch="Shift(unbounded)", scenario=scen))
        else:
            print(f"  [warn] missing original Shift@low reference: {d}", file=sys.stderr)
    # 4. extras
    for x in extras:
        d = Path(x)
        if d.is_dir():
            rows += list(collect_run(d, group="extra", budget=e1_budget(d.name)))
        else:
            print(f"  [skip] --extra not a dir: {x}", file=sys.stderr)
    return rows


FIELDS = ["group", "budget", "scenario", "run", "model", "seed",
          "h_rel_l2_median", "c_rel_l2_median", "h_rel_l2_mean", "c_rel_l2_mean",
          "front_error_mean_cm", "bt_mae_hours",
          "stop_epoch", "best_epoch", "train_wallclock_min"]


def fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def arch_of(row):
    """Map a (group, run, model) row to an architecture label for comparison."""
    if row.get("arch_hint"):
        return row["arch_hint"]
    name = row["run"]
    for a in ("regshift", "deeponet", "shift", "fnn"):
        if f"e1_{a}_" in name:
            return {"regshift": "RegShift", "deeponet": "DeepONet",
                    "shift": "Shift(unbounded)", "fnn": "FNN"}[a]
    if row["group"] in ("orig_regshift", "regshift_high_fixed"):
        return "RegShift"
    if row["group"] == "orig_baseline":
        return {"m1": "DeepONet", "m2_data": "DeepONet", "fnn": "FNN",
                "grid_fnn": "GridFNN"}.get(row["model"], row["model"])
    return row["model"]


def aggregate(rows):
    """(arch, scenario, budget, group, model) -> seed-aggregated stats."""
    buckets = {}
    for r in rows:
        key = (arch_of(r), r["scenario"], r["budget"], r["group"], r["model"])
        buckets.setdefault(key, []).append(r)
    out = []
    for key, rs in sorted(buckets.items()):
        def agg(field):
            vals = [r[field] for r in rs if r[field] is not None]
            if not vals:
                return None, None
            a = np.asarray(vals, dtype=float)
            return float(np.mean(a)), (float(np.std(a)) if len(a) > 1 else 0.0)
        arch, scen, budget, group, model = key
        row = {"arch": arch, "scenario": scen, "budget": budget,
               "group": group, "model": model, "n_seeds": len(rs)}
        for f in ("h_rel_l2_median", "c_rel_l2_median",
                  "front_error_mean_cm", "bt_mae_hours",
                  "stop_epoch", "train_wallclock_min"):
            m, s = agg(f)
            row[f + "_mean"] = m
            row[f + "_std"] = s
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--extra", action="append", default=[],
                    help="extra run dir to include (repeatable)")
    ap.add_argument("--out-dir", default=str(PKG))
    args = ap.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = gather(repo, args.extra)
    if DATALESS:
        print(f"\nWARNING: {len(DATALESS)} JSON file(s) are iCloud dataless "
              "placeholders and were SKIPPED (opening them would hang).",
              file=sys.stderr)
        print("Materialize them first, e.g.:", file=sys.stderr)
        for p in DATALESS[:8]:
            print(f"  brctl download '{p}'", file=sys.stderr)
        if len(DATALESS) > 8:
            print(f"  ... and {len(DATALESS) - 8} more "
                  f"(see {out_dir / 'e1_dataless_files.txt'})",
                  file=sys.stderr)
        (out_dir / "e1_dataless_files.txt").write_text("\n".join(DATALESS) + "\n")
    if not rows:
        print("No results found. Have the cluster runs finished and synced back?")
        return 1

    long_csv = out_dir / "e1_results_long.csv"
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in FIELDS})
    print(f"wrote {long_csv} ({len(rows)} rows)")

    agg = aggregate(rows)
    comp_csv = out_dir / "e1_budget_comparison.csv"
    agg_fields = ["arch", "scenario", "budget", "group", "model", "n_seeds",
                  "h_rel_l2_median_mean", "h_rel_l2_median_std",
                  "c_rel_l2_median_mean", "c_rel_l2_median_std",
                  "front_error_mean_cm_mean", "front_error_mean_cm_std",
                  "bt_mae_hours_mean", "bt_mae_hours_std",
                  "stop_epoch_mean", "train_wallclock_min_mean"]
    with open(comp_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for r in agg:
            w.writerow({k: fmt(r.get(k)) for k in agg_fields})
    print(f"wrote {comp_csv} ({len(agg)} rows)")

    # ---- plain-text comparison table: original budget vs parity budget ----
    print("\n=== E1 budget-parity comparison (c rel-L2 median, mean±std over seeds) ===")
    print("Convention: m2_data = final model of the two-stage pipeline; m1 shown for reference.")
    hdr = f"{'scenario':<15}{'arch':<18}{'model':<10}{'budget':<7}{'group':<21}" \
          f"{'c-L2':<16}{'h-L2':<16}{'front(cm)':<12}{'bt(h)':<10}{'ep_stop':<8}{'min':<8}"
    print(hdr)
    print("-" * len(hdr))
    for scen in ("iid", "ood_peak", "ood_peak_time"):
        for r in agg:
            if r["scenario"] != scen:
                continue
            if r["model"] not in ("m1", "m2_data", "fnn"):
                continue

            def pm(f, nd=4):
                m, s = r.get(f + "_mean"), r.get(f + "_std")
                if m is None:
                    return ""
                return f"{m:.{nd}f}±{s:.{nd}f}" if s is not None else f"{m:.{nd}f}"
            print(f"{scen:<15}{r['arch']:<18}{r['model']:<10}{r['budget']:<7}{r['group']:<21}"
                  f"{pm('c_rel_l2_median'):<16}{pm('h_rel_l2_median'):<16}"
                  f"{pm('front_error_mean_cm', 2):<12}{pm('bt_mae_hours', 2):<10}"
                  f"{fmt(r.get('stop_epoch_mean'), 0):<8}{fmt(r.get('train_wallclock_min_mean'), 1):<8}")
        print("-" * len(hdr))
    print("\nAcceptance check (README §验收标准):")
    print("  A) RegShift@low-budget still beats every original baseline per scenario;")
    print("  B) baselines@high-budget improve but remain behind RegShift, and unbounded")
    print("     Shift still degrades on ood_peak_time. Record ANY counterexample honestly.")
    print("\nReference caveats (see README §2 / E0):")
    print("  * group=orig_regshift (g4_*) carries the historical bound bug: the g4 config")
    print("    snapshots put transform_bound_* at TOP-LEVEL `shift_deeponet:`, which the")
    print("    current code path silently ignores -> that 'RegShift@high' reference was")
    print("    effectively trained UNBOUNDED. When group=regshift_high_fixed rows exist")
    print("    (E0 bound-fixed rerun, runs_revision/e0_regshift_*_s*), prefer them as the")
    print("    RegShift@high reference and footnote the g4 numbers; confirm with E0 (H1/H2).")
    print("  * Original FNN@low (fnn_baseline_{iid,ood_peak}_v1) and unbounded Shift@low")
    print("    (boost_A4_{iid,ood}) are SINGLE-SEED and have NO ood_peak_time run: those")
    print("    direction-B cells have no historical reference (report the E1 parity value")
    print("    or schedule a low-budget rerun).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
