"""
HYDRUS 文本输出解析工具。

当前项目只需要从工作目录中提取:
  - T_LEVEL.OUT: 水量边界通量与累计量
  - solute1.out: 溶质边界通量与累计量
  - BALANCE.OUT: HYDRUS 自身的质量平衡误差
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


TLEVEL_COLUMNS = [
    "Time",
    "rTop",
    "rRoot",
    "vTop",
    "vRoot",
    "vBot",
    "sum(rTop)",
    "sum(rRoot)",
    "sum(vTop)",
    "sum(vRoot)",
    "sum(vBot)",
    "hTop",
    "hRoot",
    "hBot",
    "RunOff",
    "sum(RunOff)",
    "Volume",
    "sum(Infil)",
    "sum(Evap)",
    "TLevel",
    "Cum(WTrans)",
    "SnowLayer",
]

SOLUTE_COLUMNS = [
    "Time",
    "cvTop",
    "cvBot",
    "Sum(cvTop)",
    "Sum(cvBot)",
    "cvCh0",
    "cvCh1",
    "cTop",
    "cRoot",
    "cBot",
    "cvRoot",
    "Sum(cvRoot)",
    "Sum(cvNEql)",
    "TLevel",
    "cGWL",
    "cRunOff",
    "Sum(cRunOff)",
]


def _read_numeric_table(path: str | Path, columns: list[str], anchor: str) -> pd.DataFrame:
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    rows: list[list[float]] = []
    anchored = False
    for line in lines:
        if not anchored:
            if anchor in line:
                anchored = True
            continue

        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].lower() == "end":
            break

        try:
            float(tokens[0])
        except ValueError:
            continue

        if len(tokens) < len(columns):
            continue
        rows.append([float(tok) for tok in tokens[: len(columns)]])

    return pd.DataFrame(rows, columns=columns)


def read_tlevel_output(path: str | Path) -> pd.DataFrame:
    """读取 T_LEVEL.OUT。"""
    return _read_numeric_table(path, TLEVEL_COLUMNS, "Time          rTop")


def read_solute_output(path: str | Path) -> pd.DataFrame:
    """读取 solute1.out。"""
    return _read_numeric_table(path, SOLUTE_COLUMNS, "Time         cvTop")


def read_balance_output(path: str | Path) -> pd.DataFrame:
    """读取 BALANCE.OUT 中的水量/溶质相对平衡误差。"""
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    times: list[float] = []
    wat_rel: list[float] = []
    cnc_rel: list[float] = []

    time_re = re.compile(r"Time\s+\[T\]\s+([+-]?\d+(?:\.\d+)?(?:E[+-]?\d+)?)", re.I)
    value_re = re.compile(r"([+-]?\d+(?:\.\d+)?(?:E[+-]?\d+)?)$", re.I)

    current_time: float | None = None
    current_wat: float | None = None
    current_cnc: float | None = None

    for line in lines:
        stripped = line.strip()

        match_time = time_re.search(stripped)
        if match_time:
            if current_time is not None and current_wat is not None and current_cnc is not None:
                times.append(current_time)
                wat_rel.append(current_wat)
                cnc_rel.append(current_cnc)
            current_time = float(match_time.group(1))
            current_wat = None
            current_cnc = None
            continue

        if stripped.startswith("WatBalR"):
            match = value_re.search(stripped)
            if match:
                current_wat = float(match.group(1))
            continue

        if stripped.startswith("CncBalR"):
            match = value_re.search(stripped)
            if match:
                current_cnc = float(match.group(1))

    if current_time is not None and current_wat is not None and current_cnc is not None:
        times.append(current_time)
        wat_rel.append(current_wat)
        cnc_rel.append(current_cnc)

    return pd.DataFrame(
        {
            "Time": np.asarray(times, dtype=np.float64),
            "WatBalR": np.asarray(wat_rel, dtype=np.float64),
            "CncBalR": np.asarray(cnc_rel, dtype=np.float64),
        }
    )


def _align_series(target_t: np.ndarray, raw_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    target_t = np.asarray(target_t, dtype=np.float64)
    raw_t = np.asarray(raw_t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if raw_t.size == 0:
        return np.zeros_like(target_t, dtype=np.float32)

    if raw_t[0] > 0.0:
        raw_t = np.concatenate([[0.0], raw_t])
        values = np.concatenate([[0.0], values])

    aligned = np.interp(target_t, raw_t, values)
    return aligned.astype(np.float32)


def extract_boundary_flux_series(workspace_dir: str | Path, target_t: np.ndarray) -> dict[str, np.ndarray]:
    """从单个 HYDRUS 工作目录提取与训练/评估一致的累计边界通量。"""
    workspace_dir = Path(workspace_dir)
    tlevel = read_tlevel_output(workspace_dir / "T_LEVEL.OUT")
    solute = read_solute_output(workspace_dir / "solute1.out")
    balance = read_balance_output(workspace_dir / "BALANCE.OUT")

    t_w = tlevel["Time"].to_numpy(dtype=np.float64)
    t_c = solute["Time"].to_numpy(dtype=np.float64)
    t_b = balance["Time"].to_numpy(dtype=np.float64)

    # 约定: 所有 *_cum_* 都表示“流入系统为正，流出系统为负”。
    water_top_cum = _align_series(
        target_t,
        t_w,
        tlevel["sum(Infil)"].to_numpy(dtype=np.float64)
        - tlevel["sum(Evap)"].to_numpy(dtype=np.float64),
    )
    water_bottom_cum = _align_series(target_t, t_w, tlevel["sum(vBot)"].to_numpy(dtype=np.float64))
    water_root_cum = _align_series(target_t, t_w, -tlevel["Cum(WTrans)"].to_numpy(dtype=np.float64))
    water_runoff_cum = _align_series(target_t, t_w, -tlevel["sum(RunOff)"].to_numpy(dtype=np.float64))

    solute_top_cum = _align_series(target_t, t_c, solute["Sum(cvTop)"].to_numpy(dtype=np.float64))
    solute_bottom_cum = _align_series(target_t, t_c, solute["Sum(cvBot)"].to_numpy(dtype=np.float64))
    solute_root_cum = _align_series(target_t, t_c, solute["Sum(cvRoot)"].to_numpy(dtype=np.float64))
    solute_runoff_cum = _align_series(target_t, t_c, -solute["Sum(cRunOff)"].to_numpy(dtype=np.float64))

    water_top_flux = _align_series(target_t, t_w, -tlevel["vTop"].to_numpy(dtype=np.float64))
    water_bottom_flux = _align_series(target_t, t_w, tlevel["vBot"].to_numpy(dtype=np.float64))
    solute_top_flux = _align_series(target_t, t_c, solute["cvTop"].to_numpy(dtype=np.float64))
    solute_bottom_flux = _align_series(target_t, t_c, solute["cvBot"].to_numpy(dtype=np.float64))

    return {
        "water_flux_top": water_top_flux,
        "water_flux_bottom": water_bottom_flux,
        "water_cum_top": water_top_cum,
        "water_cum_bottom": water_bottom_cum,
        "water_cum_root": water_root_cum,
        "water_cum_runoff": water_runoff_cum,
        "water_cum_net": water_top_cum + water_bottom_cum + water_root_cum,
        "solute_flux_top": solute_top_flux,
        "solute_flux_bottom": solute_bottom_flux,
        "solute_cum_top": solute_top_cum,
        "solute_cum_bottom": solute_bottom_cum,
        "solute_cum_root": solute_root_cum,
        "solute_cum_runoff": solute_runoff_cum,
        "solute_cum_net": solute_top_cum + solute_bottom_cum + solute_root_cum + solute_runoff_cum,
        "water_balance_rel": _align_series(target_t, t_b, balance["WatBalR"].to_numpy(dtype=np.float64)),
        "solute_balance_rel": _align_series(target_t, t_b, balance["CncBalR"].to_numpy(dtype=np.float64)),
    }
