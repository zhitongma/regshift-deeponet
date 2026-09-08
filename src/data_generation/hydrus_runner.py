"""
HYDRUS-1D 模拟运行器 (基于 phydrus)

封装单次/批量 HYDRUS-1D 模拟:
  Richards 方程 (水流) + ADE (溶质运移)
  VG 本构模型, 均质土柱, 支持恒定或时变入渗 + 恒定浓度边界
"""

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generation.hydrus_outputs import extract_boundary_flux_series

logger = logging.getLogger(__name__)


def run_single_simulation(
    sample_id: int,
    params: dict,
    exe_path: str,
    base_ws: str,
    physics_cfg: dict,
    keep_workspace: bool = False,
) -> dict:
    """运行一组参数的 HYDRUS-1D 模拟并返回结构化结果。

    Parameters
    ----------
    sample_id : int
        样本编号
    params : dict
        包含 theta_r, theta_s, alpha, n_vg, K_s, D_L, c_top,
        以及 q_top 或 q_top_series
    exe_path : str
        HYDRUS-1D 可执行文件路径
    base_ws : str
        工作空间根目录
    physics_cfg : dict
        物理场景配置 (domain_length, simulation_time, etc.)
    keep_workspace : bool
        是否保留 HYDRUS 工作目录 (调试用)

    Returns
    -------
    dict
        h: (n_z, n_t), c: (n_z, n_t), theta: (n_z, n_t),
        z: (n_z,), t: (n_t,), params, success, error
    """
    import phydrus as ps

    L = physics_cfg["domain_length"]       # 100 cm
    T = physics_cfg["simulation_time"]     # 48 h
    n_z = physics_cfg["n_spatial_nodes"]   # 101
    n_t = physics_cfg["n_time_steps"]      # 49
    h_init = params.get("h_init", physics_cfg["initial_head"])
    c_init = params.get("c_init", physics_cfg["initial_conc"])
    dz = L / (n_z - 1)                    # 1 cm

    ws_path = str(Path(base_ws) / f"sim_{sample_id:04d}")
    z_coords = np.linspace(0, L, n_z)     # 0..100 cm (depth from surface)
    t_coords = np.linspace(0, T, n_t)     # 0..48 h

    result = dict(
        sample_id=sample_id, params=params,
        h=None, c=None, theta=None,
        z=z_coords, t=t_coords,
        water_flux_top=None,
        water_flux_bottom=None,
        water_cum_top=None,
        water_cum_bottom=None,
        water_cum_root=None,
        water_cum_runoff=None,
        water_cum_net=None,
        solute_flux_top=None,
        solute_flux_bottom=None,
        solute_cum_top=None,
        solute_cum_bottom=None,
        solute_cum_root=None,
        solute_cum_runoff=None,
        solute_cum_net=None,
        water_balance_rel=None,
        solute_balance_rel=None,
        success=False, error=None,
    )

    try:
        ml = _build_model(ps, sample_id, params, exe_path, ws_path,
                          L, T, n_z, n_t, dz, h_init, c_init)
        ml.write_input()
        ml.simulate()

        _extract_results(ml, result, ws_path, n_z, n_t, h_init, c_init, params)

    except Exception as e:
        result["error"] = str(e)
        logger.error("Sample %d failed: %s", sample_id, e)

    if not keep_workspace:
        shutil.rmtree(ws_path, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# 内部函数
# ---------------------------------------------------------------------------

def _build_model(ps, sample_id, params, exe_path, ws_path,
                 L, T, n_z, n_t, dz, h_init, c_init):
    """组装 phydrus 模型对象。"""
    ml = ps.Model(
        exe_name=exe_path,
        ws_name=ws_path,
        name=f"sim_{sample_id:04d}",
        description=f"Richards-ADE coupled, sample {sample_id}",
        mass_units="mg",
        time_unit="hours",
        length_unit="cm",
    )

    # --- 时间 ---
    # Keep HYDRUS on a constant print interval so solute output files are
    # generated consistently while still allowing denser temporal sampling.
    dtprint = T / max(n_t - 1, 1)
    ml.add_time_info(
        tinit=0, tmax=T,
        print_times=True,
        printinit=0, printmax=T, dtprint=dtprint,
        dt=0.01, dtmin=1e-5, dtmax=1.0,
    )

    # --- 水流 ---
    # top_bc=3: 大气 BC (支持入渗限制 / 地表径流)
    # bot_bc=4: 自由排水
    ml.add_waterflow(model=0, top_bc=3, bot_bc=4, ha=1e-6, hb=1e4)

    # --- 溶质运移 ---
    # model=0: 平衡运移; top_bc=-1: Cauchy; bot_bc=0: 零浓度梯度
    ml.add_solute_transport(
        model=0, epsi=0.5, lupw=True,
        tpulse=T, top_bc=-1, bot_bc=0,
    )

    # --- 土壤材料 (单层均质) ---
    m = ml.get_empty_material_df(n=1)
    vg_vals = [
        params["theta_r"], params["theta_s"],
        params["alpha"], params["n_vg"],
        params["K_s"], 0.5,
    ]
    n_cols = len(m.columns)
    if n_cols > 6:
        solute_mat_vals = [1.5, float(params["D_L"]), 1.0, 0.0]
        vg_vals.extend(solute_mat_vals[:n_cols - 6])
    m.loc[1] = vg_vals
    ml.add_material(m)

    # --- 土壤剖面 ---
    profile = ps.create_profile(
        top=0, bot=-L, dx=dz,
        h=h_init, mat=1, conc=c_init, sconc=0,
    )
    ml.add_profile(profile)

    # --- 大气边界条件 (恒定或时变入渗 + 恒定或时变溶质浓度) ---
    n_atm = int(T)
    if "q_top_series" in params:
        q_top_series = np.asarray(params["q_top_series"], dtype=float).reshape(-1)
        if q_top_series.size != n_atm:
            raise ValueError(
                f"q_top_series length mismatch: expected {n_atm}, got {q_top_series.size}"
            )
    else:
        q_top_series = np.full(n_atm, float(params["q_top"]), dtype=float)

    if "c_top_series" in params:
        c_top_series = np.asarray(params["c_top_series"], dtype=float).reshape(-1)
        if c_top_series.size != n_atm:
            raise ValueError(
                f"c_top_series length mismatch: expected {n_atm}, got {c_top_series.size}"
            )
    else:
        c_top_series = np.full(n_atm, float(params["c_top"]), dtype=float)

    atm = pd.DataFrame({
        "tAtm": np.arange(1, n_atm + 1, dtype=float),
        "Prec": q_top_series,
        "rSoil": 0.0, "rRoot": 0.0, "hCritA": 1e5,
        "rB": 0.0, "hB": 0.0, "hT": 0.0,
        "tTop": 0.0, "tBot": 0.0, "Ampl": 0.0,
        "cTop": c_top_series,
        "cBot": 0.0,
    })
    ml.add_atmospheric_bc(atm)

    # --- 溶质属性 (保守性溶质, 无吸附/降解) ---
    c_top_initial = float(c_top_series[0]) if len(c_top_series) > 0 else float(params["c_top"])
    sol = ml.get_empty_solute_df()
    sol.loc[1, "beta"] = 1.0
    ml.add_solute(sol, difw=0.0, difg=0.0, top_conc=c_top_initial)

    return ml


def _extract_results(model, result, ws_path, n_z, n_t, h_init, c_init, params):
    """从 HYDRUS 输出中提取 h(z,t), c(z,t), theta(z,t)。"""

    h_arr = np.full((n_z, n_t), np.nan)
    c_arr = np.full((n_z, n_t), np.nan)
    theta_arr = np.full((n_z, n_t), np.nan)

    # --- 初始条件 (t = 0) ---
    h_arr[:, 0] = h_init
    c_arr[:, 0] = c_init
    theta_arr[:, 0] = _vg_theta(h_init, params)

    # --- 读取 NOD_INF (dict: {time: DataFrame} 或单个 DataFrame) ---
    nod_data = model.read_nod_inf()
    if isinstance(nod_data, pd.DataFrame):
        nod_data = {0.0: nod_data}

    sorted_times = sorted(nod_data.keys())
    for j, t_val in enumerate(sorted_times):
        col = j + 1          # col 0 = 初始条件
        if col >= n_t:
            break
        df = nod_data[t_val]
        n_rows = min(len(df), n_z)

        h_arr[:n_rows, col] = df["Head"].values[:n_rows]

        if "Moisture" in df.columns:
            theta_arr[:n_rows, col] = df["Moisture"].values[:n_rows]

        conc_col = _find_conc_column(df)
        if conc_col is not None:
            c_arr[:n_rows, col] = df[conc_col].values[:n_rows]

    flux_series = extract_boundary_flux_series(ws_path, result["t"])

    result.update(
        h=h_arr, c=c_arr, theta=theta_arr,
        **flux_series,
        success=not np.any(np.isnan(h_arr)),
    )


def _find_conc_column(df: pd.DataFrame) -> str | None:
    """在 NOD_INF DataFrame 中查找浓度列名。"""
    candidates = ["Conc(1..NMat)", "Conc(1)", "Conc", "Conc1", "c1"]
    for name in candidates:
        if name in df.columns:
            return name
    for col in df.columns:
        if "conc" in col.lower():
            return col
    return None


def _vg_theta(h: float, params: dict) -> float:
    """van Genuchten 水分特征曲线: h -> theta。"""
    tr = params["theta_r"]
    ts = params["theta_s"]
    alpha = params["alpha"]
    n = params["n_vg"]
    m = 1.0 - 1.0 / n
    if h >= 0:
        return ts
    Se = 1.0 / (1.0 + abs(alpha * h) ** n) ** m
    return tr + (ts - tr) * Se
