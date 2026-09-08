#!/usr/bin/env python3
"""E3 第1步: 程序化生成 FNO IID 超参搜索配置 (本地可跑, 仅标准库).

网格: width{32,64,128} x (modes_z,modes_t){(16,10),(32,16),(48,24)}
      x lr{1e-3,5e-4} x padding{0,8}  => 36 组

基版: 主仓库 configs/qtop_func/qtop_func_iid_medium.yaml (完整段落内嵌于下方模板,
      仅 fno 段与 training.learning_rate 随网格变化; data 段与基版完全一致;
      training.seed 固定 42, epochs 固定 2000)。

输出:
  configs/search/fno_w{W}_m{MZ}x{MT}_lr{LR}_p{P}.yaml  (36 份)
  configs/search_manifest.csv  (含剪枝 stage 标记)

剪枝分期 (见 README):
  stage1 = padding=8 且 width∈{64,128} 的 12 组 (优先跑)
  stage2 = 其余 24 组 (视 stage1 结论补跑)
"""

import csv
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIR = os.path.join(PKG_DIR, "..", "..", "outputs", "fno_search", "configs")
MANIFEST = os.path.join(PKG_DIR, "..", "..", "outputs", "fno_search", "search_manifest.csv")

WIDTHS = [32, 64, 128]
MODES = [(16, 10), (32, 16), (48, 24)]
LRS = [("1e-3", "1.0e-3"), ("5e-4", "5.0e-4")]
PADDINGS = [0, 8]

# 与主仓库 configs/qtop_func/qtop_func_iid_medium.yaml 逐段一致的完整模板;
# 可变项: {width} {modes_z} {modes_t} {padding_z} {padding_t} {lr} {name}
TEMPLATE = """\
# ==============================================================================
# E3 FNO hyperparameter search — {name}
# Base: configs/qtop_func/qtop_func_iid_medium.yaml (IID, medium)
# Grid point: width={width}, modes=({modes_z},{modes_t}), lr={lr}, padding=({padding_z},{padding_t})
# NOTE: fno.padding_z / fno.padding_t require fno_padding.patch applied to
#       src/models/fno.py (padding=0 runs work with or without the patch).
# ==============================================================================

physics:
  domain_length: 100.0
  simulation_time: 48.0
  n_spatial_nodes: 101
  n_time_steps: 49
  initial_head: -100.0
  initial_conc: 0.0

parameters:
  theta_r:
    min: 0.034
    max: 0.098
    distribution: uniform
  theta_s:
    min: 0.36
    max: 0.51
    distribution: uniform
  alpha:
    min: 0.005
    max: 0.145
    distribution: log_uniform
  n_vg:
    min: 1.09
    max: 2.68
    distribution: uniform
  K_s:
    min: 0.25
    max: 29.7
    distribution: log_uniform
  D_L:
    min: 1.0
    max: 20.0
    distribution: uniform
  q_top:
    min: 0.1
    max: 5.0
    distribution: uniform
  c_top:
    min: 0.1
    max: 1.0
    distribution: uniform

q_top_function:
  enabled: true
  n_steps: 48
  n_control_points: 6
  min_flux: 0.1
  max_flux: 5.0

data:
  n_total: 1024
  n_train: 768
  n_val: 128
  n_test: 128
  seed: 42
  split_mode: iid
  ood_parameter: q_top_peak
  ood_quantile: 0.9
  ood_tail: high
  qc_mass_balance_threshold: 0.01

model:
  branch_input_dim: 57
  branch_hidden: [512, 512, 512, 512]
  trunk_input_dim: 2
  trunk_hidden: [256, 256, 256, 256]
  trunk_output_dim: 256
  num_outputs: 2
  activation: gelu
  multi_output_strategy: split_branch
  use_residual: true
  use_layer_norm: true
  trunk_encoding: fourier
  fourier_features: 32
  fourier_scale: 4.0
  use_output_heads: true
  output_head_hidden: [256]

fnn:
  hidden_dims: [512, 512, 512, 512]

fno:
  width: {width}
  modes_z: {modes_z}
  modes_t: {modes_t}
  n_layers: 4
  padding_z: {padding_z}
  padding_t: {padding_t}

grid_fnn:
  hidden_dims: [1024, 1024, 1024]
  use_layer_norm: true

training:
  epochs: 2000
  learning_rate: {lr}
  m2_learning_rate: 1.0e-4
  lr_min: 1.0e-6
  lr_scheduler: reduce_on_plateau
  val_interval: 100
  lr_patience: 400
  lr_factor: 0.5
  batch_size: 32
  early_stopping_patience: 800
  optimizer: adam
  seed: 42
  m2_warm_start: true
  m2_warm_start_checkpoint: m1_best.pt
  m2_scheduler_metric: loss
  m2_early_stopping_metric: loss
  m2_best_alias_metric: score
  m2_phase2_selection_only: false
  m2_phase2_reset_scheduler: false
  m2_phase2_reset_early_stopping: false
  m2_phase2_require_entry_before_early_stop: false

data_loss:
  w_h: 1.0
  w_c: 1.0

mass_loss:
  target_mode: true_flux
  objective_mode: relative_mse
  selection_mass_metric: objective
  w_mass_phase1: 0.00
  w_mass_phase2: 0.05
  switch_epoch_fraction: 0.40
  w_mass_water: 1.0
  w_mass_solute: 0.2
  w_mass_water_phase1: 1.0
  w_mass_water_phase2: 1.0
  w_mass_solute_phase1: 0.2
  w_mass_solute_phase2: 0.2
  selection_data_weight: 1.0
  selection_mass_water_weight: 0.02
  selection_mass_solute_weight: 0.005

paths:
  raw_data: data/raw
  processed_data: data/processed
  parameters: data/parameters
  checkpoints: results/checkpoints
  logs: results/logs
  experiments: results/experiments
  figures: figures
"""

N_Z, N_T = 101, 49  # physics grid (与模板一致)


def check_modes(modes_z: int, modes_t: int, pad_z: int, pad_t: int) -> None:
    """FFT 截断模式数不得超过 (可能填充后的) 网格允许上限."""
    nz, nt = N_Z + pad_z, N_T + pad_t
    assert 2 * modes_z <= nz, f"modes_z={modes_z} too large for n_z={nz}"
    assert modes_t <= nt // 2 + 1, f"modes_t={modes_t} too large for n_t={nt}"


def main() -> None:
    os.makedirs(SEARCH_DIR, exist_ok=True)
    rows = []
    for width in WIDTHS:
        for modes_z, modes_t in MODES:
            for lr_tag, lr_val in LRS:
                for pad in PADDINGS:
                    check_modes(modes_z, modes_t, pad, pad)
                    name = f"fno_w{width}_m{modes_z}x{modes_t}_lr{lr_tag}_p{pad}"
                    text = TEMPLATE.format(
                        name=name, width=width,
                        modes_z=modes_z, modes_t=modes_t,
                        padding_z=pad, padding_t=pad, lr=lr_val,
                    )
                    path = os.path.join(SEARCH_DIR, name + ".yaml")
                    with open(path, "w") as f:
                        f.write(text)
                    stage = "stage1" if (pad == 8 and width in (64, 128)) else "stage2"
                    rows.append({
                        "config_name": name,
                        "width": width,
                        "modes_z": modes_z,
                        "modes_t": modes_t,
                        "lr": lr_val,
                        "padding": pad,
                        "stage": stage,
                        "run_dir_name": f"e3_fno_search_{name}",
                    })

    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    n1 = sum(1 for r in rows if r["stage"] == "stage1")
    print(f"Wrote {len(rows)} configs to {SEARCH_DIR}")
    print(f"Manifest: {MANIFEST}  (stage1={n1}, stage2={len(rows) - n1})")


if __name__ == "__main__":
    main()
