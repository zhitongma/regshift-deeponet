#!/usr/bin/env python3
"""E3 preflight (完整版, 需 torch — 在集群跑, 本机无 torch).

在训练前确认:
  1. fno_padding.patch 已应用: FNO2d 接受 padding_z/padding_t;
  2. build_fno_model 从 config 顶层 fno: 段正确读到 padding 键
     (model.padding_z/padding_t 属性 == yaml 期望值);
  3. 前向形状不因 padding 改变: forward -> (N, n_z*n_t, 2);
  4. padding=0 时与旧行为一致 (属性存在且为 0, 不影响 state_dict 键集).

用法 (集群, 先激活含 torch 的环境):
    python3 preflight_fno_torch.py <config.yaml> [<config.yaml> ...]
    python3 preflight_fno_torch.py            # 默认校验包内全部 search+final 配置
退出码 0 = 通过。
"""

from pathlib import Path
import glob
import os
import sys

REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import torch  # noqa: E402  (集群环境)
import yaml  # noqa: E402

from src.models.fno import FNO2d, build_fno_model  # noqa: E402


def check(cfg_path: str) -> None:
    base = os.path.basename(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    fno_cfg = cfg.get("fno", {})
    exp_pz = int(fno_cfg.get("padding_z", 0))
    exp_pt = int(fno_cfg.get("padding_t", 0))

    # 1) patch 已应用?
    import inspect
    sig = inspect.signature(FNO2d.__init__)
    if "padding_z" not in sig.parameters:
        raise AssertionError(
            f"{base}: src/models/fno.py 未应用 fno_padding.patch "
            "(FNO2d 无 padding_z 参数) — 先在集群仓库副本上 `patch -p1 < fno_padding.patch`")

    # 2) 构建方式与 04c_train_fno.py:153 完全一致
    model_cfg = {**cfg["model"], "fno": cfg.get("fno", {}), "physics": cfg.get("physics", {})}
    model = build_fno_model(model_cfg)
    assert model.padding_z == exp_pz, \
        f"{base}: model.padding_z={model.padding_z} != yaml fno.padding_z={exp_pz}"
    assert model.padding_t == exp_pt, \
        f"{base}: model.padding_t={model.padding_t} != yaml fno.padding_t={exp_pt}"

    # 3) 前向形状
    n_params = int(cfg["model"]["branch_input_dim"])
    n_z = int(cfg["physics"]["n_spatial_nodes"])
    n_t = int(cfg["physics"]["n_time_steps"])
    with torch.no_grad():
        out = model(torch.randn(2, n_params), torch.zeros(n_z * n_t, 2))
    assert out.shape == (2, n_z * n_t, 2), f"{base}: forward shape {tuple(out.shape)}"

    # 4) padding 不引入新参数 (checkpoint 前后兼容)
    ref = FNO2d(n_params=n_params, n_z=n_z, n_t=n_t,
                width=int(fno_cfg["width"]), modes_z=int(fno_cfg["modes_z"]),
                modes_t=int(fno_cfg["modes_t"]), n_layers=int(fno_cfg.get("n_layers", 4)))
    assert set(ref.state_dict().keys()) == set(model.state_dict().keys()), \
        f"{base}: padding 改变了 state_dict 键集 (不应发生)"

    print(f"  OK  {base}  (width={fno_cfg['width']}, modes=({fno_cfg['modes_z']},"
          f"{fno_cfg['modes_t']}), padding=({exp_pz},{exp_pt}), "
          f"params={model.num_params / 1e6:.2f}M)")


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob(os.path.join(PKG_DIR, "..", "..", "outputs", "fno_search", "configs", "*.yaml"))
                       + glob.glob(os.path.join(PKG_DIR, "..", "..", "configs", "revision", "E3_fno_final", "*.yaml")))
    if not files:
        print("No configs found — run make_search_configs.py first.")
        sys.exit(2)
    for p in files:
        check(p)
    print(f"PREFLIGHT OK: {len(files)} config(s).")


if __name__ == "__main__":
    main()
