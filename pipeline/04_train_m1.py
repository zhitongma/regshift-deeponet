#!/usr/bin/env python3
"""
Step 4: 训练 M1 (标准 DeepONet, 纯数据驱动)

用法:
  python scripts/04_train_m1.py
  python scripts/04_train_m1.py --epochs 10000 --device cpu    # 快速调试
"""

import os
import argparse
import json
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deeponet import build_model
from src.models.shift_deeponet import build_shift_deeponet
from src.models.crossattn_deeponet import build_crossattn_deeponet
from src.models.losses import data_loss


def build_model_dispatch(model_cfg: dict):
    """Dispatch model creation based on 'arch' field in config."""
    arch = model_cfg.get("arch", "deeponet")
    if arch == "shift_deeponet":
        return build_shift_deeponet(model_cfg)
    if arch == "crossattn_deeponet":
        return build_crossattn_deeponet(model_cfg)
    return build_model(model_cfg)
from src.data_generation.dataset import DeepONetDataset, SubsampledGridDataset
from src.utils.experiment import (
    configure_logger,
    ensure_dirs,
    load_config,
    resolve_artifact_paths,
    save_config_snapshot,
    save_manifest,
)
from src.utils.training import (
    epochs_without_improvement,
    patience_epochs_to_checks,
    should_validate,
)


def load_data(proc_dir: Path):
    """加载预处理后的 .npz 数据 (返回 numpy dict, 延迟送 GPU)。"""
    datasets = {}
    for split in ["train", "val", "test"]:
        d = np.load(proc_dir / f"{split}.npz")
        datasets[split] = {k: d[k] for k in d.files}
    return datasets


def train_epoch(model, loader, trunk, optimizer, device, w_h, w_c,
                n_coords_train=None):
    """Mini-batch 训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    M = trunk.shape[0]

    for branch_b, h_b, c_b in loader:
        branch_b = branch_b.to(device, non_blocking=True)
        h_b = h_b.to(device, non_blocking=True)
        c_b = c_b.to(device, non_blocking=True)

        optimizer.zero_grad()

        if n_coords_train and n_coords_train < M:
            idx = torch.randperm(M, device=device)[:n_coords_train]
            pred = model(branch_b, trunk[idx])
            loss = data_loss(pred, h_b[:, idx], c_b[:, idx], w_h=w_h, w_c=w_c)
        else:
            pred = model(branch_b, trunk)
            loss = data_loss(pred, h_b, c_b, w_h=w_h, w_c=w_c)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, trunk, device, w_h, w_c):
    """全量验证/测试评估。"""
    model.eval()
    branch = torch.from_numpy(data["branch_inputs"]).to(device)
    h_true = torch.from_numpy(data["h_targets"]).to(device)
    c_true = torch.from_numpy(data["c_targets"]).to(device)

    pred = model(branch, trunk)
    loss = data_loss(pred, h_true, c_true, w_h=w_h, w_c=w_c).item()

    h_err = torch.norm(pred[..., 0] - h_true) / torch.norm(h_true)
    c_err = torch.norm(pred[..., 1] - c_true) / torch.norm(c_true)

    return {"loss": loss, "h_rel_l2": h_err.item(), "c_rel_l2": c_err.item()}


def main():
    parser = argparse.ArgumentParser(description="训练 M1 DeepONet")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg, cfg_path = load_config(PROJECT_ROOT, args.config)
    paths = resolve_artifact_paths(PROJECT_ROOT, cfg, args.run_dir)
    ensure_dirs(paths, "processed_data", "checkpoints", "logs", "experiments", "metadata")
    logger = configure_logger(__name__, paths.logs / "train_m1.log")

    tcfg = cfg["training"]
    epochs = args.epochs or tcfg["epochs"]
    lr = args.lr or tcfg.get("learning_rate", tcfg.get("lr"))
    lr_min = tcfg["lr_min"]
    patience_epochs = tcfg["early_stopping_patience"]
    seed = args.seed if args.seed is not None else tcfg["seed"]
    batch_size = tcfg.get("batch_size") or None
    val_interval = args.val_interval or tcfg.get("val_interval", 100)
    lr_patience_epochs = tcfg.get("lr_patience", 1000)
    lr_factor = tcfg.get("lr_factor", 0.5)
    n_coords_train = tcfg.get("n_coords_train", None)
    w_h = cfg.get("data_loss", {}).get("w_h", 1.0)
    w_c = cfg.get("data_loss", {}).get("w_c", 1.0)
    scheduler_patience_checks = patience_epochs_to_checks(lr_patience_epochs, val_interval)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    save_config_snapshot(
        cfg,
        cfg_path,
        paths,
        "train_m1",
        extra={
            "epochs": epochs,
            "lr": lr,
            "val_interval": val_interval,
            "lr_patience_epochs": lr_patience_epochs,
            "scheduler_patience_checks": scheduler_patience_checks,
        },
    )

    logger.info("=" * 60)
    logger.info("M1 DeepONet 训练 (纯数据驱动)")
    logger.info("  设备: %s", device)
    logger.info("  run_root: %s", paths.root)
    logger.info("  Epochs: %d, LR: %.1e -> %.1e", epochs, lr, lr_min)
    logger.info("  Batch size: %s", batch_size or "full")
    logger.info(
        "  val_interval=%d, lr_patience=%d epochs -> %d checks, early_stop=%d epochs",
        val_interval,
        lr_patience_epochs,
        scheduler_patience_checks,
        patience_epochs,
    )
    logger.info("=" * 60)

    # 数据
    datasets = load_data(paths.processed_data)

    time_stride = tcfg.get("time_stride", None)
    if time_stride and time_stride > 1:
        n_z = cfg["physics"]["n_spatial_nodes"]
        n_t = cfg["physics"]["n_time_steps"]
        train_ds = SubsampledGridDataset(datasets["train"], n_z=n_z, n_t=n_t,
                                         time_stride=time_stride)
        logger.info("Using SubsampledGridDataset: time_stride=%d, trunk %d -> %d points",
                    time_stride, n_z * n_t, train_ds.trunk_inputs.shape[0])
    else:
        train_ds = DeepONetDataset(datasets["train"])

    bs = batch_size if batch_size else len(train_ds)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    trunk = train_ds.trunk_inputs.to(device)

    # For validation/test, always use the full grid trunk
    if hasattr(train_ds, 'full_trunk_inputs'):
        full_trunk = train_ds.full_trunk_inputs.to(device)
    else:
        full_trunk = trunk

    logger.info("数据加载完成: train=%d, val=%d, test=%d (batch_size=%d, steps/epoch=%d)",
                len(train_ds),
                len(datasets["val"]["branch_inputs"]),
                len(datasets["test"]["branch_inputs"]),
                bs, len(train_loader))

    # 模型
    model = build_model_dispatch(cfg["model"]).to(device)
    logger.info("模型参数量: %d (%.1f K)", model.num_params, model.num_params / 1000)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor,
                                  patience=scheduler_patience_checks, min_lr=lr_min)

    # 训练
    best_val_loss = float("inf")
    best_epoch = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "h_rel_l2": [],
        "c_rel_l2": [],
        "val_epoch": [],
        "lr": [],
    }
    ckpt_dir = paths.checkpoints

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, trunk, optimizer,
                                 device, w_h, w_c,
                                 n_coords_train=n_coords_train)
        history["train_loss"].append(train_loss)

        if should_validate(epoch, val_interval):
            val_metrics = evaluate(model, datasets["val"], full_trunk, device, w_h, w_c)
            history["val_loss"].append(val_metrics["loss"])
            history["h_rel_l2"].append(val_metrics["h_rel_l2"])
            history["c_rel_l2"].append(val_metrics["c_rel_l2"])
            history["val_epoch"].append(epoch)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / "m1_best.pt")

            no_improve_epochs = epochs_without_improvement(epoch, best_epoch)
            if epoch == 1 or epoch % (val_interval * 10) == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %5d | train=%.4e | val=%.4e | "
                    "h_L2=%.4f | c_L2=%.4f | lr=%.1e | no_improve=%d",
                    epoch, train_loss, val_metrics["loss"],
                    val_metrics["h_rel_l2"], val_metrics["c_rel_l2"],
                    lr_now, no_improve_epochs,
                )

            if no_improve_epochs >= patience_epochs:
                logger.info("早停触发 (patience=%d epochs)", patience_epochs)
                break

    elapsed = time.time() - t_start
    logger.info("训练完成! 耗时: %.1f min", elapsed / 60)

    # 加载最优模型, 在测试集上评估
    model.load_state_dict(torch.load(ckpt_dir / "m1_best.pt", map_location=device, weights_only=True))
    test_metrics = evaluate(model, datasets["test"], full_trunk, device, w_h, w_c)
    logger.info("测试集结果: loss=%.4e | h_L2=%.4f | c_L2=%.4f",
                test_metrics["loss"], test_metrics["h_rel_l2"], test_metrics["c_rel_l2"])

    # 保存训练历史和最终模型
    torch.save(model.state_dict(), ckpt_dir / "m1_final.pt")

    history_path = paths.experiments / "m1_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "test_metrics": test_metrics,
            "training_time_min": elapsed / 60,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
        }, f, indent=2)
    logger.info("历史记录保存: %s", history_path)

    save_manifest(
        paths,
        "train_m1",
        {
            "history_path": str(history_path),
            "checkpoints": {
                "best": str(ckpt_dir / "m1_best.pt"),
                "final": str(ckpt_dir / "m1_final.pt"),
            },
            "total_epochs": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_time_seconds": elapsed,
            "test_metrics": test_metrics,
            "val_interval": val_interval,
            "lr_patience_epochs": lr_patience_epochs,
            "scheduler_patience_checks": scheduler_patience_checks,
            "early_stopping_patience_epochs": patience_epochs,
        },
    )


if __name__ == "__main__":
    main()
