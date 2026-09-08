#!/usr/bin/env python3
"""
Step 4e: train a fixed-grid FNN baseline.

This model predicts the full native (z, t) field directly from the parameter
vector and relies on post-hoc interpolation for off-grid time queries.
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

from src.models.grid_fnn import build_grid_fnn_model
from src.models.losses import data_loss
from src.data_generation.dataset import DeepONetDataset
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
    datasets = {}
    for split in ["train", "val", "test"]:
        data = np.load(proc_dir / f"{split}.npz")
        datasets[split] = {key: data[key] for key in data.files}
    return datasets


EVAL_BATCH = 32


def train_epoch(model, loader, trunk, optimizer, device, w_h, w_c):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for branch_b, h_b, c_b in loader:
        branch_b = branch_b.to(device, non_blocking=True)
        h_b = h_b.to(device, non_blocking=True)
        c_b = c_b.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(branch_b, trunk)
        loss = data_loss(pred, h_b, c_b, w_h=w_h, w_c=w_c)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataset, trunk, device, w_h, w_c):
    model.eval()
    branch_all = dataset.branch.to(device)
    h_all = dataset.h_targets.to(device)
    c_all = dataset.c_targets.to(device)
    n_samples = branch_all.shape[0]

    all_h_pred, all_c_pred = [], []
    total_loss = 0.0
    for start in range(0, n_samples, EVAL_BATCH):
        end = min(start + EVAL_BATCH, n_samples)
        pred = model(branch_all[start:end], trunk)
        total_loss += data_loss(
            pred,
            h_all[start:end],
            c_all[start:end],
            w_h=w_h,
            w_c=w_c,
        ).item() * (end - start)
        all_h_pred.append(pred[..., 0].cpu().numpy())
        all_c_pred.append(pred[..., 1].cpu().numpy())

    total_loss /= max(n_samples, 1)
    h_pred = np.concatenate(all_h_pred, axis=0)
    c_pred = np.concatenate(all_c_pred, axis=0)
    h_ref = h_all.cpu().numpy()
    c_ref = c_all.cpu().numpy()

    h_l2 = np.mean([
        np.linalg.norm(h_pred[i] - h_ref[i]) / (np.linalg.norm(h_ref[i]) + 1e-12)
        for i in range(n_samples)
    ])
    c_l2 = np.mean([
        np.linalg.norm(c_pred[i] - c_ref[i]) / (np.linalg.norm(c_ref[i]) + 1e-12)
        for i in range(n_samples)
    ])
    return total_loss, h_l2, c_l2


def main():
    parser = argparse.ArgumentParser(description="Train fixed-grid FNN baseline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg, cfg_path = load_config(PROJECT_ROOT, args.config)
    paths = resolve_artifact_paths(PROJECT_ROOT, cfg, args.run_dir)
    ensure_dirs(paths, "checkpoints", "experiments", "logs", "metadata")
    logger = configure_logger(__name__, paths.logs / "train_grid_fnn.log")
    save_config_snapshot(cfg, cfg_path, paths, "train_grid_fnn")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tcfg = cfg["training"]
    total_epochs = args.epochs or tcfg["epochs"]
    lr = tcfg["learning_rate"]
    batch_size = tcfg["batch_size"]
    val_interval = tcfg["val_interval"]
    w_h = cfg["data_loss"]["w_h"]
    w_c = cfg["data_loss"]["w_c"]

    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    datasets = load_data(paths.processed_data)
    train_ds = DeepONetDataset(datasets["train"])
    val_ds = DeepONetDataset(datasets["val"])
    test_ds = DeepONetDataset(datasets["test"])
    trunk = train_ds.trunk_inputs.float().to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model_cfg = {**cfg["model"], "grid_fnn": cfg.get("grid_fnn", {}), "physics": cfg.get("physics", {})}
    model = build_grid_fnn_model(model_cfg).to(device)
    logger.info("Grid FNN baseline: %d parameters", model.num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pat_checks = patience_epochs_to_checks(tcfg["lr_patience"], val_interval)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=tcfg["lr_factor"],
        patience=pat_checks,
        min_lr=tcfg["lr_min"],
    )

    best_val = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_h_l2": [], "val_c_l2": [], "lr": []}
    t0 = time.time()

    for epoch in range(1, total_epochs + 1):
        train_loss = train_epoch(model, train_loader, trunk, optimizer, device, w_h, w_c)
        history["train_loss"].append(train_loss)

        if should_validate(epoch, val_interval):
            val_loss, val_h_l2, val_c_l2 = evaluate(model, val_ds, trunk, device, w_h, w_c)
            history["val_loss"].append(val_loss)
            history["val_h_l2"].append(val_h_l2)
            history["val_c_l2"].append(val_c_l2)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), paths.checkpoints / "grid_fnn_best.pt")

            if epoch % (val_interval * 10) == 0 or epoch == total_epochs:
                logger.info(
                    "Epoch %d/%d  train=%.6f  val=%.6f  h_l2=%.5f  c_l2=%.5f  lr=%.1e  best@%d",
                    epoch,
                    total_epochs,
                    train_loss,
                    val_loss,
                    val_h_l2,
                    val_c_l2,
                    optimizer.param_groups[0]["lr"],
                    best_epoch,
                )

            no_improve = epochs_without_improvement(epoch, best_epoch)
            if no_improve >= tcfg["early_stopping_patience"]:
                logger.info(
                    "Early stopping at epoch %d (no improve for %d epochs)",
                    epoch,
                    tcfg["early_stopping_patience"],
                )
                break

    elapsed = time.time() - t0
    torch.save(model.state_dict(), paths.checkpoints / "grid_fnn_final.pt")

    test_loss, test_h_l2, test_c_l2 = evaluate(model, test_ds, trunk, device, w_h, w_c)
    logger.info("Test: loss=%.6f  h_l2=%.5f  c_l2=%.5f", test_loss, test_h_l2, test_c_l2)

    def _to_py(value):
        if hasattr(value, "item"):
            return value.item()
        return value

    history_out = {
        **{
            key: [_to_py(x) for x in values] if isinstance(values, list) else _to_py(values)
            for key, values in history.items()
        },
        "training_time_min": elapsed / 60,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        "test_h_l2": float(test_h_l2),
        "test_c_l2": float(test_c_l2),
        "model_params": model.num_params,
    }
    with open(paths.experiments / "grid_fnn_history.json", "w") as f:
        json.dump(history_out, f, indent=2)

    save_manifest(
        paths,
        "train_grid_fnn",
        {
            "checkpoints": {
                "best": str(paths.checkpoints / "grid_fnn_best.pt"),
                "final": str(paths.checkpoints / "grid_fnn_final.pt"),
            },
            "total_epochs": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "training_time_seconds": elapsed,
            "test_metrics": {"loss": test_loss, "h_rel_l2": test_h_l2, "c_rel_l2": test_c_l2},
            "val_interval": val_interval,
            "lr_patience_epochs": tcfg["lr_patience"],
            "scheduler_patience_checks": pat_checks,
            "early_stopping_patience_epochs": tcfg["early_stopping_patience"],
        },
    )
    logger.info("Done. Best epoch=%d, elapsed=%.1f min", best_epoch, elapsed / 60.0)


if __name__ == "__main__":
    main()
