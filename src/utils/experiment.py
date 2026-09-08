"""
实验运行与产物管理工具。

统一处理:
  - 配置加载与快照
  - run_dir 下的目录解析
  - 日志初始化
  - manifest / json / yaml 落盘
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ArtifactPaths:
    root: Path
    raw_data: Path
    processed_data: Path
    parameters: Path
    checkpoints: Path
    logs: Path
    experiments: Path
    figures: Path
    metadata: Path

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


def load_config(project_root: Path, config_path: str | None = None) -> tuple[dict, Path]:
    """加载 YAML 配置，支持相对项目根目录的路径。"""
    if config_path is None:
        cfg_path = project_root / "configs/revision/E0_bounded/e0_regshift_highbudget_iid.yaml"
    else:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "shift_deeponet" in cfg:
        raise ValueError(
            "Place coordinate transform settings under model.shift_deeponet; "
            "top-level shift_deeponet keys were ignored by the historical model builder."
        )
    return cfg, cfg_path


def resolve_artifact_paths(
    project_root: Path,
    cfg: dict,
    run_dir: str | None = None,
) -> ArtifactPaths:
    """解析一次运行的所有核心目录。"""
    if run_dir is None:
        root = project_root / "outputs" / "default_run"
    else:
        root = Path(run_dir)
        if not root.is_absolute():
            root = project_root / root

    paths_cfg = cfg["paths"]
    return ArtifactPaths(
        root=root,
        raw_data=root / paths_cfg["raw_data"],
        processed_data=root / paths_cfg["processed_data"],
        parameters=root / paths_cfg["parameters"],
        checkpoints=root / paths_cfg["checkpoints"],
        logs=root / paths_cfg["logs"],
        experiments=root / paths_cfg["experiments"],
        figures=root / paths_cfg["figures"],
        metadata=root / "metadata",
    )


def ensure_dirs(paths: ArtifactPaths, *names: str):
    """创建指定的产物目录；为空时创建全部核心目录。"""
    targets = names or (
        "raw_data",
        "processed_data",
        "parameters",
        "checkpoints",
        "logs",
        "experiments",
        "figures",
        "metadata",
    )
    for name in targets:
        getattr(paths, name).mkdir(parents=True, exist_ok=True)


def configure_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    """创建同时写终端和文件的 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    return value


def save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)


def save_yaml(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_to_serializable(payload), f, sort_keys=False, allow_unicode=True)


def save_config_snapshot(
    cfg: dict,
    cfg_path: Path,
    paths: ArtifactPaths,
    name: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """保存一次运行的配置快照。"""
    ensure_dirs(paths, "metadata")
    snapshot = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_config": str(cfg_path),
        "artifacts_root": str(paths.root),
        "config": cfg,
    }
    if extra:
        snapshot["extra"] = _to_serializable(extra)

    out_path = paths.metadata / f"{name}_config_snapshot.yaml"
    save_yaml(out_path, snapshot)
    return out_path


def save_manifest(
    paths: ArtifactPaths,
    name: str,
    payload: dict[str, Any],
) -> Path:
    """保存脚本级 manifest。"""
    ensure_dirs(paths, "metadata")
    out_path = paths.metadata / f"{name}_manifest.json"
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        **_to_serializable(payload),
    }
    save_json(out_path, payload)
    return out_path
