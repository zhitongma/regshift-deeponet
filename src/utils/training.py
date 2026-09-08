"""
训练流程公共工具。
"""

from __future__ import annotations

import math


def should_validate(epoch: int, val_interval: int) -> bool:
    return epoch == 1 or epoch % val_interval == 0


def patience_epochs_to_checks(patience_epochs: int, val_interval: int) -> int:
    """把以 epoch 为单位的 patience 折算成验证次数。"""
    return max(1, math.ceil(patience_epochs / max(val_interval, 1)))


def epochs_without_improvement(current_epoch: int, best_epoch: int | None) -> int:
    if best_epoch is None:
        return 0
    return max(0, current_epoch - best_epoch)
