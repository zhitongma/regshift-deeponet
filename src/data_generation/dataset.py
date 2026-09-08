"""
PyTorch Dataset / DataLoader for DeepONet 训练

DeepONet 数据格式:
  Branch 输入: (N_samples, 10)    -- 参数向量
  Trunk  输入: (N_points,  2)     -- (z, t) 坐标
  Target:      (N_samples, N_points, 2) -- (h, c)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DeepONetDataset(Dataset):
    """DeepONet 训练数据集 (Cartesian product 模式)。

    每个样本是一组参数 -> 全时空场的映射。
    """

    def __init__(self, data_dict: dict):
        self.branch = torch.from_numpy(data_dict["branch_inputs"])  # (N, 10)
        self.h_targets = torch.from_numpy(data_dict["h_targets"])   # (N, n_z*n_t)
        self.c_targets = torch.from_numpy(data_dict["c_targets"])   # (N, n_z*n_t)
        self.trunk = torch.from_numpy(data_dict["trunk_inputs"])    # (n_z*n_t, 2)

    def __len__(self):
        return len(self.branch)

    def __getitem__(self, idx):
        return (
            self.branch[idx],       # (10,)
            self.h_targets[idx],    # (n_z*n_t,)
            self.c_targets[idx],    # (n_z*n_t,)
        )

    @property
    def trunk_inputs(self):
        return self.trunk


class MassConservationDataset(DeepONetDataset):
    """在标准数据集基础上额外返回累计边界通量目标。"""

    def __init__(self, data_dict: dict):
        super().__init__(data_dict)
        self.water_cum_net = torch.from_numpy(data_dict["water_cum_net"])      # (N, n_t)
        self.solute_cum_net = torch.from_numpy(data_dict["solute_cum_net"])    # (N, n_t)

    def __getitem__(self, idx):
        return (
            self.branch[idx],
            self.h_targets[idx],
            self.c_targets[idx],
            self.water_cum_net[idx],
            self.solute_cum_net[idx],
        )


class SparseQueryDataset(Dataset):
    """Dataset that subsamples random query points per __getitem__ call.

    Each time a sample is accessed, a fresh random subset of trunk points
    is selected.  This forces models to generalize across query locations
    rather than memorizing fixed-grid outputs.

    The trunk tensor returned per item is (n_query, 2) and the targets are
    (n_query,) for h and c respectively.
    """

    def __init__(self, data_dict: dict, n_query: int = 1024, seed: int | None = None):
        self.branch = torch.from_numpy(data_dict["branch_inputs"])
        self.h_targets = torch.from_numpy(data_dict["h_targets"])
        self.c_targets = torch.from_numpy(data_dict["c_targets"])
        self.trunk = torch.from_numpy(data_dict["trunk_inputs"])
        self.n_query = min(n_query, self.trunk.shape[0])
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.branch)

    def __getitem__(self, idx):
        indices = torch.from_numpy(
            self.rng.choice(self.trunk.shape[0], size=self.n_query, replace=False)
        )
        return (
            self.branch[idx],
            self.h_targets[idx, indices],
            self.c_targets[idx, indices],
            self.trunk[indices],
        )

    @property
    def trunk_inputs(self):
        return self.trunk


class SubsampledGridDataset(Dataset):
    """Dataset that uses a structured subsample of the (z,t) grid for training.

    Takes every time_stride-th time step (and optionally every z_stride-th
    spatial node) from the full grid.  Evaluation can still use the full grid.
    """

    def __init__(self, data_dict: dict, n_z: int = 101, n_t: int = 49,
                 time_stride: int = 2, z_stride: int = 1):
        full_trunk = torch.from_numpy(data_dict["trunk_inputs"])
        full_h = torch.from_numpy(data_dict["h_targets"])
        full_c = torch.from_numpy(data_dict["c_targets"])
        self.branch = torch.from_numpy(data_dict["branch_inputs"])

        z_idx = list(range(0, n_z, z_stride))
        t_idx = list(range(0, n_t, time_stride))
        keep = []
        for ti in t_idx:
            for zi in z_idx:
                keep.append(ti * n_z + zi)
        keep = sorted(keep)
        self._keep = torch.tensor(keep, dtype=torch.long)

        self.trunk = full_trunk[self._keep]
        self.h_targets = full_h[:, self._keep]
        self.c_targets = full_c[:, self._keep]
        self._full_trunk = full_trunk

    def __len__(self):
        return len(self.branch)

    def __getitem__(self, idx):
        return (
            self.branch[idx],
            self.h_targets[idx],
            self.c_targets[idx],
        )

    @property
    def trunk_inputs(self):
        return self.trunk

    @property
    def full_trunk_inputs(self):
        return self._full_trunk


def create_dataloaders(
    train_dict: dict,
    val_dict: dict,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """创建训练和验证 DataLoader。

    Returns
    -------
    train_loader, val_loader, trunk_inputs
    """
    train_ds = DeepONetDataset(train_dict)
    val_ds = DeepONetDataset(val_dict)

    bs_train = batch_size if batch_size else len(train_ds)
    bs_val = len(val_ds)

    train_loader = DataLoader(
        train_ds, batch_size=bs_train, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs_val, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_ds.trunk_inputs
