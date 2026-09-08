#!/usr/bin/env python3
"""E0 行为探针 (torch-free): 判断历史 RegShift checkpoint 训练时界限是否生效。

原理 (见 E0/README.md §四):
  加载 checkpoint 中 transform_net 的权重, 用 numpy 前向传播测试集 branch 输入,
  观察 raw 输出分布:
    - raw scale 通道 ≈ (0.7,1.3)、shift 通道 ≈ ±0.2, 与无界 boost_A4 同形态
        -> 训练时按"直接尺度/平移"使用 (H1: 界限未生效);
    - raw 明显超出该区间 (tanh 饱和量级) 而历史评估精度仍好
        -> 只能是 tanh 分支在起作用 (H2: 界限生效过)。

用法:
    python3 probe_transform.py --runs-root <...>/experiments/qtop_func/runs \
        --run g4_regshift_iid_s42 --run boost_A4_iid [--ckpt m1_best.pt m2_best.pt]

⚠️ 本机若为 iCloud 同步盘, 先确认 .pt/.npz 已物化 (brctl download), 否则读取会挂起。
"""
import argparse
import io
import os
import pickle
import re
import zipfile

import numpy as np

DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "HalfStorage": np.float16, "LongStorage": np.int64,
    "IntStorage": np.int32, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _TensorStub:
    def __init__(self, storage_key, dtype):
        self.storage_key, self.dtype = storage_key, dtype


def load_state_dict_numpy(path):
    """torch-free 读取 torch>=1.6 zip 格式 checkpoint 的 state_dict -> numpy."""
    zf = zipfile.ZipFile(path)
    pkl_name = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl_name[: -len("data.pkl")]

    def rebuild_tensor_v2(storage, offset, size, stride, *_):
        raw = np.frombuffer(zf.read(prefix + "data/" + storage.storage_key),
                            dtype=storage.dtype)
        size = tuple(size)
        if not size:
            return raw[offset]
        expect = []
        acc = 1
        for s in size[::-1]:
            expect.append(acc)
            acc *= s
        expect = expect[::-1]
        if list(stride) == expect:
            n = int(np.prod(size))
            return raw[offset: offset + n].reshape(size)
        itemsize = raw.itemsize
        return np.lib.stride_tricks.as_strided(
            raw[offset:], shape=size,
            strides=tuple(s * itemsize for s in stride)).copy()

    class _Stub:
        def __init__(self, *a, **k):
            pass

    class Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            # pid = ('storage', StorageType, key, location, numel)
            return _TensorStub(pid[2], DTYPES[pid[1].__name__])

        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild_tensor_v2
            if name.endswith("Storage"):
                return type(name, (), {})
            if module == "collections" and name == "OrderedDict":
                from collections import OrderedDict
                return OrderedDict
            return _Stub

    return Unpickler(io.BytesIO(zf.read(pkl_name))).load()


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def probe(run_dir, ckpt):
    path = os.path.join(run_dir, "results", "checkpoints", ckpt)
    sd = load_state_dict_numpy(path)
    for wrap in ("model_state_dict", "state_dict", "model"):
        if hasattr(sd, "keys") and wrap in sd and hasattr(sd[wrap], "keys"):
            sd = sd[wrap]
            break
    tkeys = [k for k in sd if "transform_net" in k]
    if not tkeys:
        print(f"  (无 transform_net — 非 shift 家族 checkpoint) keys 示例: {list(sd)[:4]}")
        return
    lin_idx = sorted({int(re.search(r"transform_net\.(\d+)\.", k).group(1))
                      for k in tkeys})
    test_npz = os.path.join(run_dir, "data", "processed", "test.npz")
    X = np.asarray(np.load(test_npz)["branch_inputs"][:256], dtype=np.float64)

    h = X
    for i, li in enumerate(lin_idx):
        W = np.asarray(sd[f"transform_net.{li}.weight"], dtype=np.float64)
        b = np.asarray(sd[f"transform_net.{li}.bias"], dtype=np.float64)
        h = h @ W.T + b
        if i < len(lin_idx) - 1:
            h = gelu(h)
    raw = h
    D = raw.shape[1] // 2
    s_raw, d_raw = raw[:, :D], raw[:, D:]
    fb = np.asarray(sd[f"transform_net.{lin_idx[-1]}.bias"], dtype=np.float64)

    print(f"  {ckpt}: 末层 bias = {np.round(fb, 4).tolist()}")
    for j, nm in enumerate(["s_z", "s_t", "d_z", "d_t"][: raw.shape[1]]):
        c = raw[:, j]
        print(f"    raw[{nm}]: min={c.min():+.3f} max={c.max():+.3f} "
              f"mean={c.mean():+.3f} std={c.std():.3f}")
    print(f"    无界解读: scale∈[{s_raw.min():.3f},{s_raw.max():.3f}] "
          f"shift∈[{d_raw.min():.3f},{d_raw.max():.3f}]")
    bs, bd = 1 + 0.3 * np.tanh(s_raw), 0.2 * np.tanh(d_raw)
    print(f"    有界解读: scale∈[{bs.min():.3f},{bs.max():.3f}] "
          f"shift∈[{bd.min():.3f},{bd.max():.3f}]")
    # 判定说明（2026-07-16 实测修正）：
    # "raw 超出名义界限" 不能区分 H1/H2 —— 已知无界的 boost_A4 的 raw 同样
    # 落在窗口外（无界模型自然学到小尺度系数且工作正常）。
    # 有判别力的是【末层 bias】：当前代码把 bias 初始化为 [1,1,0,0]（无界恒等），
    # 论文描述的有界版应初始化为 0。若训练走的是 tanh 分支且初始为 0，
    # bias 不应恰好停在 ~[1,1,0,0]。bias≈[1,1,0,0] => 强烈支持 H1。
    near_unbounded_init = (abs(fb[:D] - 1.0).max() < 0.15
                           and abs(fb[D:]).max() < 0.15)
    print(f"    bias 判定: {'≈[1,1,0,0] 无界初始值 -> 支持 H1(界限未生效)' if near_unbounded_init else '偏离无界初始值 -> 需进一步核查(结合旧代码 init 判断)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ckpt", nargs="+", default=["m1_best.pt", "m2_best.pt"])
    args = ap.parse_args()
    for run in args.run:
        rd = os.path.join(args.runs_root, run)
        print(f"\n== {run} ==")
        for ck in args.ckpt:
            try:
                probe(rd, ck)
            except FileNotFoundError as e:
                print(f"  缺文件: {e}")


if __name__ == "__main__":
    main()
