#!/usr/bin/env python3
"""E3 preflight (降级版, 无 torch, 本地可跑): 纯 YAML 键层级校验.

校验对象: configs/search/*.yaml 与 configs/final/*.yaml
检查项:
  1. fno 段存在且含 width / modes_z / modes_t / n_layers / padding_z / padding_t;
  2. padding 键必须位于顶层 fno: 段内 (不得写成顶层 padding_z 或 model.fno.*
     —— 04c_train_fno.py 只合并顶层 fno: 段, 写错位置会被静默忽略,
     同类历史 bug 见 REPO_FACTS.md §1 的 shift_deeponet 教训);
  3. 模式数不超过 (填充后) 网格允许上限: 2*modes_z <= n_z+padding_z,
     modes_t <= (n_t+padding_t)//2 + 1;
  4. training.seed == 42, training.epochs == 2000;
  5. search 配置文件名与内容一致 (fno_w{W}_m{MZ}x{MT}_lr{LR}_p{P}.yaml);
  6. final 配置 data 段与场景匹配 (split_mode / ood_parameter / n_test / n_total);
  7. 防御性: 若 model.arch == shift_deeponet, 必须存在
     model.shift_deeponet.transform_bound_scale / transform_bound_shift
     (本包不训练 shift 家族, 此检查仅防误用模板)。

用法:  python3 preflight_fno_yaml.py            # 校验包内全部配置
       python3 preflight_fno_yaml.py <yaml>...  # 校验指定文件
退出码 0 = 全部通过。仅用标准库; 若装有 PyYAML 则优先使用。
"""

import glob
import os
import re
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import yaml as _yaml

    def load_yaml(path):
        with open(path) as f:
            return _yaml.safe_load(f)
except ImportError:
    def load_yaml(path):
        """极简 YAML 子集解析: 缩进嵌套的 `key: value` 标量与 flow 列表.

        足以覆盖本包配置 (无块列表 / 无多行标量 / 无锚点)。
        """
        root = {}
        stack = [(-1, root)]  # (indent, dict)
        with open(path) as f:
            for raw in f:
                line = raw.rstrip("\n")
                stripped = line.split("#", 1)[0].rstrip()
                if not stripped.strip():
                    continue
                indent = len(stripped) - len(stripped.lstrip())
                key_val = stripped.strip()
                if ":" not in key_val:
                    continue
                key, _, val = key_val.partition(":")
                key, val = key.strip(), val.strip()
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                parent = stack[-1][1]
                if val == "":
                    child = {}
                    parent[key] = child
                    stack.append((indent, child))
                else:
                    parent[key] = _coerce(val)
        return root

    def _coerce(v):
        if v.startswith("["):
            return v
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v


NAME_RE = re.compile(
    r"^fno_w(?P<w>\d+)_m(?P<mz>\d+)x(?P<mt>\d+)_lr(?P<lr>[0-9e.\-]+)_p(?P<p>\d+)\.yaml$")

FINAL_EXPECT = {
    "fno_best_iid.yaml": {"split_mode": "iid", "ood_parameter": "q_top_peak",
                          "n_test": 128, "n_total": 1024},
    "fno_best_ood_peak.yaml": {"split_mode": "ood", "ood_parameter": "q_top_peak",
                               "n_test": 96, "n_total": 1024},
    "fno_best_ood_peak_time.yaml": {"split_mode": "ood",
                                    "ood_parameter": "q_top_peak_time",
                                    "n_test": 96, "n_total": 1152},
}


def check_file(path):
    errors = []
    cfg = load_yaml(path)
    base = os.path.basename(path)

    def err(msg):
        errors.append(f"{base}: {msg}")

    fno = cfg.get("fno")
    if not isinstance(fno, dict):
        err("缺少顶层 fno: 段")
        return errors
    for k in ("width", "modes_z", "modes_t", "n_layers", "padding_z", "padding_t"):
        if k not in fno:
            err(f"fno 段缺键 {k}")
    if "padding_z" in cfg or "padding_t" in cfg:
        err("padding_z/padding_t 出现在顶层 (必须在 fno: 段内)")
    model = cfg.get("model", {})
    if isinstance(model, dict) and isinstance(model.get("fno"), dict):
        err("model.fno 段不会被 04c 读取 (fno 配置必须在顶层 fno: 段)")

    phys = cfg.get("physics", {})
    n_z = int(phys.get("n_spatial_nodes", 0))
    n_t = int(phys.get("n_time_steps", 0))
    if errors:
        return errors
    mz, mt = int(fno["modes_z"]), int(fno["modes_t"])
    pz, pt = int(fno["padding_z"]), int(fno["padding_t"])
    if 2 * mz > n_z + pz:
        err(f"modes_z={mz} 超限: 需 2*modes_z <= n_z+padding_z = {n_z + pz}")
    if mt > (n_t + pt) // 2 + 1:
        err(f"modes_t={mt} 超限: 需 <= (n_t+padding_t)//2+1 = {(n_t + pt) // 2 + 1}")

    tr = cfg.get("training", {})
    if int(tr.get("seed", -1)) != 42:
        err(f"training.seed = {tr.get('seed')} (要求 42, 多种子经 --seed 传)")
    if int(tr.get("epochs", -1)) != 2000:
        err(f"training.epochs = {tr.get('epochs')} (要求 2000)")

    m = NAME_RE.match(base)
    if m:
        if int(m.group("w")) != int(fno["width"]):
            err(f"文件名 width={m.group('w')} 与 fno.width={fno['width']} 不一致")
        if int(m.group("mz")) != mz or int(m.group("mt")) != mt:
            err(f"文件名 modes 与 fno.modes ({mz},{mt}) 不一致")
        if int(m.group("p")) != pz or int(m.group("p")) != pt:
            err(f"文件名 p{m.group('p')} 与 fno.padding ({pz},{pt}) 不一致")
        lr = float(tr.get("learning_rate", 0))
        if abs(lr - float(m.group("lr"))) > 1e-12:
            err(f"文件名 lr={m.group('lr')} 与 training.learning_rate={lr} 不一致")

    if base in FINAL_EXPECT:
        data = cfg.get("data", {})
        for k, v in FINAL_EXPECT[base].items():
            got = data.get(k)
            if (int(got) if isinstance(v, int) else str(got)) != v:
                err(f"data.{k} = {got} (场景要求 {v})")

    arch = model.get("arch") if isinstance(model, dict) else None
    if arch == "shift_deeponet":
        sd = model.get("shift_deeponet")
        if not (isinstance(sd, dict) and "transform_bound_scale" in sd
                and "transform_bound_shift" in sd):
            err("model.arch=shift_deeponet 但 model.shift_deeponet.transform_bound_* 缺失"
                " (历史 bug, 见 REPO_FACTS.md §1)")
    return errors


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob(os.path.join(PKG_DIR, "..", "..", "outputs", "fno_search", "configs", "*.yaml"))
                       + glob.glob(os.path.join(PKG_DIR, "..", "..", "configs", "revision", "E3_fno_final", "*.yaml")))
    if not files:
        print("No yaml files found — run make_search_configs.py first.")
        sys.exit(2)
    all_errors = []
    for p in files:
        all_errors += check_file(p)
    if all_errors:
        print(f"PREFLIGHT FAILED ({len(all_errors)} errors):")
        for e in all_errors:
            print("  -", e)
        sys.exit(1)
    print(f"PREFLIGHT OK: {len(files)} config(s) passed structural checks.")


if __name__ == "__main__":
    main()
