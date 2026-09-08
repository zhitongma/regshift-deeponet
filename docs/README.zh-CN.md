# 中文使用说明

这是从“最终回复”整理出来的新代码库。原论文目录未修改。

## 从哪里开始

| 想做的事 | 入口 |
|---|---|
| 看模型实现 | `src/models/`，有界模型在 `shift_deeponet.py` 中 |
| 看最终实验参数 | `configs/revision/` |
| 生成参数、模拟、训练、评估 | `pipeline/`，推荐通过 `scripts/run_experiment.py` 调用 |
| 复画论文全部图件 | `python scripts/reproduce_figures.py` |
| 导出最终汇总表和计时分析 | `python scripts/export_tables.py` |
| 查看实验分析代码 | `analysis/E0/` 至 `analysis/E11/` |
| 检查文件完整性及模型配置 | `python scripts/check_repo.py --models` |
| 查看图号与脚本的对应关系 | [FIGURES.md](FIGURES.md) |

## 安装和绘图

推荐 Python 3.12。在代码库根目录打开终端：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/plotting.txt
python scripts/check_repo.py
python scripts/reproduce_figures.py
```

结果保存在 `outputs/figures/`，统一使用 `Figure_1a.pdf`、`Figure_2.pdf` 等投稿文件名。正文有 13 个图件文件，补充材料有 9 个。

图 1d 和图 2 是 TikZ 绘制的示意图，需要 LaTeX 的 `pdflatex`、`standalone` 和 `tikz`。暂时没有 LaTeX 时，可用 `--skip-diagrams` 生成其余 20 个图件。

## 训练和数据

训练前安装 `requirements/training.txt`。先运行下面的命令预览实际训练入口，不会启动实验：

```bash
python scripts/run_experiment.py \
  --config configs/revision/E0_bounded/e0_regshift_highbudget_iid.yaml \
  --model regshift --seed 42 --dry-run
```

正式执行时去掉 `--dry-run`，增加 `--data-dir /你的数据目录`。该目录应包含 `processed/train.npz`、`val.npz`、`test.npz`、`scaler.npz`，以及所需的 `parameters/`。更多模型与参数见 [REPRODUCING.md](REPRODUCING.md)。

本库只保留几 MB 的绘图输入、汇总表和三个代表性案例所需的数组。完整原始数据、训练数据、模型权重、全量预测场都没有放入。**这些少量绘图数据不能替代完整训练数据。** 最终服务器实验的全量权重和数据仍需从外部备份恢复。

## 本次整理处理的内容

- 补入原补充包遗漏的 `generate_restored_figures.py` 和两个 TikZ 图源文件。
- 从旧绘图脚本中提取最终论文仍使用的图 1a–c、S1、S5；不再把已被替换的旧版图作为默认绘图入口。
- 将原包附带的 FNO padding 修复补丁合入模型文件，并保留补丁来源。
- 保留最终有界模型配置及原模型初始化，不擅自修改实验算法；对顶层误放的界限配置增加报错，避免静默按无界模型运行。
- 提供统一训练入口，替代依赖旧中文目录和服务器路径的任务启动脚本。
- 生成 `.gitignore`、依赖列表、文档、数据校验清单及回归检查。

这是根据本地最终包和补丁整理的代码库，未与最终服务器源码逐文件比对，也没有重跑完整论文训练。已经验证的范围见 [VALIDATION.md](VALIDATION.md)。

上传 GitHub 时使用这个新文件夹即可；`outputs/`、权重、完整数据和环境目录已配置为不纳入 Git。许可证和数据仓库链接由作者确定后补充，见 [GITHUB.md](GITHUB.md)。
