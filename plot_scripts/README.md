# Plot Scripts

这组脚本只读 `summary.csv`、`per_run_summary.csv`、`per_object_summary.csv`，不依赖训练/评估代码。默认只输出 `png`。输出会自动写到 `<out-dir>/<group>/`，例如 `plot_scripts/generated/main/` 或 `plot_scripts/generated/ablation/`。

## 共享用法

```bash
python /rl-grasp-refine/plot_scripts/<script>.py
python /rl-grasp-refine/plot_scripts/<script>.py --group ablation
python /rl-grasp-refine/plot_scripts/<script>.py --labels no-action rand-action full-latefus-128-epi
```

常用参数：

- `--root`：实验结果根目录，默认 `/rl-grasp-refine/outputs/unseen_test_formal`
- `--group`：方法组，默认看各脚本预设
- `--labels`：显式指定方法，覆盖 `--group`
- `--out-dir`：图片基础输出目录，脚本会自动追加 `--group` 子目录
- `--dpi`：默认 `330`

## 脚本索引

| Script | 使用数据 | 图含义 | 默认组 |
| --- | --- | --- | --- |
| `fig01_main_overall_performance.py` | `summary.csv` | 主结果 `macro_success_lift`，seed-avg + CI | `main` |
| `fig02_main_risk_return.py` | `summary.csv` | 正样本破坏率 vs 负样本保留率，seed-avg + CI | `main` |
| `fig03_risk_return_scatter.py` | `summary.csv` | 风险-收益散点，每个实验一个 seed-avg 点 | `main` |
| `fig04_mechanism_triplet.py` | `summary.csv` | `t_cover_delta` / `t_edge_delta` / `prob_delta_mean` 三联图 | `ablation` |
| `fig06_object_stability_boxplot.py` | `per_object_summary.csv` | 跨 object 成功率提升分布，先按同一 `object_id` 做 3-seed 平均 | `ablation` |
| `fig07_object_stability_bar.py` | `summary.csv` | across-object 稳定性摘要，默认 IQR | `ablation` |
| `fig08_per_object_rank_curve.py` | `per_object_summary.csv` | per-object 排序曲线，先按 `object_id` 做 3-seed 平均 | `ablation` |
| `fig09_per_run_overlay.py` | `summary.csv` + `per_run_summary.csv` | experiment-level 均值 + CI，并叠加 3 个 test-seed 点 | `main` |

## 默认分组

- `main`：`no-action`, `rand-action`, `drop-only-latent-only-128-epi`, `full-latefus-128-epi`
- `ablation`：`drop-only-latent-only-128-epi`, `wo-onl-cal_latefus_128-epi`, `wo-stb-rwd_latefus_128-epi`, `wo-tac-rwd_latefus_128-epi`, `wo-tac-sem-n-rwd_latefus_128-epi`, `full-latefus-128-epi`
