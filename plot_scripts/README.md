# Plot Scripts

这组脚本只读 `summary.csv`、`per_run_summary.csv`、`per_object_summary.csv`，不依赖训练/评估代码。默认输出写到 `plot_scripts/generated/`。

## 共享用法

```bash
python /rl-grasp-refine/plot_scripts/<script>.py
python /rl-grasp-refine/plot_scripts/<script>.py --group group_b
python /rl-grasp-refine/plot_scripts/<script>.py --labels drop-only stb-rwd-5x-full
```

常用参数：

- `--root`：实验结果根目录，默认 `/rl-grasp-refine/outputs/unseen_test_formal`
- `--group`：方法组，默认看各脚本预设
- `--labels`：显式指定方法，覆盖 `--group`
- `--out-dir`：图片输出目录
- `--formats`：输出格式，默认 `png pdf`
- `--dpi`：默认 `330`

## 脚本索引

| Script | 使用数据 | 图含义 | 使用方式 |
| --- | --- | --- | --- |
| `fig01_main_overall_performance.py` | `summary.csv` | 各实验 `macro_success_lift` 主结果图，默认 seed-avg + CI | `python .../fig01_main_overall_performance.py [--style point|bar]` |
| `fig02_main_risk_return.py` | `summary.csv` | 正样本破坏率 vs 负样本保留率并列图，默认 seed-avg + CI | `python .../fig02_main_risk_return.py` |
| `fig03_risk_return_scatter.py` | `summary.csv` | 收益-风险散点图，每个实验一个 seed-avg 点 | `python .../fig03_risk_return_scatter.py` |
| `fig04_mechanism_triplet.py` | `summary.csv` | `t_cover_delta` / `t_edge_delta` / `prob_delta_mean` 三联图，默认 seed-avg + CI | `python .../fig04_mechanism_triplet.py [--style point|bar]` |
| `fig05_reward_scale_response.py` | `summary.csv` | reward scan 响应图，默认只画 `group_b`，默认指标 `macro_success_lift` | `python .../fig05_reward_scale_response.py [--metric macro_success_lift|prob_delta_mean|neg_hold] [--style line|bar]` |
| `fig06_object_stability_boxplot.py` | `per_object_summary.csv` | 跨 object 稳定性箱线图，先按同一 `object_id` 做 3-seed 平均 | `python .../fig06_object_stability_boxplot.py` |
| `fig07_object_stability_bar.py` | `summary.csv` | across-object 稳定性摘要柱状图，默认画 IQR | `python .../fig07_object_stability_bar.py [--metric iqr|std]` |
| `fig08_per_object_rank_curve.py` | `per_object_summary.csv` | per-object 排序曲线，先按 `object_id` 做 3-seed 平均后排序 | `python .../fig08_per_object_rank_curve.py` |
| `fig09_per_run_overlay.py` | `summary.csv` + `per_run_summary.csv` | experiment-level 均值 + CI，并叠加 3 个单 seed run 点 | `python .../fig09_per_run_overlay.py` |

## 默认分组

- `all_formal` / `group_a`：`drop-only`, `stb-rwd-5x-full`, `stb-rwd-10x-full`, `stb-rwd-15x-full`, `stb-rwd-5x-full-latefus-128-epi`, `wo-tac-rwd`, `wo-tac-sem-n-rwd`
- `group_b`：`drop-only`, `stb-rwd-5x-full`, `stb-rwd-10x-full`, `stb-rwd-15x-full`
