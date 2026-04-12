# Plot Scripts

这组脚本只读 formal eval 已有 CSV，不依赖训练/评估代码。当前所有图统一采用 no-action-adjusted 口径：自动额外读取 `no-action/per_object_summary.csv`，按 `(test_seed, object_id)` 做 `method - no-action` 配对差分，再汇总作图。默认只输出 `png`，输出会自动写到 `<out-dir>/<group>/`，例如 `plot_scripts/generated/main/` 或 `plot_scripts/generated/ablation/`。

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

注意：`--labels` 只决定最终画哪些方法；`no-action` baseline 会被额外读取用于差分。如果结果目录缺少 `no-action`，adjusted 图会直接报错。

## 脚本索引

| Script | 使用数据 | 图含义 | 默认组 |
| --- | --- | --- | --- |
| `fig01_main_overall_performance.py` | `per_object_summary.csv` | Success Gain over No-Action，object-bootstrap CI | `main` |
| `fig02_main_risk_return.py` | `per_object_summary.csv` | Excess Degradation / Excess Recovery，object-bootstrap CI | `main` |
| `fig03_risk_return_scatter.py` | `per_object_summary.csv` | adjusted 风险-收益散点，每个实验一个点 | `main` |
| `fig04_mechanism_triplet.py` | `per_object_summary.csv` | Excess T-Cover / T-Edge / Probability Delta 三联图 | `ablation` |
| `fig06_object_stability_boxplot.py` | `per_object_summary.csv` | adjusted object success gain 分布，先按同一 `object_id` 做 3-seed 平均 | `ablation` |
| `fig07_object_stability_bar.py` | `per_object_summary.csv` | adjusted object success gain 的 IQR 或 std | `ablation` |
| `fig08_per_object_rank_curve.py` | `per_object_summary.csv` | adjusted per-object success gain 排序曲线 | `ablation` |
| `fig09_per_run_overlay.py` | `per_object_summary.csv` | adjusted experiment 均值 + CI，并叠加 3 个 test-seed run mean 点 | `main` |

## 默认分组

- `main`：`no-action`, `rand-action`, `drop-only-latent-only-128-epi`, `full-latefus-128-epi`
- `ablation`：`drop-only-latent-only-128-epi`, `wo-onl-cal_latefus_128-epi`, `wo-stb-rwd_latefus_128-epi`, `wo-tac-rwd_latefus_128-epi`, `wo-tac-sem-n-rwd_latefus_128-epi`, `full-latefus-128-epi`
