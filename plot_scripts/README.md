# Plot Scripts

这组脚本只读 formal eval 已有 CSV，不依赖训练/评估代码。当前所有图统一采用 no-action-adjusted 口径：自动额外读取 `no-action/per_object_summary.csv`，按 `(test_seed, object_id)` 做 `method - no-action` 配对差分，再汇总作图。默认只输出 `png`，输出会自动写到 `<out-dir>/<group>/`，例如 `plot_scripts/generated/main/` 或 `plot_scripts/generated/ablation/`。

## 共享用法

```bash
python /rl-grasp-refine/plot_scripts/<script>.py
python /rl-grasp-refine/plot_scripts/<script>.py --group ablation
python /rl-grasp-refine/plot_scripts/<script>.py --labels no-action rand-action full-latefus-128-epi
python /rl-grasp-refine/plot_scripts/<script>.py --group main --print-data-format csv
./plot_scripts/plot_group.sh main --print-data-format csv
python /rl-grasp-refine/plot_scripts/fig10_full_training_curves.py --smooth-window 15 --align common
```

常用参数：

- `--root`：实验结果根目录，默认 `/rl-grasp-refine/outputs/unseen_test_formal`
- `--group`：方法组，默认看各脚本预设
- `--labels`：显式指定方法，覆盖 `--group`
- `--out-dir`：图片基础输出目录，脚本会自动追加 `--group` 子目录
- `--dpi`：默认 `330`
- `--print-data` / `--no-print-data`：默认打印该图实际使用的最终数据，可用 `--no-print-data` 关闭
- `--print-data-format`：打印格式，`table` 或 `csv`

注意：`--labels` 只决定最终画哪些方法；`no-action` baseline 会被额外读取用于差分。如果结果目录缺少 `no-action`，adjusted 图会直接报错。

批量绘图：`plot_group.sh` 会对指定 group 运行全部 8 个脚本，并把每个脚本打印的最终数据汇总到 `<out-dir>/<group>/plot_data_<group>.txt`。

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
| `fig10_full_training_curves.py` | Full 三个 seed 的 `metrics.jsonl` | Full policy training-time curve；Validation/Training mean 实线，阴影为 across-seed std | training 专用 |

## Training Curve

`fig10_full_training_curves.py` 不读取 formal eval CSV，也不属于 `plot_group.sh`。默认读取 Full 的三个训练目录：

- `outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus`
- `outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_seed8`
- `outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_seed9`

默认指标是 `validation/outcome/success_lift_vs_dataset` 和 `outcome/success_lift_vs_dataset`。`--align common` 只画所有 seed 都有的 step；seed9 未完成时会自然截断，seed9 到 500 后重跑即可得到完整曲线。`--smooth-window` 是对每个 seed/metric 的可用点做 rolling mean，`1` 表示不平滑。输出为 `plot_scripts/generated/training/fig10_full_training_curves.png` 和对应 `_data.csv`。

## 默认分组

- `main`：`no-action`, `rand-action`, `drop-only-latent-only-128-epi`, `full-latefus-128-epi`
- `ablation`：`drop-only-latent-only-128-epi`, `wo-onl-cal_latefus_128-epi`, `wo-stb-rwd_latefus_128-epi`, `wo-tac-rwd_latefus_128-epi`, `wo-tac-sem-n-rwd_latefus_128-epi`, `full-latefus-128-epi`
