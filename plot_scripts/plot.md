下面是当前 `plot_scripts` 的作图口径摘要，可直接给 GPT thinking 看。

**全局口径**
当前所有图都是 **no-action-adjusted** 口径，不直接解释 raw metric。

数据源只读 formal eval 已有 CSV：
- `summary.csv`
- `per_run_summary.csv`
- `per_object_summary.csv`

但当前 adjusted 图的主数据实际主要来自：
- `per_object_summary.csv`

baseline 固定为：
```text
no-action
```

每个方法的 adjusted value 按 `(test_seed, object_id)` 与 no-action 配对：

```text
adjusted_value(method, test_seed, object_id)
=
value(method, test_seed, object_id)
-
value(no-action, test_seed, object_id)
```

`--labels` 或 `--group` 只决定最终画哪些方法；`no-action` 会被自动额外加载用于差分。如果缺少 `no-action`，脚本报错。

CI 不是用 `summary.csv` 里的原始 CI，也不是 CI 端点相减。当前 CI 是：
```text
1. 先按 (method, test_seed, object_id) 做 method - no-action
2. 再对每个 (method, object_id) 在 3 个 test_seed 上求均值
3. 得到每个 method 的 13 个 object-level adjusted values
4. 对这 13 个 object values 做 bootstrap
5. 取 2.5% / 97.5% 分位数作为 95% CI
```

所以 CI 含义是：
```text
object-level paired bootstrap CI
```

它不是 episode-level paired CI。

**分组**
`main`：
```text
no-action
rand-action
drop-only-latent-only-128-epi
full-latefus-128-epi
```

`ablation`：
```text
drop-only-latent-only-128-epi
wo-onl-cal_latefus_128-epi
wo-stb-rwd_latefus_128-epi
wo-tac-rwd_latefus_128-epi
wo-tac-sem-n-rwd_latefus_128-epi
full-latefus-128-epi
```

**fig01_main_overall_performance.py**
图含义：
```text
Success Gain over No-Action
```

原始列：
```text
success_lift_vs_dataset
```

作图数据：
```text
adjusted success lift = method success_lift_vs_dataset - no-action success_lift_vs_dataset
```

统计单位：
```text
experiment-level mean + object-bootstrap CI
```

默认图形：
```text
point + error bar
```

可选：
```text
--style bar
```

**fig02_main_risk_return.py**
图含义：
```text
Excess Degradation over No-Action
Excess Recovery over No-Action
```

原始列：
```text
positive_drop_rate
negative_hold_rate
```

作图数据：
```text
excess degradation = method positive_drop_rate - no-action positive_drop_rate
excess recovery    = method negative_hold_rate - no-action negative_hold_rate
```

统计单位：
```text
experiment-level mean + object-bootstrap CI
```

图形：
```text
grouped bar
```

解释：
- `Excess Degradation` 越小越好
- `Excess Recovery` 越大越好
- 0 线表示 no-action retest floor

**fig03_risk_return_scatter.py**
图含义：
```text
adjusted risk-return scatter
```

横轴：
```text
Excess Degradation over No-Action
```

纵轴：
```text
Excess Recovery over No-Action
```

原始列：
```text
positive_drop_rate
negative_hold_rate
```

统计单位：
```text
每个 experiment 一个 adjusted experiment-level 点
```

CI：
```text
不画 error bar，只画 mean 点
```

解释：
- 左边更低风险
- 上边更高恢复收益
- 理想方向是左上

**fig04_mechanism_triplet.py**
图含义：
```text
three mechanism metrics, all no-action-adjusted
```

三个 panel：
```text
Excess T-Cover Delta over No-Action
Excess T-Edge Delta over No-Action
Excess Probability Delta over No-Action
```

原始列：
```text
t_cover_delta_mean
t_edge_delta_mean
prob_delta_mean
```

作图数据：
```text
method metric - no-action metric
```

统计单位：
```text
experiment-level mean + object-bootstrap CI
```

默认图形：
```text
point + error bar
```

可选：
```text
--style bar
```

解释：
这些不是 raw tactile delta，而是扣除 no-action retest floor 后的 excess mechanism signal。

**fig06_object_stability_boxplot.py**
图含义：
```text
object-level Success Gain over No-Action distribution
```

原始列：
```text
success_lift_vs_dataset
```

处理：
```text
1. 按 (test_seed, object_id) 做 method - no-action
2. 对同一个 object_id 的 3 个 test_seed 求平均
3. 每个 method 得到 13 个 object-level adjusted success gain
4. 对这 13 个 object values 画 boxplot
```

统计单位：
```text
object
```

不叠加 seed 原始点。

解释：
箱线图反映跨 object 的 adjusted success gain 分布，不是 seed 分布。

**fig07_object_stability_bar.py**
图含义：
```text
Across-object stability summary of adjusted success gain
```

原始列：
```text
success_lift_vs_dataset
```

处理：
```text
1. 先得到每个 method 的 13 个 object-level adjusted success gain
2. 默认计算这 13 个 object values 的 IQR
```

默认 metric：
```text
iqr
```

可选：
```text
--metric std
```

统计单位：
```text
object-level adjusted success gain
```

解释：
- IQR/std 越小，说明该方法跨 object 的净收益更稳定
- 它不是越大越好指标
- 它不表示平均性能，只表示跨 object 离散度

**fig08_per_object_rank_curve.py**
图含义：
```text
per-object adjusted success gain rank curve
```

原始列：
```text
success_lift_vs_dataset
```

处理：
```text
1. 按 (test_seed, object_id) 做 method - no-action
2. 对同一个 object_id 的 3 个 test_seed 求平均
3. 每个 method 内部按 adjusted success gain 从高到低排序
4. 横轴是 rank，不是 object_id
```

统计单位：
```text
object
```

解释：
曲线越高，说明更多 object 上有更高净收益；曲线尾部可看失败 object。

**fig09_per_run_overlay.py**
图含义：
```text
experiment-level Success Gain over No-Action with run-level dots
```

原始列：
```text
success_lift_vs_dataset
```

主体：
```text
experiment-level adjusted mean + object-bootstrap CI
```

散点：
```text
每个 test_seed 一个 adjusted run mean
```

run 点计算：
```text
adjusted_run_mean(method, test_seed)
=
mean over objects [
  method success_lift_vs_dataset(test_seed, object_id)
  -
  no-action success_lift_vs_dataset(test_seed, object_id)
]
```

解释：
- error bar 是 object-bootstrap CI
- 三个散点是 3 个 formal test seed
- 这些 seed 是 evaluation seed，不是 training seed

**批量脚本 plot_group.sh**
用途：
```text
对指定 group 运行全部 8 个 figure scripts
```

用法：
```bash
./plot_scripts/plot_group.sh main
./plot_scripts/plot_group.sh ablation
```

输出：
```text
plot_scripts/generated/<group>/*.png
plot_scripts/generated/<group>/plot_data_<group>.txt
```

`plot_data_<group>.txt` 会保存每个脚本实际用于作图的最终数据。

可选 CSV 打印：
```bash
./plot_scripts/plot_group.sh main --print-data-format csv
```

**当前打印行为**
每个 figure script 默认都会打印最终作图数据。

关闭打印：
```bash
--no-print-data
```

打印格式：
```bash
--print-data-format table
--print-data-format csv
```

**最重要的解释边界**
当前 adjusted 图可以解释为：

```text
relative policy effect beyond no-action retest floor
```

不要解释为：

```text
episode-level live-live causal effect
```

也不要把 tactile raw delta 当绝对触觉改善量解释。机制图应解释为：

```text
excess tactile/probability change over no-action baseline
```