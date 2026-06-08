下面这份可以直接交给 GPT-5.5 作为 formal test 监控量说明。

**Formal Test Outputs**

每个实验目录下都有 5 类文件：

```text
episode_records.csv      # episode 级原始记录，最重要
per_object_summary.csv   # test_seed × object_id 聚合
per_run_summary.csv      # test_seed 聚合
summary.csv              # experiment 聚合
metadata.json            # checkpoint / manifest / protocol / resolved objects
```

生成逻辑主要在 [best_checkpoint_pipeline.py](/rl-grasp-refine/src/evaluation/best_checkpoint_pipeline.py) 和 [rollout_diagnostics.py](/rl-grasp-refine/src/metrics/rollout_diagnostics.py)。

**1. 直接记录的 Episode 级监控量**

`episode_records.csv` 是最底层数据，可重新构造几乎所有论文表图。

身份与协议：
- `experiment_name`
- `test_seed`
- `object_id`
- `source_global_id`
- `source_object_id`
- `trial_status`
- `failure_reason`

抓取结果：
- `drop_success`: 微调后 drop test 是否成功，0/1。
- `legacy_drop_success_before`: 数据集中 legacy grasp 原始成功标签，通常 0/1。
- `success_delta = drop_success - legacy_drop_success_before`
- `positive_drop_event`: 原本成功但微调后失败，即 degradation event。
- `negative_recovery_event`: 原本失败但微调后成功，即 recovery event。

稳定性预测与校准：
- `raw_logit_before`
- `raw_logit_after`
- `prob_before`
- `prob_after`
- `prob_delta = prob_after - prob_before`

触觉接触语义：
- `t_cover_before`
- `t_cover_after`
- `t_cover_delta`
- `t_edge_before`
- `t_edge_after`
- `t_edge_delta`

动作：
- `action_0` 至 `action_5`
- `translation_norm = ||action[:3]||`
- `rotation_norm = ||action[3:6]||`
- `saturation_rate`: 6 维动作中 `abs(action_i) > 0.9` 的比例。

表征诊断：
- `latent_before_norm/mean/std`
- `latent_after_norm/mean/std`
- `policy_latent_hidden_before_norm/mean/std`

**2. 已直接输出的聚合监控量**

`per_object_summary.csv` 是 `test_seed × object_id` 级别，适合论文 object-level CI 的基础。

核心结果：
- `success_lift_vs_dataset = mean(drop_success) - mean(legacy_drop_success_before)`
- `positive_drop_rate`: legacy success 子集上的退化率。
- `negative_hold_rate`: 命名历史遗留，实际含义是 legacy fail 子集上的 recovery rate。
- `t_cover_delta_mean`
- `t_edge_delta_mean`
- `prob_delta_mean`
- `num_episodes`
- `positive_count`, `negative_count`
- `positive_drop_count`, `negative_hold_count`

动作分布：
- `action_translation_norm_mean/std`
- `action_rotation_norm_mean/std`
- `action_dim_0_mean/std` 至 `action_dim_5_mean/std`
- `action_dim_0_saturation_rate` 至 `action_dim_5_saturation_rate`
- `action_saturation_rate`

动作-结果相关性：
- `corr_translation_norm_prob_delta`
- `corr_rotation_norm_prob_delta`
- `corr_translation_norm_success_delta`
- `corr_rotation_norm_success_delta`
- `corr_dim_i_prob_delta`
- `corr_dim_i_positive_drop`
- `corr_dim_i_negative_recovery`

动作分桶：
- `action_bin_trans_bin_{0..3}_count`
- `action_bin_trans_bin_{0..3}_prob_delta_mean`
- `action_bin_trans_bin_{0..3}_success_delta_mean`
- `action_bin_trans_bin_{0..3}_drop_rate`
- `action_bin_rot_bin_{0..3}_...`

bin 定义是 action norm / sqrt(3) 后分到：
`[0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0]`。

校准与判别：
- `calibrator_after_brier`: `prob_after` vs `drop_success`
- `calibrator_before_brier_vs_legacy`: `prob_before` vs `legacy_drop_success_before`
- `calibrator_prob_after_auc`
- `calibrator_raw_logit_after_auc`
- `calibrator_prob_before_auc_vs_legacy`
- `calibrator_raw_logit_before_auc_vs_legacy`
- `calibrator_prob_delta_recovery_auc`
- `calibrator_neg_prob_delta_degradation_auc`

`per_run_summary.csv` 是每个 `test_seed` 聚合；`summary.csv` 是实验级聚合，包含 mean/std/CI。注意：`summary.csv` 里的 CI 是 pipeline 原有 object-bootstrap CI；如果论文要 object-level t interval，应从 `per_object_summary.csv` 重算。

**3. 可进一步聚合出的论文实证形式**

主结果表：
- 以 `no-action` 为 reference，配对 `test_seed + object_id` 后相减。
- 可构造：
  - `Success Gain`
  - `Excess Degradation`
  - `Excess Recovery`
- 推荐论文 CI：同一 object 先跨 test seed 平均，再对 object 做 95% t interval。

风险-收益图：
- x/y 可用：
  - `positive_drop_rate - no_action_positive_drop_rate`
  - `negative_hold_rate - no_action_negative_hold_rate`
- 适合支撑“收益-风险权衡”。

稳定性预测是否有效：
- `prob_delta_mean` vs `success_lift_vs_dataset`
- `prob_delta > 0` 的比例 vs success gain
- `prob_after` vs `drop_success`
- `prob_delta_recovery_auc`
- `neg_prob_delta_degradation_auc`

概率校准是否成立：
- 已有：
  - Brier
  - AUC
  - raw logit AUC vs calibrated prob AUC
- 可从 episode 级进一步算：
  - ECE
  - NLL/BCE
  - reliability diagram
  - before/after calibration table

动作机制：
- 动作幅度与结果耦合：
  - `translation_norm/rotation_norm` vs `prob_delta`
  - `translation_norm/rotation_norm` vs `success_delta`
  - action bin 中的 `prob_delta_mean` 与 `success_delta_mean`
- 可进一步构造统一 6-DoF bin：
  - 用 episode 级 `sqrt(||trans||^2 + ||rot||^2)` 或两个 bin 的组合规则重分桶。

触觉机制：
- `t_cover_delta_mean`
- `t_edge_delta_mean`
- paired no-action excess contact delta
- 可和 success/recovery/degradation 关联，用于说明触觉语义是否改善最终抓取。

失败模式：
- `trial_status`
- `failure_reason`
- 可按方法、object、action bin 聚合，判断失败是否集中在 contact lost、undesired contact、drop failure 等。

跨物体稳健性：
- `per_object_summary.csv` 支持：
  - object-level rank curve
  - object-level boxplot
  - worst-object / best-object
  - across-object std / IQR
  - object-level paired t interval

**关键注意事项**

- `negative_hold_rate` 在代码名上像 hold，但论文解释应写成 recovery rate。
- `summary.csv` 可快速查看趋势，但论文最终表建议从 `per_object_summary.csv` 重新按目标 CI 口径计算。
- 校准的 `before` 指标多数是相对 legacy label；`after` 指标是相对 final drop success。
- episode 数很多，但泛化单位是 object；论文 CI 最好以 object 为独立样本单位。