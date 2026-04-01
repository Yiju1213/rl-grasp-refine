# 正式训练监控信号说明

本文件结合 [v5.md](./v5.md) 与当前代码实现，整理正式训练主链中的监控信号。

这里的“监控信号”特指训练过程中由代码显式汇总并写入日志系统的字段，主要来自：

- `src/rl/trainer.py`
- `src/rl/ppo_agent.py`
- `src/utils/logger.py`
- `src/utils/system_diagnostics.py`
- `src/envs/grasp_refine_env.py`
- `src/envs/reward_manager.py`
- `src/envs/pybullet_scene.py`

若与旧文档冲突，以当前代码为准。

---

## 1. 信号写到哪里

正式训练中的监控信号主要有三类：

1. `metrics.jsonl`
   - 迭代级聚合信号。
   - 也是 TensorBoard 的同源数据。
2. `run.log`
   - 把同一轮的聚合信号格式化后打印出来，便于直接看训练进度。
   - 它不是新信号源，只是 `metrics.jsonl` 的文本展开。
3. `episode_metrics.jsonl`
   - 可选的样本级信号。
   - 只有 `logging.sample_metrics.enabled: true` 时才会写。

当前正式实验默认配置见 `configs/experiment/exp_debug.yaml`：

- `logging.metric_profile: paper`
- `logging.diagnostics.enabled: false`
- `logging.sample_metrics.enabled: false`

因此，**正式训练默认会看到的是 paper profile 下的一组核心信号**；代码里还有一批更偏工程诊断的扩展信号，只有切到 `full` profile，或再打开 `diagnostics` 后才会出现。

---

## 2. 命名规则与统计口径

### 2.1 命名规则

所有聚合信号都使用：

- `<module>/<metric>`

例如：

- `reward/total_mean`
- `calibrator/posterior_trace_post_update`
- `ppo/approx_kl`

如果开启 validation，同类信号会再复制一份，加上前缀：

- `validation/<module>/<metric>`

例如：

- `validation/outcome/success_rate_live_after`
- `validation/contact/t_cover_after_mean`

它们的含义与训练信号相同，只是统计对象变成 validation object split，且**不参与 PPO 更新**。

### 2.2 统计口径

需要区分两种口径：

1. 按 `valid_for_learning=True` 的有效 episode 统计
   - 大多数 `reward/*`、`contact/*`、`calibrator/*`、`action/*`、`ppo/*` 都属于这一类。
   - 这些 episode 会真正进入 `RolloutBuffer`，用于 PPO 和 calibrator 更新。
2. 按所有 rollout attempt 统计
   - `collection/*`
   - `outcome/system_invalid_rate`
   - `outcome/trial_status_<status>_rate`
   - 这一类会把无效样本和系统错误也算进去，因此更适合看训练链路是否健康。

### 2.3 三个基础语义量

这几个量会在很多信号里反复出现：

- `t_cover`
  - 接触覆盖率。
  - 当前实现是 tactile map 里 active contact 点数占有效点数的比例。
- `t_edge`
  - 接触点到 sensor 边界的归一化距离均值。
  - 越大表示接触更偏向 sensor 内部，而不是贴边。
- calibrated probability
  - `p = sigmoid(a * logit + b)`。
  - 由 online logit calibrator 给出。
- `posterior_trace`
  - calibrator 后验协方差 `P` 的迹 `tr(P)`。
  - 越大通常表示当前校准不确定性越高。

---

## 3. 迭代级聚合信号

下表中的“出现条件”按正式训练配置理解：

- `默认`：`metric_profile: paper` 下就会记录
- `full`：需要 `metric_profile: full`
- `full + diagnostics`：还需要 `logging.diagnostics.enabled: true`

### 3.1 `collection/*`

这些信号反映采样链路本身，而不是策略质量。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `collection/attempts_total` | 当前 iteration 为拿到足够有效 episode，一共尝试了多少次 rollout。 | 默认 |
| `collection/valid_episodes` | 当前 iteration 最终收集到多少个有效 episode。正常情况下应等于 `batch_episodes`。 | 默认 |
| `collection/valid_rate` | `valid_episodes / attempts_total`。越低说明无效样本、系统错误或场景失败越多。 | 默认 |
| `collection/scene_rebuild_performed` | 本轮采样前是否触发 worker scene rebuild。`1` 表示触发。 | full |
| `collection/scene_rebuild_workers` | 本轮有多少个 worker 参与了 scene rebuild。 | full |
| `collection/worker_recycle_performed` | 本轮采样前是否触发 worker recycle。`1` 表示触发。 | full |
| `collection/worker_recycle_slots` | 本轮 recycle 了多少个 worker slot。 | full |
| `collection/worker_recycle_prefetched` | 本轮或下一轮 recycle 之前，后台预热了多少个 standby worker。 | full |
| `collection/worker_recycle_prefetch_ready` | 当前已经处于 ready 状态的 standby worker 数量。 | full |

### 3.2 `outcome/*`

这些信号直接反映 refinement 后抓取结果，以及和数据集原始抓取标签的对比。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `outcome/success_rate_live_after` | refinement 后 live release trial 的成功率，即 `drop_success` 的均值。 | 默认 |
| `outcome/success_rate_dataset_before` | 数据集原始抓取标签 `legacy_drop_success_before` 的均值。它是 before 基线，不是 live 重放结果。 | 默认 |
| `outcome/success_lift_vs_dataset` | `success_rate_live_after - success_rate_dataset_before`。反映 refine 后相对数据集基线的提升或下降。 | 默认 |
| `outcome/drop_rate_after_given_dataset_positive` | 条件掉落率 `P(drop_success_after_live = 0 | legacy_drop_success_before = 1)`。反映原本数据集正样本在 refine 后掉落的比例。 | 默认 |
| `outcome/hold_rate_after_given_dataset_negative` | 条件保持率 `P(drop_success_after_live = 1 | legacy_drop_success_before = 0)`。反映原本数据集负样本在 refine 后被纠正并最终 hold 住的比例。 | 默认 |
| `outcome/dataset_positive_count` | 当前有效 episode 中，`legacy_drop_success_before = 1` 的样本数。用于给条件掉落率提供分母规模。 | full |
| `outcome/dataset_negative_count` | 当前有效 episode 中，`legacy_drop_success_before = 0` 的样本数。用于给条件保持率提供分母规模。 | full |
| `outcome/system_invalid_rate` | 所有 attempt 中，`trial_status` 以 `system_` 开头的比例。用于看环境/渲染/仿真链路错误。 | full |
| `outcome/trial_status_<status>_rate` | 所有 attempt 中，某个 `trial_status` 的占比。默认只保留非 `system_*` 状态；`full` 下会连 `system_*` 一起保留。 | 默认（非 system）/ full（全部） |

当前代码里，`<status>` 可能出现的主要取值如下：

| `trial_status` | 含义 |
| --- | --- |
| `success` | refine 后重新观测成功，release 后仍保持接触，最终判为成功。 |
| `failure_release_drop` | refine 后能完成观测与 release，但 release 过程中或结束后失去接触。 |
| `failure_contact_lost` | refine 重建抓取时，连初始接触都没重新建立起来。 |
| `failure_pre_release_drop` | refine 重建阶段一度建立接触，但在达到目标夹持力或 settle 过程中丢失了接触。 |
| `failure_effort_timeout` | refine 重建阶段在限定时间内没达到目标夹持力。 |
| `failure_interference` | refine 后 undesired contact 比 baseline 更差，判为干扰失败。 |
| `system_invalid_observation` | after observation 缺失或被标记为无效。 |
| `system_sim_error` | 仿真、观测或 release 过程出现系统级异常。 |

### 3.3 `reward/*`

这些信号对应论文对齐后的三项 reward 以及总 reward。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `reward/total_mean` | 单步总 reward 的均值。 | 默认 |
| `reward/total_std` | 单步总 reward 的标准差。波动过大时可帮助判断 reward 是否过噪。 | full |
| `reward/drop_mean` | `R_drop` 的均值。成功给正值，失败给负值。 | 默认 |
| `reward/stability_mean` | `R_stability` 的均值，即稳定性概率提升项在 uncertainty 衰减后的均值。 | 默认 |
| `reward/contact_mean` | `R_contact` 的均值，即对低 `t_cover` / 低 `t_edge` 的软惩罚均值。 | 默认 |

### 3.4 `contact/*`

这些信号反映论文状态量 `t = [t_cover, t_edge]` 的 before / after 水平，以及 refine 带来的变化。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `contact/t_cover_before_mean` | refine 前 `t_cover` 的均值。 | 默认 |
| `contact/t_cover_before_std` | refine 前 `t_cover` 的标准差。 | full |
| `contact/t_cover_after_mean` | refine 后 `t_cover` 的均值。 | 默认 |
| `contact/t_cover_after_std` | refine 后 `t_cover` 的标准差。 | full |
| `contact/t_cover_delta_mean` | `t_cover_after - t_cover_before` 的均值。正值表示覆盖率平均提升。 | 默认 |
| `contact/t_edge_before_mean` | refine 前 `t_edge` 的均值。 | 默认 |
| `contact/t_edge_before_std` | refine 前 `t_edge` 的标准差。 | full |
| `contact/t_edge_after_mean` | refine 后 `t_edge` 的均值。 | 默认 |
| `contact/t_edge_after_std` | refine 后 `t_edge` 的标准差。 | full |
| `contact/t_edge_delta_mean` | `t_edge_after - t_edge_before` 的均值。正值表示接触更向 sensor 内部移动。 | 默认 |

### 3.5 `calibrator/*`

这些信号对应 `v5.md` 里的 `raw logit -> calibrated probability -> posterior uncertainty` 链路。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `calibrator/raw_logit_before_mean` | refine 前 raw stability logit 的均值。 | full |
| `calibrator/raw_logit_after_mean` | refine 后 raw stability logit 的均值。 | full |
| `calibrator/prob_before_mean` | refine 前 calibrated stability probability 的均值。 | 默认 |
| `calibrator/prob_after_mean` | refine 后 calibrated stability probability 的均值。 | 默认 |
| `calibrator/prob_delta_mean` | `prob_after - prob_before` 的均值。正值表示策略平均在提升校准后的稳定性概率。 | 默认 |
| `calibrator/prob_delta_std` | `prob_after - prob_before` 的标准差。 | full |
| `calibrator/prob_delta_positive_rate` | `prob_after > prob_before` 的 episode 比例。 | 默认 |
| `calibrator/posterior_trace_snapshot` | rollout 期间 worker 使用的 calibrator snapshot 的 `tr(P)` 均值。它对应的是**采样时**的 uncertainty。 | 默认 |
| `calibrator/posterior_trace_post_update` | 主进程用本轮 `raw_logit_after + drop_success` 更新 calibrator 后得到的 `tr(P)`。 | 默认 |
| `calibrator/param_a` | 当前 calibrator 参数 `a`。 | full |
| `calibrator/param_b` | 当前 calibrator 参数 `b`。 | full |
| `calibrator/after_brier` | 用 `prob_after` 预测 `drop_success` 的 Brier score，越低越好。 | 默认 |
| `calibrator/after_bce` | 用 `prob_after` 预测 `drop_success` 的 binary cross-entropy，越低越好。 | full |

### 3.6 `ppo/*`

这些信号只出现在训练主链，不会出现在 `validation/*` 下。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `ppo/policy_loss` | PPO policy surrogate loss 的均值。 | 默认 |
| `ppo/value_loss` | value function 的 MSE loss 均值。 | 默认 |
| `ppo/entropy` | 策略分布熵的均值。越大通常表示探索更强。 | 默认 |
| `ppo/total_loss` | `policy_loss + value_loss_coef * value_loss - entropy_coef * entropy` 的均值。 | 默认 |
| `ppo/approx_kl` | PPO 更新前后策略的近似 KL。过高通常表示步子过大。 | 默认 |
| `ppo/clip_fraction` | PPO ratio 被 clip 的样本比例。过高通常表示更新过猛。 | 默认 |
| `ppo/explained_variance` | value 预测对 return 方差的解释程度。越高通常表示 critic 拟合更好。 | 默认 |
| `ppo/grad_norm` | 更新时梯度范数。 | full |
| `ppo/returns_mean` | 当前 batch return 的均值。 | full |
| `ppo/returns_std` | 当前 batch return 的标准差。 | full |
| `ppo/advantages_mean` | 当前 batch 原始 advantage 的均值。 | full |
| `ppo/advantages_std` | 当前 batch 原始 advantage 的标准差。 | full |
| `ppo/value_pred_mean` | PPO 更新前，critic 对 batch 的 value 预测均值。 | full |
| `ppo/value_pred_std` | PPO 更新前，critic 对 batch 的 value 预测标准差。 | full |
| `ppo/policy_log_std_mean` | policy 高斯分布 `log_std` 参数的均值，用来观察探索尺度是否在收缩或发散。 | full |

### 3.7 `action/*`

这些信号用于看策略动作是否过小、过激或大量打到边界。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `action/abs_mean` | 动作各维绝对值的均值。 | full |
| `action/l2_mean` | 动作向量 L2 范数的均值。 | full |
| `action/saturation_rate` | 动作分量绝对值达到 `0.999` 及以上的比例，可近似理解为动作饱和率。 | full |

### 3.8 `timing/*`

这些信号用于看训练吞吐、采样耗时和 worker 运维开销。

| 信号名 | 含义 | 出现条件 |
| --- | --- | --- |
| `timing/policy_forward_s_mean` | rollout 时每个 attempt 的 policy forward 耗时均值。 | full |
| `timing/scene_rebuild_wall_s` | 本轮 scene rebuild 总耗时。 | full |
| `timing/worker_recycle_wall_s` | 本轮 worker recycle 总耗时。 | full |
| `timing/worker_recycle_wait_ready_wall_s` | recycle 过程中等待新 worker ready 的累计耗时。 | full |
| `timing/collect_wall_s` | 本轮训练 rollout collection 总耗时。 | 默认 |
| `timing/update_wall_s` | 本轮 PPO + calibrator update 总耗时。 | 默认 |
| `timing/validation_wall_s` | 本轮 validation 阶段总耗时。注意它是训练主链自己的计时，不带 `validation/` 前缀。 | 默认 |
| `timing/iteration_wall_s` | 整个 iteration 的总耗时。 | 默认 |

### 3.9 `system/*`

这些是系统级诊断信号，默认正式训练不会写；需要：

- `logging.metric_profile: full`
- `logging.diagnostics.enabled: true`

其中一部分还要求：

- 主设备是 CUDA
- 机器上 `nvidia-smi` 可用

#### 进程内存

| 信号名 | 含义 |
| --- | --- |
| `system/process_rss_mb` | 当前主进程 RSS 内存。 |
| `system/process_rss_peak_mb` | 当前主进程历史 RSS 峰值。 |
| `system/process_vms_mb` | 当前主进程虚拟内存大小。 |
| `system/process_swap_mb` | 当前主进程 swap 占用。 |
| `system/process_rss_anon_mb` | 当前主进程匿名页 RSS。 |
| `system/process_rss_file_mb` | 当前主进程文件页 RSS。 |
| `system/process_rss_shmem_mb` | 当前主进程共享内存 RSS。 |

#### cgroup 内存

| 信号名 | 含义 |
| --- | --- |
| `system/cgroup_memory_current_mb` | 当前 cgroup 总内存占用。 |
| `system/cgroup_memory_swap_current_mb` | 当前 cgroup swap 占用。 |
| `system/cgroup_memory_anon_mb` | cgroup 匿名内存占用。 |
| `system/cgroup_memory_file_mb` | cgroup 文件缓存占用。 |
| `system/cgroup_memory_kernel_mb` | cgroup kernel memory 占用。 |
| `system/cgroup_memory_slab_mb` | cgroup slab 占用。 |
| `system/cgroup_memory_shmem_mb` | cgroup shared memory 占用。 |

#### PyTorch / GPU 内存

| 信号名 | 含义 |
| --- | --- |
| `system/gpu_torch_allocated_mb` | 当前主进程 Torch 已分配显存。 |
| `system/gpu_torch_reserved_mb` | 当前主进程 Torch 已保留显存。 |
| `system/gpu_torch_max_allocated_mb` | 当前主进程 Torch 历史分配显存峰值。 |
| `system/gpu_torch_max_reserved_mb` | 当前主进程 Torch 历史保留显存峰值。 |
| `system/gpu_total_used_mb` | 所有 GPU 总显存占用。 |
| `system/gpu_process_reporting_count` | `nvidia-smi` 报告到的相关进程数。 |
| `system/gpu_main_process_used_mb` | 主训练进程自身显存占用。 |
| `system/gpu_worker_process_reporting_count` | 有显存记录的 worker 进程数。 |
| `system/gpu_worker_process_used_mb_sum` | worker 显存占用总和。 |
| `system/gpu_worker_process_used_mb_max` | 单个 worker 显存占用最大值。 |

#### worker 进程健康度

| 信号名 | 含义 |
| --- | --- |
| `system/collector_workers_total` | 当前 active + standby worker 总数。 |
| `system/collector_workers_alive` | 当前仍然存活的 worker 数。 |
| `system/collector_workers_exited` | 已退出的 worker 数。 |
| `system/collector_workers_dead` | 当前不存活的 worker 数。 |
| `system/collector_workers_nonzero_exitcodes` | 以非零退出码结束的 worker 数。 |

---

## 4. `validation/` 前缀信号

如果打开 validation，`Trainer.run_validation()` 会对 validation object split 再做一次 rollout 汇总，并把同类信号加上 `validation/` 前缀。

典型例子：

- `validation/collection/attempts_total`
- `validation/outcome/success_rate_live_after`
- `validation/reward/total_mean`
- `validation/contact/t_cover_after_mean`
- `validation/calibrator/posterior_trace_snapshot`
- `validation/timing/collect_wall_s`

需要注意两点：

1. `validation/*` 不包含 `ppo/*`
   - 因为 validation 不做参数更新。
2. `timing/validation_wall_s` 不在 `validation/` 命名空间下
   - 它表示“整段 validation 阶段花了多久”，属于训练主迭代自己的 timing 信号。

---

## 5. 样本级信号：`episode_metrics.jsonl`

只有开启：

- `logging.sample_metrics.enabled: true`

才会写样本级信号。它们不是 `<module>/<metric>` 平铺形式，而是嵌套 JSON。

### 5.1 `contact`

| 字段名 | 含义 |
| --- | --- |
| `contact.t_cover_before` | 单个 episode refine 前的 `t_cover`。 |
| `contact.t_cover_after` | 单个 episode refine 后的 `t_cover`。 |
| `contact.t_edge_before` | 单个 episode refine 前的 `t_edge`。 |
| `contact.t_edge_after` | 单个 episode refine 后的 `t_edge`。 |

### 5.2 `calibrator`

| 字段名 | 含义 |
| --- | --- |
| `calibrator.raw_logit_before` | 单个 episode refine 前 raw stability logit。 |
| `calibrator.raw_logit_after` | 单个 episode refine 后 raw stability logit。 |
| `calibrator.prob_before` | 单个 episode refine 前 calibrated probability。 |
| `calibrator.prob_after` | 单个 episode refine 后 calibrated probability。 |
| `calibrator.posterior_trace_snapshot` | rollout 当下 calibrator snapshot 的 `tr(P)`。 |

### 5.3 `reward`

| 字段名 | 含义 |
| --- | --- |
| `reward.total` | 单个 episode 总 reward。 |
| `reward.drop` | 单个 episode 的 `R_drop`。 |
| `reward.stability` | 单个 episode 的 `R_stability`。 |
| `reward.contact` | 单个 episode 的 `R_contact`。 |

### 5.4 `outcome`

| 字段名 | 含义 |
| --- | --- |
| `outcome.drop_success_after_live` | 单个 episode 的 live release 结果。 |
| `outcome.legacy_drop_success_before` | 数据集自带的 before 标签。 |
| `outcome.trial_status` | 本条 episode 的 trial status。 |

### 5.5 `action`

| 字段名 | 含义 |
| --- | --- |
| `action.value` | 6 维归一化动作向量本身。 |
| `action.l2` | 动作向量的 L2 范数。 |
| `action.abs_mean` | 动作各维绝对值均值。 |
| `action.reward_total` | 与该动作对应的总 reward，便于把动作强度和结果直接对照。 |

---

## 6. 正式训练里最值得优先盯的信号

如果只看正式训练的核心健康度，优先级最高的一般是：

- `outcome/success_rate_live_after`
- `outcome/success_lift_vs_dataset`
- `outcome/drop_rate_after_given_dataset_positive`
- `outcome/hold_rate_after_given_dataset_negative`
- `reward/total_mean`
- `reward/drop_mean`
- `reward/stability_mean`
- `reward/contact_mean`
- `contact/t_cover_after_mean`
- `contact/t_edge_after_mean`
- `calibrator/prob_delta_mean`
- `calibrator/posterior_trace_post_update`
- `calibrator/after_brier`
- `ppo/approx_kl`
- `ppo/clip_fraction`
- `ppo/explained_variance`
- `collection/valid_rate`
- `timing/iteration_wall_s`

其中：

- `success_rate_live_after` 看最终任务结果
- `success_lift_vs_dataset` 看 refine 相对原始抓取是否真的有增益
- `t_cover/t_edge` 看策略是否在往论文想要的接触机制方向走
- `prob_delta_mean`、`posterior_trace_*`、`after_brier` 看稳定性信号链是否可信
- `approx_kl`、`clip_fraction`、`explained_variance` 看 PPO 是否在稳定更新
- `valid_rate` 看训练链路是否健康

这几组信号合在一起，基本就覆盖了 `v5.md` 里正式训练系统的四条主线：

- single-step grasp outcome
- contact semantic
- online calibrator
- PPO optimization
