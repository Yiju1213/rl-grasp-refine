# Config：当前真实训练链配置说明

本文档描述当前正式训练链从 `scripts/train.py` 启动时会读取和消费的配置项，并按执行顺序介绍每个 yaml 中的可调参数。

当前默认训练入口：

1. `scripts/train.py`
2. `configs/experiment/exp_debug.yaml`
3. `configs/env/grasp_refine_env.yaml`
4. `configs/perception/perception.yaml`
5. `configs/calibration/online_calibrator.yaml`
6. `configs/model/actor_critic.yaml`
7. `configs/rl/ppo.yaml`

## 总体说明

- 当前默认训练不是单环境串行训练。虽然环境类本身是单 env 实现，但 `configs/rl/ppo.yaml` 里当前默认 `num_envs: 12`，所以正式入口默认会走多进程异步 rollout 收集。
- `experiment.seed` 现在是唯一人工维护的 seed 事实源。`env.seed`、`dataset.seed`、`perception.sga_gsn.runtime.seed` 会在运行时由 `src/runtime/experiment_config.py` 自动同步注入，不再建议在子 yaml 里单独维护。
- 日志产物会自动带上 `experiment.name` 前缀。例如 `exp_debug.metrics.jsonl`、`exp_debug.run.log`、`exp_debug.tensorboard/`。
- 当前 `best.pt` 的保存标准是 rollout 级别的 `outcome/success_rate_live_after`。等以后补上独立 eval，再建议切到固定 eval 集上的 success rate。
- 仿真默认是非实时执行。`src/envs/pybullet_scene.py` 显式调用 `setRealTimeSimulation(0)`，所以训练时是“能多快 step 就多快”，不会和真实时钟对齐。

## 0. 真实训练链的配置流

### 顶层 experiment 配置

来源文件：

- `configs/experiment/exp_debug.yaml`

主要作用文件：

- `scripts/train.py`
- `scripts/_common.py`
- `src/runtime/experiment_config.py`
- `src/utils/logger.py`

职责：

- 指定实验名、主 seed、训练轮数
- 指定各个子配置文件路径
- 指定日志、TensorBoard、sample metrics、best checkpoint 的保存策略

### 环境配置

来源文件：

- `configs/env/grasp_refine_env.yaml`

主要作用文件：

- `src/runtime/builders.py`
- `src/envs/grasp_refine_env.py`
- `src/envs/pybullet_scene.py`
- `src/envs/action_executor.py`
- `src/envs/reward_manager.py`

职责：

- 控制 dataset-backed env、物理参数、scene 渲染、动作尺度、reward 形状

### 感知配置

来源文件：

- `configs/perception/perception.yaml`

主要作用文件：

- `src/perception/factory.py`
- `src/perception/contact_semantics.py`
- `src/perception/adapters.py`
- `src/perception/sga_gsn_runtime.py`

职责：

- 选择 perception adapter
- 配置 SGA-GSN runtime
- 定义 contact semantic 的上游阈值

### 校准器配置

来源文件：

- `configs/calibration/online_calibrator.yaml`

主要作用文件：

- `src/calibration/online_logit_calibrator.py`

职责：

- 定义在线 logit calibrator 的初值和正则强度

### 策略网络配置

来源文件：

- `configs/model/actor_critic.yaml`

主要作用文件：

- `src/runtime/builders.py`
- `src/models/rl/policy_network.py`
- `src/models/rl/value_network.py`
- `src/rl/observation_spec.py`

职责：

- 定义 actor / critic MLP 结构
- 定义 policy observation 的组成

### PPO / Trainer 配置

来源文件：

- `configs/rl/ppo.yaml`

主要作用文件：

- `scripts/train.py`
- `src/rl/trainer.py`
- `src/rl/ppo_agent.py`
- `src/rl/subproc_async_rollout_collector.py`

职责：

- 定义 rollout 规模、PPO 优化超参数、worker 数量和收集策略

## 1. 顶层实验配置：`configs/experiment/exp_debug.yaml`

### `name`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/runtime/experiment_config.py`、`src/utils/logger.py`
- 含义：实验名，同时也会作为日志文件和 TensorBoard 目录的前缀。
- 建议值/范围：建议使用稳定短名，例如 `exp_v5_main_seed7`。只用字母、数字、`-`、`_`、`.` 最稳妥。

### `seed`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/runtime/experiment_config.py`、`scripts/train.py`
- 含义：当前唯一的随机种子事实源，会被同步注入 env、dataset、SGA-GSN runtime。
- 建议值/范围：单次 baseline 可用 `7`；正式实验建议跑 `3-5` 个不同 seed。

### `num_iterations`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/train.py`、`src/rl/trainer.py`
- 含义：本次训练要执行多少轮 PPO iteration。
- 建议值/范围：调试 `5-20`，看趋势 `100-300`，正式训练 `300-1000`。

### `configs.env`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/_common.py`
- 含义：环境配置文件路径。
- 建议值/范围：正式训练使用 `configs/env/grasp_refine_env.yaml`；仅 debug fallback 使用 `configs/env/grasp_refine_env_debug_fallback.yaml`。

### `configs.perception`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/_common.py`
- 含义：perception 配置文件路径。
- 建议值/范围：当前正式训练使用 `configs/perception/perception.yaml`。

### `configs.calibration`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/_common.py`
- 含义：在线校准器配置文件路径。
- 建议值/范围：当前正式训练使用 `configs/calibration/online_calibrator.yaml`。

### `configs.rl`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/_common.py`
- 含义：PPO / Trainer 配置文件路径。
- 建议值/范围：当前正式训练使用 `configs/rl/ppo.yaml`。

### `configs.actor_critic`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/_common.py`
- 含义：策略网络和 value 网络配置文件路径。
- 建议值/范围：当前正式训练使用 `configs/model/actor_critic.yaml`。

### `logging.log_dir`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/utils/logger.py`
- 含义：日志根目录。`run.log`、`metrics.jsonl`、TensorBoard 目录都会从这里派生。
- 建议值/范围：建议放在本地 SSD，例如 `outputs/<experiment_name>`。

### `logging.checkpoint_dir`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/train.py`
- 含义：checkpoint 保存目录。当前会写 `final.pt`，也支持按规则写 `best.pt`。
- 建议值/范围：建议单独目录，例如 `outputs/<experiment_name>/checkpoints`。

### `logging.tensorboard.enabled`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/utils/logger.py`
- 含义：是否写 TensorBoard 事件文件。
- 建议值/范围：正式训练建议 `true`。

### `logging.tensorboard.dir`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/utils/logger.py`
- 含义：TensorBoard 目录。实际落盘时会自动带上 `experiment.name` 前缀。
- 建议值/范围：建议放在 experiment 输出目录下。

### `logging.sample_metrics.enabled`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/utils/logger.py`、`src/rl/trainer.py`
- 含义：是否输出 episode-sample 级 JSONL 调试记录。
- 建议值/范围：默认 `false`；只有排查训练异常时建议打开。

### `logging.sample_metrics.path`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/utils/logger.py`
- 含义：episode sample metrics 的 JSONL 路径。
- 建议值/范围：建议保留在 experiment 输出目录下。

### `logging.sample_metrics.every_n_iterations`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`src/utils/logger.py`
- 含义：每隔多少个 iteration 写一次 sample metrics。
- 建议值/范围：建议 `10-50`；更小会让调试日志变大。

### `logging.best_checkpoint.enabled`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/train.py`
- 含义：是否开启 best checkpoint 保存。
- 建议值/范围：正式训练建议 `true`。

### `logging.best_checkpoint.metric`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/train.py`
- 含义：用于判断 best checkpoint 的指标名。
- 建议值/范围：当前建议使用 `outcome/success_rate_live_after`；后续有 eval 后建议切到 `eval/success_rate`。

### `logging.best_checkpoint.mode`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/train.py`
- 含义：best metric 的比较方向，支持 `max` 或 `min`。
- 建议值/范围：当前 success rate 类型指标应使用 `max`。

### `logging.best_checkpoint.filename`

- 来源文件：`configs/experiment/exp_debug.yaml`
- 作用文件：`scripts/train.py`
- 含义：best checkpoint 的文件名。
- 建议值/范围：建议保持 `best.pt`。

## 2. 环境配置：`configs/env/grasp_refine_env.yaml`

### `max_reset_attempts`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/grasp_refine_env.py`
- 含义：环境 reset 失败时允许的最大重试次数。
- 建议值/范围：建议 `32`；常用范围 `16-64`。

### `dataset.enabled`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/runtime/builders.py`
- 含义：是否启用真实 dataset-backed 训练路径。
- 建议值/范围：正式训练必须 `true`。

### `dataset.dataset_root`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/dataset_sample_provider.py`
- 含义：真实数据集根目录。
- 建议值/范围：必须是有效路径，不属于数值调参项。

### `dataset.metadata_cache_size`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/dataset_sample_provider.py`
- 含义：对象 metadata 的 LRU cache 大小。
- 建议值/范围：建议 `4`；常用范围 `4-16`。

### `dataset.object_block_size`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/dataset_sample_provider.py`
- 含义：单个 worker 连续从同一个 object 采样多少个 grasp，再切到下一个 object。
- 建议值/范围：建议 `4`；常用范围 `2-8`。

### `dataset.runtime_defaults.time_step`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/pybullet_scene.py`
- 含义：仿真 time step。现在它是正式训练链里唯一的人类维护 time step 配置源。
- 建议值/范围：建议 `0.005`；常用范围 `0.0025-0.01`。越小通常越稳，但更慢。

### `dataset.runtime_defaults.close_timeout_s`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/pybullet_scene.py`
- 含义：夹爪闭合阶段建立接触的超时时间。
- 建议值/范围：建议 `1.0`；常用范围 `0.5-1.5`。

### `dataset.runtime_defaults.effort_timeout_s`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/pybullet_scene.py`
- 含义：达到目标夹持力的等待超时时间。
- 建议值/范围：建议 `1.0`；常用范围 `0.5-1.5`。

### `dataset.runtime_defaults.grip_force`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/pybullet_scene.py`
- 含义：release 之前的目标夹持力。
- 建议值/范围：建议 `30.0`；常用范围 `20-40`。

### `dataset.runtime_defaults.release_duration_s`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/pybullet_scene.py`
- 含义：after 阶段 release trial 的持续时间。
- 建议值/范围：建议 `1.5-2.0`；常用范围 `1.0-2.5`。

### `dataset.runtime_defaults.release_check_interval_steps`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/pybullet_scene.py`
- 含义：release trial 中每隔多少个仿真 step 检查一次掉落状态。
- 建议值/范围：建议 `10`；常用范围 `5-20`。

### `dataset.runtime_defaults.post_refine_settle_steps`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/pybullet_scene.py`
- 含义：执行 refine 动作后再静置多少步，之后再采集 after observation。
- 建议值/范围：建议 `8`；常用范围 `4-12`。

### `scene.use_gui`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/runtime/render_env.py`、`src/envs/pybullet_scene.py`
- 含义：是否启用 PyBullet GUI。
- 建议值/范围：正式训练建议 `false`；只在单环境可视化调试时设为 `true`。

### `scene.tacto_width`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/scene_assets.py`、`src/envs/scene_observation.py`
- 含义：tactile render 宽度。
- 建议值/范围：建议 `240`；追求速度时可试 `160`。

### `scene.tacto_height`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/scene_assets.py`、`src/envs/scene_observation.py`
- 含义：tactile render 高度。
- 建议值/范围：建议 `320`；追求速度时可试 `240`。

### `scene.visual_width`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/scene_observation.py`
- 含义：视觉渲染宽度。
- 建议值/范围：建议 `448`；追求速度时可试 `224-320`。

### `scene.visual_height`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/scene_observation.py`
- 含义：视觉渲染高度。
- 建议值/范围：建议 `448`；追求速度时可试 `224-320`。

### `scene.visual_near`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/scene_observation.py`
- 含义：视觉深度相机的 near plane。
- 建议值/范围：建议保持 `0.01`。

### `scene.visual_far`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/scene_observation.py`
- 含义：视觉深度相机的 far plane。
- 建议值/范围：建议保持 `2.0`。

### `scene.visualize_tacto_gui`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/scene_assets.py`
- 含义：是否打开 TACTO GUI 窗口。
- 建议值/范围：正式训练建议 `false`。

### `action.translation_bound`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/action_executor.py`
- 含义：policy 输出前 3 维动作映射到物理平移时的上下界，单位米。
- 建议值/范围：建议 `[0.01, 0.01, 0.01]`；单轴常用范围 `0.005-0.02`。

### `action.rotation_bound`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/action_executor.py`
- 含义：policy 输出后 3 维动作映射到 rotation vector 时的上下界，单位弧度。
- 建议值/范围：建议 `[0.1, 0.1, 0.1]`；单轴常用范围 `0.05-0.2`。

### `reward.stability_kappa`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/reward_manager.py`
- 含义：论文形式 `R_stability = Delta p / (1 + kappa * tr(P))` 中的不确定性抑制项。
- 建议值/范围：建议 `1.0`；常用范围 `0.5-2.0`。如果 calibrator 明显抖动，可试 `2.0-5.0`。

### `reward.contact_lambda_cover`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/reward_manager.py`
- 含义：`t_cover` 软阈值惩罚的权重。
- 建议值/范围：建议从 `0.1` 起步；常用范围 `0.1-0.5`。

### `reward.contact_lambda_edge`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/reward_manager.py`
- 含义：`t_edge` 软阈值惩罚的权重。
- 建议值/范围：建议从 `0.1` 起步；常用范围 `0.1-0.5`。

### `reward.contact_threshold_cover`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/reward_manager.py`
- 含义：`t_cover` 的 reward soft-hinge 阈值。它不定义接触本身，只定义从什么水平开始罚。
- 建议值/范围：建议 `0.2`；常用范围 `0.1-0.4`。

### `reward.contact_threshold_edge`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/reward_manager.py`
- 含义：`t_edge` 的 reward soft-hinge 阈值。它不定义接触本身，只定义从什么水平开始罚。
- 建议值/范围：建议 `0.2`；常用范围 `0.1-0.4`。

### `reward.drop_success_reward`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/reward_manager.py`
- 含义：release outcome 成功时的基础 reward。
- 建议值/范围：建议保持 `1.0`。

### `reward.drop_failure_reward`

- 来源文件：`configs/env/grasp_refine_env.yaml`
- 作用文件：`src/envs/reward_manager.py`
- 含义：release outcome 失败时的基础 reward。
- 建议值/范围：建议保持 `-1.0`。

### 附注：`contact_semantics.tactile_threshold` 和 `reward.contact_threshold_*` 的区别

- `contact_semantics.tactile_threshold` 位于感知上游，用于把 tactile map 中的数值判成 active contact 或 non-contact。
- `reward.contact_threshold_cover` / `reward.contact_threshold_edge` 位于 reward 下游，用于决定 `t_cover` / `t_edge` 低到什么程度时开始产生惩罚。
- 简单说：前者定义“如何测量 contact”，后者定义“如何用 contact 指标计算 reward”。

## 3. 感知配置：`configs/perception/perception.yaml`

### `adapter_type`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/factory.py`
- 含义：决定用哪种 perception adapter。当前支持 `sga_gsn` 和 `dgcnn`。
- 建议值/范围：正式训练建议固定为 `sga_gsn`。

### `feature_extractor.freeze`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/feature_extractor.py`、`src/perception/stability_predictor.py`
- 含义：fallback perception 路径是否在 `eval + no_grad` 下运行。
- 建议值/范围：建议保持 `true`。

### `backbone.latent_dim`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/factory.py`、`src/models/backbones/dgcnn_encoder.py`
- 含义：perception latent feature 维度。当前 `adapter_type=sga_gsn` 时，真实特征维数主要受 runtime 输出约束；这个字段更多是结构说明和 fallback 兼容。
- 建议值/范围：当前保持 `576`；如果将来重新启用 `dgcnn` 路径，常见范围可考虑 `32-128` 或按真实 encoder 输出再定。

### `backbone.hidden_dim`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/models/backbones/dgcnn_encoder.py`
- 含义：仅 `dgcnn` fallback 路径使用的隐藏维度。
- 建议值/范围：当前保持 `64`；若使用 fallback，常用范围 `64-256`。

### `predictor.hidden_dim`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/models/predictors/stability_head.py`
- 含义：仅 `dgcnn` fallback 路径使用的 predictor 隐藏维度。
- 建议值/范围：当前保持 `64`；若使用 fallback，常用范围 `64-256`。

### `contact_semantics.tactile_threshold`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/contact_semantics.py`
- 含义：上游 tactile 激活阈值。超过该值的 contact map 区域会被视作 active contact。
- 建议值/范围：建议 `0.2`；常用范围 `0.1-0.3`。

### `sga_gsn.runtime.source_root`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/sga_gsn_runtime.py`
- 含义：AdaPoinTr / SGSNet 源码根目录。
- 建议值/范围：必须是有效路径。

### `sga_gsn.runtime.device`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/sga_gsn_runtime.py`
- 含义：perception runtime 的设备。
- 建议值/范围：单卡训练建议设为 `cuda:0`，并与 `rl.device`、`worker_policy_device` 保持一致。

### `sga_gsn.runtime.config_path`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/sga_gsn_runtime.py`
- 含义：SGA-GSN / AdaPoinTr 的配置文件路径。
- 建议值/范围：必须是有效路径。

### `sga_gsn.runtime.shape_checkpoint`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/sga_gsn_runtime.py`
- 含义：shape completion checkpoint 路径。
- 建议值/范围：必须是有效路径。

### `sga_gsn.runtime.grasp_checkpoint`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/sga_gsn_runtime.py`
- 含义：grasp / stability 模型 checkpoint 路径。
- 建议值/范围：必须是有效路径。

### `sga_gsn.runtime.vis_points`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/adapters.py`
- 含义：视觉点云的采样点数。
- 建议值/范围：建议 `2048`；常用范围 `1024-4096`。

### `sga_gsn.runtime.tac_points_per_side`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/adapters.py`
- 含义：每一侧 tactile 世界点的采样数量。
- 建议值/范围：建议 `1200`；常用范围 `800-1600`。

### `sga_gsn.runtime.sc_input_points`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/adapters.py`
- 含义：shape completion 输入点数。
- 建议值/范围：建议 `2048`；常用范围 `1024-4096`。

### `sga_gsn.runtime.tactile_step`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/adapters.py`
- 含义：从 tactile depth 转世界点时的采样步长。
- 建议值/范围：建议 `8`；常用范围 `4-16`。越小越密、越慢。

### `sga_gsn.runtime.camera_distance_to_gel_m`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/adapters.py`
- 含义：tactile 相机到 gel 的几何距离参数。
- 建议值/范围：建议保持真实值 `0.02315`。

### `sga_gsn.runtime.tactile_noise_eps`

- 来源文件：`configs/perception/perception.yaml`
- 作用文件：`src/perception/adapters.py`
- 含义：tactile contact / gel mask 的容差。
- 建议值/范围：建议 `1e-4`；常用范围 `1e-5` 到 `5e-4`。

### 附注：`sensor_map` / `contact_map` 的语义

- 当前正式主环境里，contact semantic 上游优先读取的是 `raw_obs.tactile_data["contact_map"]`。
- 这个 `contact_map` 由 tactile depth 归一化得到，而不是原始米制 depth 直接输入。
- 归一化过程会把 tactile depth 映射到 `[0, 1]`，再由 `contact_semantics.tactile_threshold` 判定 active contact。

## 4. 在线校准器配置：`configs/calibration/online_calibrator.yaml`

### `init_a`

- 来源文件：`configs/calibration/online_calibrator.yaml`
- 作用文件：`src/calibration/online_logit_calibrator.py`
- 含义：校准器中 logit 线性变换的初始斜率，形式为 `a * logit + b`。
- 建议值/范围：建议保持 `1.0`。

### `init_b`

- 来源文件：`configs/calibration/online_calibrator.yaml`
- 作用文件：`src/calibration/online_logit_calibrator.py`
- 含义：校准器中 logit 线性变换的初始偏置。
- 建议值/范围：建议保持 `0.0`。

### `lambda`

- 来源文件：`configs/calibration/online_calibrator.yaml`
- 作用文件：`src/calibration/online_logit_calibrator.py`
- 含义：logistic calibration 的 ridge / damping 系数，也会影响 posterior covariance 和 `tr(P)`。
- 建议值/范围：建议 `1.0`；常用范围 `0.5-2.0`，更宽可试 `0.1-5.0`。

## 5. 策略网络配置：`configs/model/actor_critic.yaml`

### `policy_hidden_dims`

- 来源文件：`configs/model/actor_critic.yaml`
- 作用文件：`src/models/rl/policy_network.py`
- 含义：policy MLP 的隐藏层宽度列表。
- 建议值/范围：建议 `[128, 128]`；常用范围 `[64, 64]` 到 `[256, 256]`。

### `value_hidden_dims`

- 来源文件：`configs/model/actor_critic.yaml`
- 作用文件：`src/models/rl/value_network.py`
- 含义：value MLP 的隐藏层宽度列表。
- 建议值/范围：建议 `[128, 128]`；常用范围 `[64, 64]` 到 `[256, 256]`。

### `initial_log_std`

- 来源文件：`configs/model/actor_critic.yaml`
- 作用文件：`src/models/rl/policy_network.py`、`src/models/rl/actor_critic.py`
- 含义：policy 高斯分布的初始 `log_std`，决定初始探索强度。
- 建议值/范围：建议 `-0.5`；常用范围 `-1.0` 到 `0.0`。

### `policy_observation.preset`

- 来源文件：`configs/model/actor_critic.yaml`
- 作用文件：`src/rl/observation_spec.py`
- 含义：用预定义套餐选择 policy observation 的组成。
- 建议值/范围：
  - `current`：当前工程 baseline，包含 `latent_feature + contact_semantic + grasp_position + grasp_rotation + raw_stability_logit`
  - `paper`：更贴近论文主表达，只包含 `latent_feature + contact_semantic`
  - `no_pose`：去掉 grasp pose
  - `no_logit`：去掉 raw stability logit

### `policy_observation.components`

- 来源文件：`configs/model/actor_critic.yaml`
- 作用文件：`src/rl/observation_spec.py`
- 含义：手工指定 observation 组件集合。只要它不为 `null`，就会覆盖 `preset`，并按框架约定顺序解析。
- 建议值/范围：常规训练建议保持 `null`；做精细 ablation 时再显式指定。
- 支持的组件：
  - `latent_feature`
  - `contact_semantic`
  - `grasp_position`
  - `grasp_rotation`
  - `raw_stability_logit`

### 附注：`preset` 与 `components` 的区别

- `preset` 是命名好的固定套餐，适合日常切换。
- `components` 是手工拼 observation，用于精细 ablation。
- 优先级上，`components` 高于 `preset`。一旦 `components` 非空，`preset` 只起文档作用。

## 6. PPO / Trainer 配置：`configs/rl/ppo.yaml`

### `device`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`scripts/train.py`、`src/rl/ppo_agent.py`
- 含义：主进程 learner 的设备。
- 建议值/范围：单卡建议 `cuda:0`。

### `worker_policy_device`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/subproc_async_rollout_collector.py`
- 含义：worker 里 policy / value 前向推理的设备。
- 建议值/范围：单卡建议与 `device` 保持一致，即 `cuda:0`。

### `batch_episodes`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/trainer.py`
- 含义：每个 iteration 先从环境里收集多少个新的有效 episode。当前是 single-step 任务，因此也可近似理解为“每轮新鲜 on-policy 样本数”。
- 建议值/范围：建议 `32`；常用范围 `16-64`。

### `gamma`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/trainer.py`、`src/rl/advantage.py`
- 含义：折扣因子。
- 建议值/范围：当前 single-step 设定下不敏感，建议保持 `0.99`。

### `lam`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/trainer.py`、`src/rl/advantage.py`
- 含义：GAE 的 lambda。
- 建议值/范围：当前 single-step 设定下不敏感，建议保持 `0.95`。

### `learning_rate`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`scripts/train.py`
- 含义：Adam 学习率。
- 建议值/范围：建议 `3e-4`；常用范围 `1e-4` 到 `5e-4`。

### `clip_range`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/ppo_agent.py`
- 含义：PPO clipped objective 的截断范围。
- 建议值/范围：建议 `0.2`；常用范围 `0.15-0.25`。

### `value_loss_coef`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/ppo_agent.py`
- 含义：value loss 的权重。
- 建议值/范围：建议 `0.5`；常用范围 `0.25-1.0`。

### `entropy_coef`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/ppo_agent.py`
- 含义：entropy bonus 的权重，越大通常探索越强。
- 建议值/范围：建议 `0.01`；常用范围 `0.003-0.02`。

### `update_epochs`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/ppo_agent.py`
- 含义：对同一批刚采回来的 rollout 数据重复训练多少遍。
- 建议值/范围：建议 `4`；常用范围 `3-6`。

### `minibatch_size`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/ppo_agent.py`
- 含义：每次参数更新时，一个 mini-batch 里吃多少条样本。如果超过当前 batch size，代码会自动截到 batch size。
- 建议值/范围：建议 `16`；常用范围 `8-32`。

### `max_grad_norm`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/ppo_agent.py`
- 含义：梯度裁剪上限。
- 建议值/范围：建议 `0.5`；常用范围 `0.3-1.0`。

### `normalize_advantages`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/ppo_agent.py`
- 含义：是否对 advantage 做标准化。
- 建议值/范围：建议保持 `true`。

### `max_collect_attempt_factor`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`src/rl/trainer.py`、`src/rl/subproc_async_rollout_collector.py`
- 含义：为了拿到足够多 valid episode，最多允许尝试多少倍的无效 episode。
- 建议值/范围：建议 `10`；常用范围 `5-15`。

### `num_envs`

- 来源文件：`configs/rl/ppo.yaml`
- 作用文件：`scripts/train.py`、`src/rl/subproc_async_rollout_collector.py`
- 含义：worker 数量。`>1` 时会启用异步多进程 rollout collector。
- 建议值/范围：当前默认是 `12`。常见范围依赖机器资源，通常 `2-12`。显存、CPU、PyBullet 和 SGA-GSN 资源都足够时再拉高。

### 附注：`batch_episodes`、`update_epochs`、`minibatch_size` 在 PPO 里的含义

- `batch_episodes`：这一轮先从环境里采多少条新的 on-policy 样本。
- `update_epochs`：对这批新样本重复训练多少遍。
- `minibatch_size`：每一遍训练时，每次梯度更新吃多少样本。

例如：

- `batch_episodes=32`
- `update_epochs=4`
- `minibatch_size=16`

这表示：

1. 先采 32 个新的 episode
2. 再拿这 32 个样本做 4 遍 PPO 更新
3. 每一遍分成 2 个 mini-batch
4. 总共执行 8 次 optimizer step

这符合 PPO 的常见训练方式。原因是 rollout 成本高，PPO 会在“刚采回来的、仍然近似 on-policy 的数据”上做有限次重复优化，以提升样本利用率；而 `clip_range` 就是控制“重复训练但不要偏离采样策略太远”的关键约束。

## 7. Debug-only fallback 环境配置：`configs/env/grasp_refine_env_debug_fallback.yaml`

这个文件不是正式训练主配置，只给不接真实 dataset 的 toy / debug 路径使用。

### `dataset.enabled`

- 来源文件：`configs/env/grasp_refine_env_debug_fallback.yaml`
- 作用文件：`src/runtime/builders.py`
- 含义：关闭 dataset-backed 路径，改为 fallback sample。
- 建议值/范围：这个文件里应保持 `false`。

### `sampling.position_noise`

- 来源文件：`configs/env/grasp_refine_env_debug_fallback.yaml`
- 作用文件：`src/envs/grasp_refine_env.py`
- 含义：fallback 初始抓取位置扰动。
- 建议值/范围：当前默认 `[0.015, 0.015, 0.015]`；如果只是轻量 debug，可再减小。

### `sampling.rotation_noise`

- 来源文件：`configs/env/grasp_refine_env_debug_fallback.yaml`
- 作用文件：`src/envs/grasp_refine_env.py`
- 含义：fallback 初始抓取姿态扰动。
- 建议值/范围：当前默认 `[0.12, 0.12, 0.12]`；如果只是轻量 debug，可再减小。

### `default_sample_cfg.target_grasp_pose.position`

- 来源文件：`configs/env/grasp_refine_env_debug_fallback.yaml`
- 作用文件：`src/envs/grasp_refine_env.py`
- 含义：fallback 目标抓取位姿的位置部分。
- 建议值/范围：只在 toy 场景调试时使用。

### `default_sample_cfg.target_grasp_pose.rotation`

- 来源文件：`configs/env/grasp_refine_env_debug_fallback.yaml`
- 作用文件：`src/envs/grasp_refine_env.py`
- 含义：fallback 目标抓取位姿的旋转部分。
- 建议值/范围：只在 toy 场景调试时使用。

## 8. 调参优先级建议

如果你的目标是尽快把正式训练跑稳，优先关注这几组参数：

1. PPO 主参数：`num_envs`、`batch_episodes`、`learning_rate`、`clip_range`、`entropy_coef`
2. 动作尺度：`action.translation_bound`、`action.rotation_bound`
3. 物理运行参数：`grip_force`、`close_timeout_s`、`effort_timeout_s`、`release_duration_s`
4. reward 形状：`stability_kappa`、`contact_lambda_cover`、`contact_lambda_edge`
5. 感知成本：`vis_points`、`tac_points_per_side`、`sc_input_points`、`tactile_step`

相对低优先级的参数包括：

- `gamma`
- `lam`
- `feature_extractor.freeze`
- `backbone.hidden_dim`
- `predictor.hidden_dim`

## 9. 一组推荐 baseline

如果你想直接从相对稳妥的 baseline 开始，建议优先保持：

- `experiment.seed: 7`
- `dataset.runtime_defaults.time_step: 0.005`
- `dataset.runtime_defaults.grip_force: 30.0`
- `dataset.runtime_defaults.release_duration_s: 1.5-2.0`
- `contact_semantics.tactile_threshold: 0.2`
- `reward.stability_kappa: 1.0`
- `reward.contact_lambda_cover: 0.1`
- `reward.contact_lambda_edge: 0.1`
- `policy_hidden_dims: [128, 128]`
- `value_hidden_dims: [128, 128]`
- `initial_log_std: -0.5`
- `policy_observation.preset: current`
- `batch_episodes: 32`
- `learning_rate: 3e-4`
- `clip_range: 0.2`
- `entropy_coef: 0.01`
- `update_epochs: 4`
- `minibatch_size: 16`

如果资源足够，可从当前默认的 `num_envs: 12` 启动；如果只是先做稳定性验证，建议先降到 `2-4`，更容易定位问题。
