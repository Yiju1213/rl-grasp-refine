# 五类模型训练与验证监控汇总

生成时间：2026-04-25

本文只整理当前监控数据中可以直接读出的结果、过程趋势和指标解释能力。涉及机制原因的判断不写成结论；后续可交由进一步推理模型在这些事实基础上继续分析。

## 1. 数据范围与口径

### 1.1 模型分组

| 简称 | 类型 | 原始路径 |
|---|---:|---|
| `normal_main` | 正常模型 | `/rl-grasp-refine/outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus` |
| `normal_dgcnn` | 正常模型 | `/rl-grasp-refine/outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_dgcnn` |
| `abnormal_cnnmca_allgeom` | 明显异常模型 | `/rl-grasp-refine/outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_latefus_cnnmca_allgeom` |
| `abnormal_cnnmca_table_camgeom` | 明显异常模型 | `/rl-grasp-refine/outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_latefus_cnnmca_table_camgeom` |
| `middle_cnnmca_table_allgeom` | 中间模型 | `/rl-grasp-refine/outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table_allgeom` |

### 1.2 统计口径

| 项目 | 口径 |
|---|---|
| 训练指标 | `metrics.jsonl` 中无 `validation/` 前缀的指标，每步训练采样 128 episodes。 |
| 验证指标 | `metrics.jsonl` 中带 `validation/` 前缀的指标，每 3 step 记录一次，验证采样 384 attempts。 |
| 过程趋势 | 表格中的 `0-99`、`100-199` 等为对应 step 区间均值；`450-499` 表示后期窗口。验证趋势在这些 step 区间内对验证点求均值。 |
| best val | 按 `validation/outcome/success_lift_vs_dataset` 最大值选出。 |
| 对比重点 | `success_lift_vs_dataset` 比 `success_rate_live_after` 更适合跨 run 比较，因为不同 run 的 `success_rate_dataset_before` 有小幅差异。 |

所有 5 个 run 均有 500 行训练记录，验证记录均为 167 行，最后训练 step 为 499，最后验证 step 为 498。

### 1.3 原始数据打包位置

本文所在目录已经包含轻量版原始监控数据副本。每个 run 仅保留 `metrics.jsonl`，因为本报告中的所有统计、趋势表和后续可脚本化分析都来自该结构化 JSONL。

| 简称 | 打包路径 | 内容 |
|---|---|---|
| `normal_main` | `raw/normal_main/` | `metrics.jsonl` |
| `normal_dgcnn` | `raw/normal_dgcnn/` | `metrics.jsonl` |
| `abnormal_cnnmca_allgeom` | `raw/abnormal_cnnmca_allgeom/` | `metrics.jsonl` |
| `abnormal_cnnmca_table_camgeom` | `raw/abnormal_cnnmca_table_camgeom/` | `metrics.jsonl` |
| `middle_cnnmca_table_allgeom` | `raw/middle_cnnmca_table_allgeom/` | `metrics.jsonl` |

`run.log` 与 `tensorboard/` 在标量监控上与 `metrics.jsonl` 高度冗余，因此不放入本报告包。checkpoint 也未复制到本报告目录；本次目标是监控数据分析，不包含模型权重归档。

### 1.4 CNNMCA 配置区别

本报告涉及 4 个 CNNMCA 相关训练：旧 `cnnmca`、异常 `cnnmca_allgeom`、异常 `cnnmca_table_camgeom`、中间 `cnnmca_table_allgeom`。这些 run 不是同一个单变量消融组，但配置差异可以从保存的 config snapshot 和 `run.log` 直接读出。

共同配置：

1. object split 相同：train objects 为 `[0, 74]`，holdout objects 为 `[75, 87]`，`split_seed=7`，validation objects 为 `[78, 82, 85, 87]`，test objects 为 `[75, 76, 77, 79, 80, 81, 83, 84, 86]`。
2. 训练长度相同：`num_iterations=500`。
3. 随机种子相同：`seed=7`。
4. ablation 都是 `baseline`。
5. reward 权重相同：`drop=1.0`、`stability=5.0`、`contact=1.0`。
6. PPO 配置相同：`batch_episodes=128`、`gamma=0.99`、`lam=0.95`、`learning_rate=0.0003`、`clip_range=0.2`、`entropy_coef=0.01`、`update_epochs=4`、`minibatch_size=16`、`num_envs=10`。
7. calibrator 配置相同：`init_a=1.0`、`init_b=0.0`、`online_update_enabled=true`、`lambda=1.0`。
8. perception 都使用 CNNMCA：`adapter_type=cnnmca`，feature extractor 冻结，latent dim 为 `512`，runtime checkpoint 为 `/AdaPoinTr/experiments/VTG_CNNMCA/3D_VTG_models/VTG2D_RGB_CNNMCA/ckpt-best.pth`。

核心差异：

| 模型 | 分组用途 | env config | table | actor-critic config | observation preset | policy components | aux_dim |
|---|---|---|---|---|---|---|---:|
| `paper-spec_latefus_cnnmca` | 旧 CNNMCA，对照附录使用 | `grasp_refine_env_stb5x.yaml` | 非 table env；旧 snapshot 无 table block | `actor_critic_latefus.yaml` | `paper` | `latent_feature`, `contact_semantic` | 2 |
| `latefus_cnnmca_allgeom` | 明显异常模型之一 | `grasp_refine_env_stb5x.yaml` | `table.enabled=false` | `actor_critic_latefus_allgeom.yaml` | `paper_allgeom` | `latent_feature`, `contact_semantic`, `action_axes_in_camera`, `hand_pose_in_camera`, `finger_geometry_in_camera` | 32 |
| `latefus_cnnmca_table_camgeom` | 明显异常模型之一 | `grasp_refine_env_stb5x_table.yaml` | `table.enabled=true` | `actor_critic_latefus_camgeom.yaml` | `paper_camgeom` | `latent_feature`, `contact_semantic`, `action_axes_in_camera`, `hand_pose_in_camera` | 23 |
| `paper-spec_latefus_cnnmca_table_allgeom` | 中间模型 | `grasp_refine_env_stb5x_table.yaml` | `table.enabled=true` | `actor_critic_latefus_allgeom.yaml` | `paper_allgeom` | `latent_feature`, `contact_semantic`, `action_axes_in_camera`, `hand_pose_in_camera`, `finger_geometry_in_camera` | 32 |

配置区别的直接含义：

1. `paper` 只给 policy/value 使用 `latent_feature + contact_semantic`。
2. `paper_camgeom` 在 `paper` 基础上加入 camera frame 下的 action axes 和 hand pose，但不加入 finger geometry。
3. `paper_allgeom` 在 `paper_camgeom` 基础上再加入 `finger_geometry_in_camera`。
4. `table_camgeom` 和 `table_allgeom` 同时改变了 observation preset 与 table env；它们不是只改变几何输入的单变量对照。
5. `cnnmca_allgeom` 与 `cnnmca_table_allgeom` 的 actor-critic 输入相同，主要差异是 table env 与 run 名/训练时间点。
6. 旧 `cnnmca` 与新 `table_allgeom` 同时差了 observation preset 和 table env，因此只能作为“不完全可消融”的整体配置对比，不能用于证明 `allgeom` 单独有效。

## 2. 可直接读出的总结果

以下结论均为当前监控表中可以直接读出的事实，不包含因果解释。

1. 两个正常模型的后期验证成功率和验证 lift 高于两个明显异常模型，也高于中间模型。后期验证 lift：`normal_main=0.0383`、`normal_dgcnn=0.0149`、`middle=-0.0812`、`abnormal_allgeom=-0.2598`、`abnormal_table_camgeom=-0.2943`。
2. 两个明显异常模型在训练集和验证集上都保持负 lift。后期训练 lift 分别为 `-0.2153`、`-0.1978`；后期验证 lift 分别为 `-0.2598`、`-0.2943`。
3. 中间模型的后期训练 lift 为正值 `0.0375`，但后期验证 lift 为负值 `-0.0812`。它的后期验证成功率 `0.5506` 介于正常模型和明显异常模型之间。
4. 中间模型的 best validation lift 为 `-0.0078 @ step 438`，接近 0；最后验证点为 `-0.1042 @ step 498`。
5. PPO 指标中，正常模型后期 `approx_kl` 和 `clip_fraction` 保持非零；两个明显异常模型与中间模型后期 `clip_fraction=0`，`approx_kl` 接近 0。
6. 两个明显异常模型后期 entropy 上升到约 `7.11-7.12`；中间模型后期 entropy 为 `4.57`；两个正常模型后期 entropy 分别为 `1.80` 和 `2.40`。
7. action 指标中，两个明显异常模型和中间模型的后期验证 action saturation 均远高于正常模型。后期验证 saturation：`normal_main=0.1431`、`normal_dgcnn=0.1651`、`abnormal_allgeom=0.6100`、`abnormal_table_camgeom=0.6090`、`middle=0.5826`。
8. calibrator 指标中，两个正常模型后期验证 `prob_delta_mean` 为正；`abnormal_cnnmca_table_camgeom` 为负；中间模型略负。后期验证值：`0.0445`、`0.0463`、`0.0027`、`-0.0368`、`-0.0028`。
9. calibrator AUC 中，`abnormal_cnnmca_allgeom` 的后期验证 `prob_after_auc=0.4518`，显著低于其它四类；`abnormal_cnnmca_table_camgeom` 的 `prob_after_auc=0.7368` 不低，但它的 `prob_delta_mean=-0.0368`。
10. 当前监控 key 中没有直接记录 calibrator 两参数 `a/b` 的数值；现有相关量为概率、logit 排序 AUC、Brier、posterior trace 和 delta 类指标。

## 3. Outcome 主结果

### 3.1 训练集 outcome 趋势

`outcome/success_rate_live_after`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.6950 | 0.7395 | 0.7498 | 0.7491 | 0.7435 | 0.7380 |
| `normal_dgcnn` | 0.5959 | 0.7114 | 0.7259 | 0.7310 | 0.7269 | 0.7267 |
| `abnormal_cnnmca_allgeom` | 0.3007 | 0.3184 | 0.3497 | 0.3766 | 0.3944 | 0.3883 |
| `abnormal_cnnmca_table_camgeom` | 0.3024 | 0.3284 | 0.3592 | 0.3827 | 0.3950 | 0.3950 |
| `middle_cnnmca_table_allgeom` | 0.6198 | 0.6269 | 0.6223 | 0.6237 | 0.6277 | 0.6280 |

`outcome/success_lift_vs_dataset`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.0896 | 0.1293 | 0.1488 | 0.1527 | 0.1416 | 0.1353 |
| `normal_dgcnn` | -0.0135 | 0.1130 | 0.1295 | 0.1272 | 0.1308 | 0.1330 |
| `abnormal_cnnmca_allgeom` | -0.3055 | -0.2792 | -0.2514 | -0.2256 | -0.2123 | -0.2153 |
| `abnormal_cnnmca_table_camgeom` | -0.3018 | -0.2715 | -0.2426 | -0.2198 | -0.2023 | -0.1978 |
| `middle_cnnmca_table_allgeom` | 0.0143 | 0.0284 | 0.0221 | 0.0222 | 0.0313 | 0.0375 |

后期训练 outcome 明细：

| 模型 | success_after | dataset_before | lift | positive_drop | negative_hold |
|---|---:|---:|---:|---:|---:|
| `normal_main` | 0.7380 | 0.6027 | 0.1353 | 0.1436 | 0.5580 |
| `normal_dgcnn` | 0.7267 | 0.5938 | 0.1330 | 0.1389 | 0.5329 |
| `abnormal_cnnmca_allgeom` | 0.3883 | 0.6036 | -0.2153 | 0.4906 | 0.2068 |
| `abnormal_cnnmca_table_camgeom` | 0.3950 | 0.5928 | -0.1978 | 0.4850 | 0.2202 |
| `middle_cnnmca_table_allgeom` | 0.6280 | 0.5905 | 0.0375 | 0.2542 | 0.4602 |

训练集直接事实：

- 两个正常模型的训练 lift 后期均约 `0.13`。
- 两个明显异常模型训练 lift 后期均约 `-0.20` 到 `-0.22`。
- 中间模型训练 lift 后期为正，但数值低于两个正常模型。
- 两个明显异常模型训练 `positive_drop` 接近 `0.49`；中间模型为 `0.2542`；两个正常模型约 `0.14`。
- 两个明显异常模型训练 `negative_hold` 约 `0.21-0.22`；中间模型为 `0.4602`；两个正常模型约 `0.53-0.56`。

### 3.2 验证集 outcome 趋势

`validation/outcome/success_rate_live_after`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.6406 | 0.6679 | 0.6682 | 0.6610 | 0.6794 | 0.6657 |
| `normal_dgcnn` | 0.5457 | 0.6425 | 0.6546 | 0.6642 | 0.6512 | 0.6454 |
| `abnormal_cnnmca_allgeom` | 0.2986 | 0.3243 | 0.3447 | 0.3578 | 0.3726 | 0.3724 |
| `abnormal_cnnmca_table_camgeom` | 0.2744 | 0.3011 | 0.3173 | 0.3299 | 0.3356 | 0.3364 |
| `middle_cnnmca_table_allgeom` | 0.5519 | 0.5534 | 0.5616 | 0.5569 | 0.5554 | 0.5506 |

`validation/outcome/success_lift_vs_dataset`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.0132 | 0.0375 | 0.0392 | 0.0306 | 0.0527 | 0.0383 |
| `normal_dgcnn` | -0.0849 | 0.0137 | 0.0277 | 0.0375 | 0.0236 | 0.0149 |
| `abnormal_cnnmca_allgeom` | -0.3312 | -0.3056 | -0.2857 | -0.2743 | -0.2588 | -0.2598 |
| `abnormal_cnnmca_table_camgeom` | -0.3503 | -0.3243 | -0.3100 | -0.2969 | -0.2940 | -0.2943 |
| `middle_cnnmca_table_allgeom` | -0.0789 | -0.0781 | -0.0707 | -0.0738 | -0.0756 | -0.0812 |

后期验证 outcome 明细：

| 模型 | success_after | dataset_before | lift | positive_drop | negative_hold |
|---|---:|---:|---:|---:|---:|
| `normal_main` | 0.6657 | 0.6275 | 0.0383 | 0.2192 | 0.4723 |
| `normal_dgcnn` | 0.6454 | 0.6305 | 0.0149 | 0.2319 | 0.4362 |
| `abnormal_cnnmca_allgeom` | 0.3724 | 0.6322 | -0.2598 | 0.5273 | 0.1997 |
| `abnormal_cnnmca_table_camgeom` | 0.3364 | 0.6307 | -0.2943 | 0.5712 | 0.1788 |
| `middle_cnnmca_table_allgeom` | 0.5506 | 0.6317 | -0.0812 | 0.3298 | 0.3453 |

验证集直接事实：

- 两个正常模型验证 lift 后期为正。
- 两个明显异常模型验证 lift 全程为负，后期绝对值大于 `0.25`。
- 中间模型验证 lift 全程为负，后期约 `-0.08`。
- 中间模型验证 success 后期为 `0.5506`，高于两个明显异常模型，低于两个正常模型。
- 中间模型验证 `positive_drop=0.3298`，高于两个正常模型，低于两个明显异常模型。
- 中间模型验证 `negative_hold=0.3453`，低于两个正常模型，高于两个明显异常模型。

### 3.3 Best validation 与最后 validation

| 模型 | best step | best lift | best success | best positive_drop | best negative_hold | best reward | best cal_delta | last step | last lift | last success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `normal_main` | 414 | 0.0911 | 0.7109 | 0.1513 | 0.4863 | 0.4135 | 0.0375 | 498 | 0.0286 | 0.6641 |
| `normal_dgcnn` | 363 | 0.0703 | 0.6927 | 0.1757 | 0.4759 | 0.3931 | 0.0456 | 498 | -0.0104 | 0.6276 |
| `abnormal_cnnmca_allgeom` | 366 | -0.2031 | 0.4271 | 0.4628 | 0.2394 | -0.2391 | 0.0037 | 498 | -0.2708 | 0.3568 |
| `abnormal_cnnmca_table_camgeom` | 444 | -0.2031 | 0.4167 | 0.4496 | 0.1986 | -0.3247 | -0.0272 | 498 | -0.2786 | 0.3490 |
| `middle_cnnmca_table_allgeom` | 438 | -0.0078 | 0.6224 | 0.2562 | 0.4155 | 0.1525 | -0.0028 | 498 | -0.1042 | 0.5286 |

直接事实：

- 正常模型 best lift 为正。
- 两个明显异常模型 best lift 仍为负，均为 `-0.2031`。
- 中间模型 best lift 为 `-0.0078`，在当前五个 run 中位于正常模型和明显异常模型之间。
- 中间模型最后验证 lift 低于它自己的 best lift。

## 4. PPO 训练监控

### 4.1 PPO 指标定义

| 指标 | 计算方式与含义 |
|---|---|
| `ppo/clip_fraction` | PPO 中新旧策略概率比 `r = pi_new(a|s) / pi_old(a|s)` 超出 `[1-eps, 1+eps]` 并触发 clipped surrogate 的样本比例。数值越高，表示越多样本的 policy update 被 clip 限制；数值为 0 表示该 batch 中没有样本触发 clip。 |
| `ppo/approx_kl` | 新旧策略之间 KL 距离的近似估计，常见实现为对 log-ratio 或 ratio 的 batch 估计。它表示一次 update 后 policy 分布变化量。 |
| `ppo/entropy` | policy 分布的平均 entropy。对离散 policy 是类别分布 entropy；对连续高斯 policy 通常是各 action 维度 entropy 求和或求均。数值高表示策略分布更分散，数值低表示更集中。 |
| `ppo/policy_loss` | PPO clipped objective 对应的 policy loss。 |
| `ppo/value_loss` | value function 的预测误差损失。 |
| `ppo/explained_variance` | value function 对 return/target 方差的解释比例。越接近 1 表示 value 拟合越好；接近 0 表示解释力弱；为负表示比常数预测更差。 |

### 4.2 PPO 过程趋势

`ppo/entropy`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 3.5673 | 3.0498 | 2.6038 | 2.1833 | 1.8659 | 1.8004 |
| `normal_dgcnn` | 3.9041 | 3.5228 | 3.1950 | 2.7102 | 2.4241 | 2.4013 |
| `abnormal_cnnmca_allgeom` | 4.0717 | 4.7549 | 5.4846 | 6.2505 | 6.9372 | 7.1209 |
| `abnormal_cnnmca_table_camgeom` | 4.2004 | 5.0748 | 5.7924 | 6.5001 | 6.9919 | 7.1108 |
| `middle_cnnmca_table_allgeom` | 3.7746 | 3.9860 | 4.1812 | 4.3858 | 4.5286 | 4.5729 |

`ppo/approx_kl`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.0691 | 0.0727 | 0.0799 | 0.0916 | 0.0984 | 0.0945 |
| `normal_dgcnn` | 0.1904 | 0.0919 | 0.0935 | 0.0802 | 0.0750 | 0.0697 |
| `abnormal_cnnmca_allgeom` | 0.2001 | 0.0020 | 0.0022 | 0.0025 | 0.0023 | 0.0025 |
| `abnormal_cnnmca_table_camgeom` | 0.2789 | 0.1026 | 0.0058 | 0.0018 | 0.0020 | 0.0018 |
| `middle_cnnmca_table_allgeom` | 0.3468 | 0.0012 | 0.0013 | 0.0011 | 0.0012 | 0.0013 |

`ppo/clip_fraction`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.4157 | 0.4025 | 0.4238 | 0.4479 | 0.4653 | 0.4620 |
| `normal_dgcnn` | 0.3960 | 0.4164 | 0.3930 | 0.2900 | 0.2970 | 0.3022 |
| `abnormal_cnnmca_allgeom` | 0.0119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `abnormal_cnnmca_table_camgeom` | 0.0371 | 0.0137 | 0.0006 | 0.0000 | 0.0000 | 0.0000 |
| `middle_cnnmca_table_allgeom` | 0.0311 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

后期 PPO 明细：

| 模型 | policy_loss | value_loss | entropy | approx_kl | clip_fraction | explained_variance |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | -0.0291 | 0.7540 | 1.8004 | 0.0945 | 0.4620 | 0.1074 |
| `normal_dgcnn` | -0.0045 | 0.7602 | 2.4013 | 0.0697 | 0.3022 | 0.1048 |
| `abnormal_cnnmca_allgeom` | -0.0005 | 0.9256 | 7.1209 | 0.0025 | 0.0000 | 0.0376 |
| `abnormal_cnnmca_table_camgeom` | -0.0003 | 0.9993 | 7.1108 | 0.0018 | 0.0000 | 0.0495 |
| `middle_cnnmca_table_allgeom` | -0.0004 | 0.9106 | 4.5729 | 0.0013 | 0.0000 | 0.0270 |

PPO 直接事实：

- 两个正常模型 entropy 随训练下降；两个明显异常模型 entropy 随训练上升；中间模型 entropy 随训练缓慢上升。
- 两个正常模型后期 `approx_kl` 保持在 `0.0697-0.0945`；两个明显异常模型和中间模型后期约 `0.0013-0.0025`。
- 两个正常模型后期 `clip_fraction` 非零；两个明显异常模型和中间模型后期为 0。
- 中间模型在 PPO 的 KL/clip 行为上与明显异常模型相同；在 entropy 数值上位于正常模型与明显异常模型之间。
- 五个模型的后期 `explained_variance` 均较低，正常模型约 `0.105`，异常和中间模型约 `0.027-0.050`。

## 5. Action 监控

### 5.1 Action 指标定义

| 指标 | 含义 |
|---|---|
| `action/abs_mean` | action 各维绝对值的均值，表示整体动作幅度。 |
| `action/l2_mean` | action 向量 L2 norm 的均值，表示整体动作能量或幅度。 |
| `action/saturation_rate` | action 各维接近边界或达到裁剪边界的比例，表示动作落在高幅度边界区间的频率。 |
| `action/translation_norm_mean` | 平移相关 action 维度的 norm 均值。 |
| `action/rotation_norm_mean` | 旋转相关 action 维度的 norm 均值。 |
| `action/dim_i_*` | 第 i 个 action 维度的均值、标准差和 saturation。 |
| `validation/action_bin/*` | 验证集中按 translation/rotation norm 分箱后的样本数、成功变化、概率变化和 drop rate。 |
| `validation/corr/*` | 验证集中 action 维度或 action norm 与 prob_delta、success_delta、positive_drop、negative_recovery 的相关系数。 |

### 5.2 Action saturation 过程趋势

训练 `action/saturation_rate`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.1532 | 0.1718 | 0.1765 | 0.1705 | 0.1473 | 0.1365 |
| `normal_dgcnn` | 0.4028 | 0.2528 | 0.2276 | 0.2011 | 0.1542 | 0.1586 |
| `abnormal_cnnmca_allgeom` | 0.5812 | 0.5803 | 0.5850 | 0.6011 | 0.6119 | 0.6127 |
| `abnormal_cnnmca_table_camgeom` | 0.5832 | 0.5786 | 0.5851 | 0.5983 | 0.6080 | 0.6121 |
| `middle_cnnmca_table_allgeom` | 0.5861 | 0.5840 | 0.5840 | 0.5814 | 0.5867 | 0.5865 |

验证 `validation/action/saturation_rate`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.1597 | 0.1794 | 0.1795 | 0.1768 | 0.1538 | 0.1431 |
| `normal_dgcnn` | 0.4004 | 0.2442 | 0.2227 | 0.1998 | 0.1585 | 0.1651 |
| `abnormal_cnnmca_allgeom` | 0.5843 | 0.5786 | 0.5867 | 0.6038 | 0.6082 | 0.6100 |
| `abnormal_cnnmca_table_camgeom` | 0.5816 | 0.5768 | 0.5865 | 0.6013 | 0.6079 | 0.6090 |
| `middle_cnnmca_table_allgeom` | 0.5840 | 0.5861 | 0.5846 | 0.5816 | 0.5827 | 0.5826 |

后期验证 action 幅度明细：

| 模型 | abs_mean | l2_mean | saturation | translation_norm | rotation_norm |
|---|---:|---:|---:|---:|---:|
| `normal_main` | 0.4894 | 1.3876 | 0.1431 | 1.0445 | 0.8660 |
| `normal_dgcnn` | 0.4727 | 1.3725 | 0.1651 | 1.0241 | 0.8686 |
| `abnormal_cnnmca_allgeom` | 0.8065 | 2.0846 | 0.6100 | 1.4772 | 1.4529 |
| `abnormal_cnnmca_table_camgeom` | 0.8069 | 2.0840 | 0.6090 | 1.4781 | 1.4494 |
| `middle_cnnmca_table_allgeom` | 0.8079 | 2.0771 | 0.5826 | 1.4591 | 1.4608 |

Action 直接事实：

- 两个明显异常模型和中间模型的后期 action 幅度接近：验证 `l2_mean` 约 `2.08`，`abs_mean` 约 `0.807`。
- 两个正常模型后期验证 `l2_mean` 约 `1.37-1.39`，低于异常和中间模型。
- 中间模型 action saturation、L2、translation norm、rotation norm 均接近两个明显异常模型。
- `normal_dgcnn` 的 saturation 从训练早期 `0.4028` 下降到后期 `0.1586`；`normal_main` 保持在较低范围；两个明显异常模型后期 saturation 上升到约 `0.61`。

### 5.3 Validation action bin 后期结果

下表为 step `450-499` 验证窗口均值。列顺序为：bin 样本数、success_delta_mean、prob_delta_mean、drop_rate。

| 模型 | 轴 | bin | count | success_delta | prob_delta | drop_rate |
|---|---|---:|---:|---:|---:|---:|
| `normal_main` | trans | 0 | 5.6 | -0.0480 | -0.0363 | 0.0539 |
| `normal_main` | trans | 1 | 87.5 | -0.0449 | -0.0022 | 0.1601 |
| `normal_main` | trans | 2 | 224.3 | 0.0555 | 0.0477 | 0.2383 |
| `normal_main` | trans | 3 | 66.6 | 0.0984 | 0.1030 | 0.3056 |
| `normal_main` | rot | 0 | 37.8 | 0.0158 | 0.0384 | 0.2116 |
| `normal_main` | rot | 1 | 147.8 | 0.0325 | 0.0404 | 0.2064 |
| `normal_main` | rot | 2 | 165.4 | 0.0398 | 0.0469 | 0.2364 |
| `normal_main` | rot | 3 | 33.0 | 0.0801 | 0.0582 | 0.2017 |
| `normal_dgcnn` | trans | 0 | 7.9 | -0.0657 | -0.0125 | 0.1028 |
| `normal_dgcnn` | trans | 1 | 75.4 | -0.0265 | 0.0078 | 0.1632 |
| `normal_dgcnn` | trans | 2 | 258.4 | 0.0133 | 0.0500 | 0.2599 |
| `normal_dgcnn` | trans | 3 | 42.4 | 0.1385 | 0.1165 | 0.2543 |
| `normal_dgcnn` | rot | 0 | 39.8 | 0.0064 | 0.0257 | 0.1954 |
| `normal_dgcnn` | rot | 1 | 147.6 | -0.0244 | 0.0294 | 0.2355 |
| `normal_dgcnn` | rot | 2 | 160.6 | 0.0305 | 0.0559 | 0.2394 |
| `normal_dgcnn` | rot | 3 | 36.1 | 0.0965 | 0.0966 | 0.2307 |
| `abnormal_cnnmca_allgeom` | trans | 0 | 0.5 | -0.1667 | 0.0104 | 0.3750 |
| `abnormal_cnnmca_allgeom` | trans | 1 | 4.5 | -0.0349 | 0.0053 | 0.2437 |
| `abnormal_cnnmca_allgeom` | trans | 2 | 70.2 | -0.2234 | 0.0022 | 0.4520 |
| `abnormal_cnnmca_allgeom` | trans | 3 | 308.8 | -0.2710 | 0.0027 | 0.5482 |
| `abnormal_cnnmca_allgeom` | rot | 0 | 0.4 | -0.3333 | 0.0073 | 0.6667 |
| `abnormal_cnnmca_allgeom` | rot | 1 | 4.9 | -0.2755 | 0.0015 | 0.4343 |
| `abnormal_cnnmca_allgeom` | rot | 2 | 80.8 | -0.2828 | 0.0023 | 0.5341 |
| `abnormal_cnnmca_allgeom` | rot | 3 | 297.9 | -0.2532 | 0.0028 | 0.5267 |
| `abnormal_cnnmca_table_camgeom` | trans | 0 | 0.6 | 0.1429 | -0.0354 | 0.0000 |
| `abnormal_cnnmca_table_camgeom` | trans | 1 | 5.1 | -0.1139 | -0.0342 | 0.3402 |
| `abnormal_cnnmca_table_camgeom` | trans | 2 | 67.8 | -0.2142 | -0.0321 | 0.4661 |
| `abnormal_cnnmca_table_camgeom` | trans | 3 | 310.5 | -0.3152 | -0.0381 | 0.5989 |
| `abnormal_cnnmca_table_camgeom` | rot | 0 | 0.4 | 0.0000 | -0.0651 | 0.3333 |
| `abnormal_cnnmca_table_camgeom` | rot | 1 | 8.1 | -0.2556 | -0.0355 | 0.5387 |
| `abnormal_cnnmca_table_camgeom` | rot | 2 | 78.6 | -0.2873 | -0.0381 | 0.5503 |
| `abnormal_cnnmca_table_camgeom` | rot | 3 | 296.8 | -0.2974 | -0.0365 | 0.5783 |
| `middle_cnnmca_table_allgeom` | trans | 0 | 0.0 | N/A | N/A | N/A |
| `middle_cnnmca_table_allgeom` | trans | 1 | 5.6 | -0.0608 | -0.0026 | 0.2346 |
| `middle_cnnmca_table_allgeom` | trans | 2 | 76.5 | -0.0465 | -0.0039 | 0.2701 |
| `middle_cnnmca_table_allgeom` | trans | 3 | 301.9 | -0.0899 | -0.0024 | 0.3456 |
| `middle_cnnmca_table_allgeom` | rot | 0 | 0.2 | 0.0000 | -0.0380 | 0.0000 |
| `middle_cnnmca_table_allgeom` | rot | 1 | 5.2 | 0.0198 | -0.0056 | 0.2256 |
| `middle_cnnmca_table_allgeom` | rot | 2 | 73.6 | -0.0572 | -0.0020 | 0.3248 |
| `middle_cnnmca_table_allgeom` | rot | 3 | 305.0 | -0.0876 | -0.0029 | 0.3312 |

Action bin 直接事实：

- 两个正常模型的验证样本主要集中在 trans bin 2，其次是 bin 1 或 bin 3；两个明显异常模型和中间模型主要集中在 trans bin 3。
- 两个正常模型 rot bin 1 和 bin 2 占比较大；两个明显异常模型和中间模型主要集中在 rot bin 3。
- 两个明显异常模型在高幅度 bin 中 success_delta 为负，drop_rate 高于正常模型。
- 中间模型的 high-bin 样本集中形态接近明显异常模型，但 high-bin drop_rate 低于两个明显异常模型。
- `abnormal_cnnmca_table_camgeom` 在各主要 trans/rot bin 的 `prob_delta_mean` 为负；`abnormal_cnnmca_allgeom` 的主要 bin `prob_delta_mean` 接近 0 且略正。

### 5.4 Validation action correlation 后期结果

下表为 step `450-499` 验证窗口均值。

| 模型 | corr(trans_norm, success_delta) | corr(rot_norm, success_delta) | corr(trans_norm, prob_delta) | corr(rot_norm, prob_delta) |
|---|---:|---:|---:|---:|
| `normal_main` | 0.0885 | 0.0264 | 0.3097 | 0.0368 |
| `normal_dgcnn` | 0.0718 | 0.0492 | 0.2652 | 0.1818 |
| `abnormal_cnnmca_allgeom` | -0.0602 | 0.0196 | 0.0084 | 0.0032 |
| `abnormal_cnnmca_table_camgeom` | -0.0879 | -0.0100 | -0.0303 | -0.0004 |
| `middle_cnnmca_table_allgeom` | -0.0409 | -0.0309 | 0.0082 | 0.0099 |

Corr 直接事实：

- 两个正常模型的 `translation_norm_prob_delta` 为正，且数值高于明显异常模型和中间模型。
- 两个正常模型的 `translation_norm_success_delta` 为正；两个明显异常模型和中间模型为负。
- 两个明显异常模型和中间模型的 norm-prob corr 接近 0 或略负。
- corr 只能描述当前验证样本上的线性相关方向和强度，不能单独给出因果关系或物体级归因。

## 6. Calibrator 监控

### 6.1 Calibrator 指标定义

| 指标 | 含义 |
|---|---|
| `prob_before_mean` | refinement 前 calibrator 预测成功概率均值。 |
| `prob_after_mean` | refinement 后 calibrator 预测成功概率均值。 |
| `prob_delta_mean` | `prob_after - prob_before` 的均值。正值表示平均预测概率上升，负值表示平均预测概率下降。 |
| `prob_delta_positive_rate` | `prob_delta > 0` 的样本比例。 |
| `after_brier` | refinement 后预测概率相对真实 binary outcome 的 Brier score，越低表示概率误差越小。 |
| `before_brier_vs_legacy` | refinement 前概率相对 legacy/dataset outcome 的 Brier score。 |
| `prob_after_auc` | 使用 refinement 后概率对真实成功/失败排序得到的 ROC AUC。 |
| `raw_logit_after_auc` | 使用 refinement 后 raw logit 对真实成功/失败排序得到的 ROC AUC。 |
| `prob_delta_recovery_auc` | 使用概率变化去排序“负样本恢复”等 recovery 目标得到的 AUC。 |
| `neg_prob_delta_degradation_auc` | 使用负的概率变化去排序“正样本退化”等 degradation 目标得到的 AUC。 |
| `posterior_trace_snapshot` / `posterior_trace_post_update` | 当前记录中可见的 posterior trace 相关量，反映 calibrator posterior 不确定性矩阵 trace 的快照和 update 后值。 |

当前监控没有直接记录 calibrator 两参数 `a/b` 的数值，也没有记录两参数的梯度、更新量或协方差分解。

### 6.2 Calibrator 过程趋势

训练 `calibrator/prob_delta_mean`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.0473 | 0.0624 | 0.0686 | 0.0673 | 0.0623 | 0.0606 |
| `normal_dgcnn` | 0.0270 | 0.0535 | 0.0550 | 0.0568 | 0.0558 | 0.0559 |
| `abnormal_cnnmca_allgeom` | 0.0098 | 0.0086 | 0.0046 | 0.0029 | 0.0013 | 0.0016 |
| `abnormal_cnnmca_table_camgeom` | -0.0427 | -0.0352 | -0.0297 | -0.0260 | -0.0236 | -0.0250 |
| `middle_cnnmca_table_allgeom` | 0.0019 | 0.0050 | 0.0049 | 0.0050 | 0.0057 | 0.0056 |

验证 `validation/calibrator/prob_delta_mean`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.0406 | 0.0499 | 0.0536 | 0.0516 | 0.0453 | 0.0445 |
| `normal_dgcnn` | 0.0203 | 0.0421 | 0.0436 | 0.0475 | 0.0465 | 0.0463 |
| `abnormal_cnnmca_allgeom` | 0.0170 | 0.0120 | 0.0057 | 0.0049 | 0.0013 | 0.0027 |
| `abnormal_cnnmca_table_camgeom` | -0.0474 | -0.0435 | -0.0398 | -0.0368 | -0.0349 | -0.0368 |
| `middle_cnnmca_table_allgeom` | -0.0031 | -0.0030 | -0.0022 | -0.0027 | -0.0027 | -0.0028 |

训练 `calibrator/prob_after_auc`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.7570 | 0.7407 | 0.7400 | 0.7393 | 0.7457 | 0.7486 |
| `normal_dgcnn` | 0.7473 | 0.7539 | 0.7539 | 0.7615 | 0.7718 | 0.7755 |
| `abnormal_cnnmca_allgeom` | 0.3916 | 0.3989 | 0.4353 | 0.4523 | 0.4834 | 0.4863 |
| `abnormal_cnnmca_table_camgeom` | 0.7474 | 0.7404 | 0.7296 | 0.7335 | 0.7283 | 0.7384 |
| `middle_cnnmca_table_allgeom` | 0.6751 | 0.6602 | 0.6629 | 0.6678 | 0.6556 | 0.6622 |

验证 `validation/calibrator/prob_after_auc`：

| 模型 | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 450-499 |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.7380 | 0.7532 | 0.7504 | 0.7403 | 0.7612 | 0.7561 |
| `normal_dgcnn` | 0.7398 | 0.7508 | 0.7591 | 0.7615 | 0.7748 | 0.7724 |
| `abnormal_cnnmca_allgeom` | 0.3522 | 0.3590 | 0.4401 | 0.4063 | 0.4774 | 0.4518 |
| `abnormal_cnnmca_table_camgeom` | 0.7503 | 0.7553 | 0.7513 | 0.7485 | 0.7345 | 0.7368 |
| `middle_cnnmca_table_allgeom` | 0.7204 | 0.7033 | 0.7167 | 0.7069 | 0.7040 | 0.7054 |

后期验证 calibrator 明细：

| 模型 | prob_delta_mean | prob_delta_positive_rate | after_brier | prob_after_auc | raw_logit_after_auc | recovery_auc | degradation_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.0445 | 0.6495 | 0.1922 | 0.7561 | 0.7561 | 0.6230 | 0.6232 |
| `normal_dgcnn` | 0.0463 | 0.6633 | 0.1917 | 0.7724 | 0.7724 | 0.5457 | 0.6339 |
| `abnormal_cnnmca_allgeom` | 0.0027 | 0.5616 | 0.2463 | 0.4518 | 0.6420 | 0.4536 | 0.4772 |
| `abnormal_cnnmca_table_camgeom` | -0.0368 | 0.3070 | 0.2168 | 0.7368 | 0.7368 | 0.6187 | 0.6198 |
| `middle_cnnmca_table_allgeom` | -0.0028 | 0.4740 | 0.2296 | 0.7054 | 0.7054 | 0.5999 | 0.6006 |

Calibrator 直接事实：

- 两个正常模型后期 train/val `prob_delta_mean` 均为正。
- `abnormal_cnnmca_allgeom` 后期 `prob_delta_mean` 接近 0，验证 `prob_after_auc=0.4518`。
- `abnormal_cnnmca_table_camgeom` 后期 train/val `prob_delta_mean` 均为负，但验证 `prob_after_auc=0.7368`。
- 中间模型后期训练 `prob_delta_mean` 为正 `0.0056`，验证为负 `-0.0028`。
- 中间模型验证 `prob_after_auc=0.7054`，低于两个正常模型和 `abnormal_cnnmca_table_camgeom`，高于 `abnormal_cnnmca_allgeom`。
- `abnormal_cnnmca_allgeom` 的 `prob_after_auc=0.4518` 与 `raw_logit_after_auc=0.6420` 不相等；其它四个模型两者相等。

## 7. Reward、Contact 与 Status 辅助监控

### 7.1 后期验证 reward

| 模型 | total | drop | stability | contact |
|---|---:|---:|---:|---:|
| `normal_main` | 0.3373 | 0.3315 | 0.0972 | -0.0914 |
| `normal_dgcnn` | 0.2979 | 0.2907 | 0.0998 | -0.0926 |
| `abnormal_cnnmca_allgeom` | -0.3506 | -0.2552 | 0.0055 | -0.1009 |
| `abnormal_cnnmca_table_camgeom` | -0.5056 | -0.3272 | -0.0764 | -0.1020 |
| `middle_cnnmca_table_allgeom` | 0.0070 | 0.1011 | -0.0057 | -0.0884 |

Reward 直接事实：

- 两个正常模型后期验证 total reward 为正，约 `0.30-0.34`。
- 两个明显异常模型后期验证 total reward 为负。
- 中间模型后期验证 total reward 接近 0，位于正常模型和明显异常模型之间。
- 中间模型后期验证 drop reward 为正 `0.1011`，低于正常模型，高于明显异常模型。
- 中间模型后期验证 stability reward 略负 `-0.0057`。

### 7.2 后期验证 contact

| 模型 | t_cover_after | t_cover_delta | t_edge_after | t_edge_delta |
|---|---:|---:|---:|---:|
| `normal_main` | 0.0466 | -0.0326 | 0.1519 | -0.0697 |
| `normal_dgcnn` | 0.0458 | -0.0342 | 0.1416 | -0.0810 |
| `abnormal_cnnmca_allgeom` | 0.0354 | -0.0431 | 0.1020 | -0.1197 |
| `abnormal_cnnmca_table_camgeom` | 0.0350 | -0.0454 | 0.0936 | -0.1317 |
| `middle_cnnmca_table_allgeom` | 0.0626 | -0.0171 | 0.1480 | -0.0750 |

Contact 直接事实：

- 中间模型后期验证 `t_cover_after=0.0626`，高于其它四个模型。
- 中间模型后期验证 `t_cover_delta=-0.0171`，绝对值小于其它四个模型。
- 两个明显异常模型后期验证 `t_edge_after` 低于两个正常模型和中间模型。
- 所有模型后期验证 `t_cover_delta` 和 `t_edge_delta` 均为负。

### 7.3 后期验证 trial status

| 模型 | contact_lost | effort_timeout | interference | pre_release_drop | release_drop | success |
|---|---:|---:|---:|---:|---:|---:|
| `normal_main` | 0.0030 | 0.0048 | 0.1255 | 0.0384 | 0.1653 | 0.6657 |
| `normal_dgcnn` | 0.0030 | 0.0055 | 0.1406 | 0.0320 | 0.1757 | 0.6454 |
| `abnormal_cnnmca_allgeom` | 0.0184 | 0.0522 | 0.1347 | 0.1103 | 0.3120 | 0.3724 |
| `abnormal_cnnmca_table_camgeom` | 0.0205 | 0.0515 | 0.1428 | 0.1029 | 0.3459 | 0.3364 |
| `middle_cnnmca_table_allgeom` | 0.0047 | 0.0138 | 0.1922 | 0.0467 | 0.1929 | 0.5506 |

Status 直接事实：

- 两个正常模型后期验证 release_drop 约 `0.165-0.176`。
- 两个明显异常模型后期验证 release_drop 约 `0.312-0.346`。
- 中间模型后期验证 release_drop 为 `0.1929`，高于正常模型，低于明显异常模型。
- 中间模型后期验证 interference 为 `0.1922`，高于其它四个模型。
- 两个明显异常模型后期验证 effort_timeout 约 `0.052`，高于正常模型和中间模型。

## 8. 当前监控量的解释能力判断

### 8.1 已经足以直接判别的内容

1. 是否在训练集和验证集上提升成功率：可以看 `success_lift_vs_dataset`、`success_rate_live_after`、`success_rate_dataset_before`。
2. 成功率下降主要表现在哪类 pair 结果上：可以看 `drop_rate_after_given_dataset_positive` 与 `hold_rate_after_given_dataset_negative`。
3. PPO update 是否活跃：可以看 `approx_kl`、`clip_fraction`、`policy_loss`、`entropy`。
4. 策略动作是否偏大或接近边界：可以看 `action/l2_mean`、`action/abs_mean`、`action/saturation_rate`、translation/rotation norm。
5. 高幅度动作与验证结果的分层关系：可以看 `validation/action_bin/*`。
6. 动作幅度与概率变化、成功变化的线性关系：可以看 `validation/corr/*`。
7. calibrator 的概率变化方向、排序能力和概率误差：可以看 `prob_delta_mean`、`prob_delta_positive_rate`、`prob_after_auc`、`raw_logit_after_auc`、`after_brier`。
8. 失败模式分布：可以看 trial status 的 `release_drop`、`pre_release_drop`、`interference`、`effort_timeout`、`contact_lost`。
9. reward 分项与 contact summary 的宏观趋势：可以看 `reward/*` 与 `contact/*`。

### 8.2 还不足以直接判别的内容

1. 当前监控不足以给出物体级别的直接结论。物体级别指标会在 formal test 中提供；本报告不把物体级差异写成结论。
2. 当前监控不足以直接记录 calibrator 两参数 `a/b` 的数值、梯度或更新幅度。
3. 当前监控不足以直接判断 CNNMCA 内部哪一路特征或哪一类几何输入导致了指标变化；没有内部激活、attention、feature norm 或分支贡献监控。
4. 当前监控不足以直接给出 episode-level 的完整 paired 轨迹表；目前 JSONL 是聚合统计。
5. 当前监控不足以直接分解 action saturation 的边界来源，例如 pre-tanh 分布、log_std、动作裁剪前后差异。
6. 当前监控不足以直接给出 PPO minibatch 内的分布形态，例如 ratio 分位数、KL 分位数、clip 正负方向比例。
7. 当前监控不足以直接说明 reward 各项与最终 formal test 成功率之间的物体级对应关系。

## 9. 稳定性判断与抓取成功率的优先指标

如果目标是从“稳定性判断 -> 抓取成功率”入手，建议优先看以下顺序：

1. 验证主结果：`validation/outcome/success_lift_vs_dataset`、`validation/outcome/success_rate_live_after`。
2. 验证 pair 结果：`validation/outcome/drop_rate_after_given_dataset_positive`、`validation/outcome/hold_rate_after_given_dataset_negative`。
3. 验证失败状态：`validation/outcome/trial_status_failure_release_drop_rate`、`pre_release_drop`、`interference`、`effort_timeout`。
4. PPO 稳定性：`ppo/approx_kl`、`ppo/clip_fraction`、`ppo/entropy`、`ppo/explained_variance`。
5. Action 稳定性：`validation/action/l2_mean`、`validation/action/saturation_rate`、translation/rotation norm。
6. Action 分层：`validation/action_bin/*_success_delta_mean`、`*_drop_rate`。
7. Calibrator 一致性：`validation/calibrator/prob_delta_mean`、`prob_delta_positive_rate`、`prob_after_auc`、`after_brier`。
8. Contact 辅助：`validation/contact/t_cover_delta_mean`、`t_edge_delta_mean`、`reward/stability_mean`。

在这五个模型中，上述指标呈现的直接事实是：正常模型在验证 lift、PPO KL/clip、action saturation、calibrator delta、release_drop 上形成一组相近数值；明显异常模型形成另一组相近数值；中间模型的 outcome、reward、status 多数位于两者之间，但 PPO KL/clip 和 action 幅度更接近明显异常模型。

## 10. 建议补充的指标

以下是建议新增的监控项。按用户要求，这里不建议新增 table 场景额外监控。

### 10.1 Formal test 中应使用的物体级指标

这部分用户已说明会在 formal test 中有，只是当前尚未使用；因此这里只注明建议接入，不作为当前训练监控缺失导致的结论。

1. per-object success before/after/lift。
2. per-object positive_drop 与 negative_hold。
3. per-object trial status 分布。
4. per-object action norm、saturation、bin 分布。
5. per-object calibrator prob_delta、AUC、Brier。
6. object category 或 shape/material/size 分组统计，如果 formal test 元数据支持。

### 10.2 Calibrator 参数级监控

1. 直接记录两参数 `a`、`b`。
2. 记录 `a/b` 的 update delta、gradient norm、clamp 状态。
3. 记录 posterior covariance 或 precision 的 trace、eigenvalues、condition number。
4. 记录 raw logit before/after 的均值、标准差和分位数。
5. 记录 probability calibration curve 或 ECE/MCE。

### 10.3 PPO 与 policy 分布监控

1. ratio 分位数：p01/p05/p50/p95/p99。
2. KL 分位数和 minibatch-level KL 均值/最大值。
3. clip 正方向比例与负方向比例。
4. action distribution 的 mean/std/log_std 分维记录。
5. pre-squash 或 pre-clip action 分布，区分 policy 输出大和环境 action clipping。
6. advantage mean/std、return mean/std、value target mean/std。
7. value prediction 与 return 的散点统计或分位误差。

### 10.4 Episode-level paired 明细

1. 每个 episode 的 before outcome、after outcome、success_delta。
2. 每个 episode 的 action 6 维、norm、saturation flag、bin。
3. 每个 episode 的 prob_before、prob_after、prob_delta、raw_logit_before/after。
4. 每个 episode 的 trial status、reward 分项、contact before/after。
5. 每个 episode 的 scene id、object id；object id 可与 formal test 物体级指标对齐。

### 10.5 分布与稳健性统计

1. success_lift、prob_delta、action norm 的 rolling mean 与 rolling std。
2. action norm、prob_delta、reward 的 p05/p25/p50/p75/p95。
3. 按 dataset_before positive/negative 分开的 action/action_bin/calibrator 统计。
4. validation 多 seed 或多固定 split 的均值和方差。

## 11. 给后续分析模型的注意事项

1. 本报告中的“正常、明显异常、中间”是用户指定分组，不是本文通过因果模型重新定义的标签。
2. `middle_cnnmca_table_allgeom` 同时具有正常和异常两侧的观测特征：outcome/reward/status 多数位于中间，PPO KL/clip 与 action 幅度更接近异常模型。
3. `abnormal_cnnmca_table_camgeom` 的 calibrator AUC 不低，但 prob_delta 为负；这两个事实应分别使用，不应把 AUC 直接等同于成功率提升。
4. `action_bin` 和 `corr` 能说明验证样本中动作幅度分层与结果的统计关系，但不能单独承担因果解释。
5. 当前缺少物体级 formal test 结果；不要把当前聚合训练/验证结果扩展成物体级结论。

## 12. 不完全可消融的 allgeom 对比

本节比较旧训练 `rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca` 与新训练 `rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table_allgeom`。旧训练监控较少，因此只能比较两者共同存在的指标。

### 12.1 对比边界

该对比不能严格说明 `allgeom` 单独有效，因为两者不是单变量消融：

1. 旧训练的 actor-critic observation preset 是 `paper`，即 `latent_feature + contact_semantic`。
2. 新训练的 actor-critic observation preset 是 `paper_allgeom`，即 `latent_feature + contact_semantic + action_axes_in_camera + hand_pose_in_camera + finger_geometry_in_camera`。
3. 新训练同时启用了 table env，配置中 `table.enabled: true`。
4. 旧训练只有 74 个监控 key，新训练有 188 个监控 key；action、action bin、corr、部分 calibrator AUC 只能在新训练中观察，不能直接与旧训练比较。

因此，这里只能说明“新 `table_allgeom` 整体配置相对旧 `cnnmca` 的共同指标差异”，不能把差异唯一归因于 `allgeom`。

### 12.2 共同指标对比

| 指标 | 旧 `cnnmca` | 新 `table_allgeom` | 直接读数 |
|---|---:|---:|---|
| train late lift, 450-499 | -0.0011 | 0.0375 | 新配置更高 |
| val late lift, 450-499 | -0.0964 | -0.0812 | 新配置更高，但仍为负 |
| val best lift | -0.0469 @ step 381 | -0.0078 @ step 438 | 新配置 best 更接近 0 |
| val late success | 0.5386 | 0.5506 | 新配置更高 |
| val best success | 0.5807 @ step 381 | 0.6224 @ step 438 | 新配置更高 |
| val late positive_drop | 0.3648 | 0.3298 | 新配置更低 |
| val late negative_hold | 0.3706 | 0.3453 | 新配置更低 |
| train late reward | 0.0751 | 0.1800 | 新配置更高 |
| val late reward | -0.0438 | 0.0070 | 新配置更高 |
| train late calibrator delta | -0.0076 | 0.0056 | 新配置更高 |
| val late calibrator delta | -0.0144 | -0.0028 | 新配置更接近 0 |
| train late entropy | 5.7166 | 4.5729 | 新配置更低 |
| train late approx_kl | 0.0015 | 0.0013 | 两者都接近 0 |
| train late clip_fraction | 0.0000 | 0.0000 | 两者都为 0 |

### 12.3 可直接写下的结论

1. 新 `table_allgeom` 在共同可比的训练 outcome 上高于旧 `cnnmca`：后期训练 lift 从 `-0.0011` 到 `0.0375`。
2. 新 `table_allgeom` 在共同可比的验证 outcome 上高于旧 `cnnmca`：后期验证 lift 从 `-0.0964` 到 `-0.0812`，best 验证 lift 从 `-0.0469` 到 `-0.0078`。
3. 新 `table_allgeom` 的验证 lift 仍为负，不能写成“已经稳定正提升”。
4. 新 `table_allgeom` 的 best 验证点优于旧 `cnnmca`，但最后验证点并不优于旧训练：旧 last val lift 为 `-0.0885`，新 last val lift 为 `-0.1042`。
5. 新 `table_allgeom` 后期验证 `positive_drop` 低于旧 `cnnmca`，但 `negative_hold` 也低于旧 `cnnmca`。
6. 新 `table_allgeom` 的 reward 与 calibrator delta 在共同指标上高于旧 `cnnmca`。
7. 两者后期 PPO `approx_kl` 都接近 0，`clip_fraction` 都为 0；共同指标显示两者后期 PPO update 都很弱。
8. 旧 `cnnmca` 缺少 action、action bin、corr 和部分 calibrator AUC 监控，因此不能通过旧/新对比直接判断 `allgeom` 是否改善了动作机制。

### 12.4 对 allgeom 的表述边界

可以表述为：

> 新 `table_allgeom` 整体配置相比旧 `cnnmca`，在共同可比的 train/val outcome、reward、calibrator delta 上有可观测改善；其中 best 验证 lift 更接近 0，但后期验证 lift 仍为负。

不应表述为：

> `allgeom` 已被证明单独有效。

若要把结论收敛到 `allgeom` 本身，需要补严格消融：

1. `cnnmca_table_no_allgeom` vs `cnnmca_table_allgeom`。
2. 或 `cnnmca_no_table_no_allgeom` vs `cnnmca_no_table_allgeom`。
3. 两组应保持 seed、object split、PPO、reward、calibrator、CNNMCA perception、训练步数和监控版本一致。
