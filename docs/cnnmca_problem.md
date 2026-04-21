下面按“事实”和“推断”分开整理，方便你直接给 GPT Thinking 继续讨论。

**一、CNNMCA 相较 SGA-GSN 在 RL 训练链条中的异同**

相同部分：

- 二者都作为 frozen perception module 使用，不在 RL 训练中反向更新。
- 二者最终都向 RL 环境输出同一类语义接口：
  - `latent_feature`
  - `raw_stability_logit`
- 后续链条一致：
  - online calibrator 接收 raw logit，输出 calibrated probability；
  - reward 仍由 drop reward、stability reward、contact reward 组成；
  - PPO actor-critic 仍执行 single-step 6D refinement；
  - env reset / step / release trial / formal eval 机制保持一致。
- CNNMCA 实验不是 fake env，也不是离线评估；它走真实 PyBullet + TACTO + live observation 链条。

不同部分：

- SGA-GSN runtime 来自 `/AdaPoinTr` 的 SGA-GSN / PPCT 路径，主要消费 3D / visuo-tactile 几何重建相关输入，输出 latent dim 为 `576`。
- CNNMCA runtime 来自 `/AdaPoinTr/experiments/VTG_CNNMCA/3D_VTG_models/VTG2D_RGB_CNNMCA`，消费 visual RGB 与 tactile RGB，输出 latent dim 为 `512`。
- 因 latent dim 不同，CNNMCA 不能复用 SGA-GSN actor checkpoint，必须从头训练 actor-critic。
- CNNMCA 的 actor 输入虽然仍是 late-fusion 形式，但 latent 分布来自 CNN 图像特征；SGA-GSN 的 latent 更偏几何/重建特征。
- 因 CNNMCA 强依赖 RGB visual branch，它更容易受到视觉域差异影响，例如 table 背景、光照、纹理、相机视角、render 风格。
- SGA-GSN full / paper-spec 已经能在 RL 中学出正向收益；CNNMCA no-table 训练完成后 formal test 接近 rand action，说明其 RL 可利用性明显弱于 SGA-GSN。

**二、CNNMCA 在 DL 与 RL 中接收图像及处理的异同**

DL 训练 / 验证中的 CNNMCA：

- 数据来自硬盘中的 tactile-extended dataset。
- visual input 使用 `vis_rgb/<global_id>.png`。
- tactile input 使用 `tac_rgb/<global_id>_l.png` 和 `_r.png`。
- tactile 左右图像会横向拼接后输入。
- visual segmentation `vis_seg` 存在于 dataset，但 CNNMCA runner 实际 forward 时主要传入 `visual_rgb` 和 `tactile_rgb_concat`，不是直接消费 segmentation。
- val/test transform 口径是：
  - resize 到 `224x224`
  - ToTensor
  - ImageNet normalize
- train transform 还额外有随机 crop、flip、rotation、color jitter、blur 等 augmentation。

RL 中的 CNNMCA：

- visual input 不读 dataset 硬盘图，而是来自当前 PyBullet scene 的 live camera render。
- tactile input 不读 dataset 硬盘图，而是来自当前 TACTO sensor live render。
- CNNMCAAdapter 做的处理与 DL val/test 基本对齐：
  - visual RGB resize 到 `224x224`
  - tactile left/right RGB 横向拼接后 resize 到 `224x224`
  - CHW float tensor
  - ImageNet normalize
- 所以“拿到图像之后”的处理方式基本一致。
- 关键差异在“图像来源”：
  - DL 是 dataset saved image；
  - RL 是 live reconstruction / live render image。
- RL visual camera 是当前 env 中的 hand-relative camera；before/after view matrix 会跟随 hand pose 改变。
- RL 中 no-table 时，visual RGB 的背景与 dataset 中带桌面纹理的视觉图不一致。
- RL 中 table enabled 时，视觉背景更像 dataset，但也会引入真实 PyBullet collision body，不只是视觉背景变化。

可以给 GPT Thinking 的一句总结：

> CNNMCA 在 DL 与 RL 中的 image preprocessing 基本一致，主要 domain gap 不在 resize/normalize，而在 image source：offline dataset render vs online PyBullet/TACTO live render。

**三、CNNMCA no-table 的 RL 表现**

CNNMCA no-table 训练事实：

- 训练能完整跑完 500 iter。
- 它不是系统崩溃型问题，`valid_rate=1.0`。
- 但学习效果弱，formal test 已接近 rand action。
- 训练日志显示它长期没有形成明显正向 stability / outcome 改善。
- final 训练末尾仍有：
  - `success_lift_vs_dataset ≈ -0.0625`
  - `prob_delta_mean ≈ -0.0128`
  - `t_edge_delta_mean ≈ -0.0587`
  - `reward/total_mean ≈ 0.051`
- 相较之下，SGA-GSN paper-spec final 有明显正向：
  - `success_lift_vs_dataset ≈ 0.1328`
  - `prob_delta_mean ≈ 0.0715`
  - `reward/total_mean ≈ 0.504`

推断：

- CNNMCA 本身有抓取稳定性判断能力，但该能力在当前 RL live observation 分布下没有很好转化为可学习的 policy signal。
- CNNMCA 的 raw logit / latent 可能在 dataset validation 上有效，但在 live PyBullet render + TACTO render 下分布偏移较大。
- online calibrator 能修正概率尺度，但不能保证 latent 对 PPO action selection 有足够可利用结构。

**四、CNNMCA + table 的训练反常情况**

CNNMCA+table 训练事实：

- table 版不是后期才学坏，而是第 0 次 validation 就已经很差。
- `cnnmca_table` 前 54 iter 平均：
  - `success_lift_vs_dataset = -0.293`
  - `success_rate_live_after = 0.316`
  - `failure_release_drop_rate = 0.488`
  - `failure_pre_release_drop_rate = 0.100`
  - `prob_delta_mean = -0.044`
  - `t_cover_delta_mean = -0.046`
  - `t_edge_delta_mean = -0.114`
  - `reward/total_mean = -0.562`
- CNNMCA no-table 前 54 iter 平均：
  - `success_lift_vs_dataset = -0.046`
  - `success_rate_live_after = 0.561`
  - `failure_release_drop_rate = 0.265`
  - `failure_pre_release_drop_rate = 0.044`
  - `prob_delta_mean = -0.007`
  - `t_cover_delta_mean = -0.008`
  - `t_edge_delta_mean = -0.060`
  - `reward/total_mean = 0.017`

解释：

- table 不是只改变 CNNMCA 的视觉背景；当前实现是 scene-level PyBullet static body，有 collision。
- 但后续 zero/random formal 诊断显示，table 并没有普遍破坏物理环境底盘。
- 因此 `cnnmca_table` 的崩坏更可能来自：
  - table 改变 visual RGB 分布；
  - CNNMCA latent/logit 分布随之改变；
  - learned actor 在该 latent 分布下输出了系统性坏动作；
  - 坏动作导致 release drop / pre-release drop 大幅上升；
  - PPO 从一开始就在差分布上学习，难以恢复。

**五、有无 table 的 fixed-policy 诊断结果**

我把 table diag 与现有 formal 的 no-action / rand-action 做了直接对比。

`table - no-table` 的 summary 差值：

| Policy | Lift Δ | Pos Drop Δ | Neg Hold Δ | T-cover Δ | T-edge Δ | Prob Δ |
|---|---:|---:|---:|---:|---:|---:|
| no-action | -0.0005 | -0.0000 | -0.0023 | +0.0001 | +0.0005 | -0.0003 |
| rand-action | -0.0005 | +0.0001 | +0.0010 | +0.0003 | +0.0003 | -0.0127 |

结论：

- no-action 下，table 与 no-table 几乎完全一致。
- rand-action 下，table 与 no-table 也几乎完全一致。
- table 没有显著拉低 fixed-policy 的 live success / success lift / pos drop / neg hold。
- 只有 rand-action 的 `prob_delta_mean` 明显更负，说明 CNNMCA 概率信号对 table 视觉分布有敏感性。
- 这反驳了“table 物理协议整体破坏 formal eval”的解释。

**六、当前最合理的综合判断**

最合理的链条是：

> CNNMCA 在当前 RL live rendering 中本来就可学习性弱；table 让 visual branch 分布进一步变化。这个变化没有让 zero/random 的物理结果变差，但会改变 CNNMCA latent，从而改变 learned actor 的动作分布。learned actor 在 CNNMCA+table latent 下产生系统性坏 refinement，导致训练从一开始 release drop 高、reward 负、prob/contact delta 负，最终表现为“训练炸”。

更短的版本：

> table 不是直接破坏环境，而是通过 CNNMCA visual latent 改变 learned policy 的动作分布；fixed zero/random 不依赖 latent，所以它们不受明显影响。

**七、后续最值得验证的问题**

- 比较 `cnnmca` 与 `cnnmca_table` 在同一批样本上的 actor normalized action 分布。
- 比较 table on/off 下 CNNMCA latent norm、raw logit、calibrated probability 分布。
- 用 `visualize_full_inference.py --enable-table / --disable-table` 对同一 sample 输出 before/after 图和 action，检查是否出现系统性 action 偏置。
- 如果想保留 table 作为视觉域对齐，应考虑 render-only table 或 collision-disabled table，避免把“视觉背景变化”和“物理环境变化”混在一个实验里。
- 如果只是验证 CNNMCA backbone 替换，当前更干净的结论应基于 no-table CNNMCA vs SGA-GSN，而不是 CNNMCA+table。