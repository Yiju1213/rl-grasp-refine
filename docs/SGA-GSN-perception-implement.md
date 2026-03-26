# SGA-GSN Perception 新计划（基于已确认事实）

## 已确认事实
- 当前 `predictor` 不是吃 `Observation`，而是吃 `raw_obs + latent_feature`。
  - 具体是：`latent_feature`、`grasp_pose`、`contact_semantic`、`sensor_summary`。
  - 其中 `contact_semantic` 不是 predictor 自己算的，而是 `ObservationBuilder` 先算完再回填到 `raw_obs.grasp_metadata`。
- 当前 `feature extractor` 和 `contact semantics` 都直接吃 `raw_obs`，不是吃 `Observation`。
- 当前默认 debug 配置下，RL 和 perception 都实际跑在 CPU，上游代码没有 perception 设备管理，因此默认是 **0 次 CPU/GPU 切换**。
- 如果按现在这套 `FeatureExtractor.extract() -> numpy/cpu -> StabilityPredictor.predict_logit()` 结构直接把 perception 放到 GPU，会产生不必要的来回搬运。
- `src/utils/geometry.py` 与 `vtg3d_utils` 在核心几何逻辑上是同构的：
  - visual：`depth + seg mask + proj -> view/camera points -> inv(view_matrix) -> world`
  - tactile：`uint8 depth -> meter -> cam-to-gel restore -> gel transform -> gel pose -> world`
- 已做真实数据多样本对齐验证：
  - visual 点数完全一致，质心误差约 `1e-6 m`
  - tactile 点数完全一致，质心误差约 `1e-5 ~ 1e-5e-5 m`
- 现差异主要不在核心几何，而在“VTG3D 输入准备细节”：
  - tactile `step` 下采样
  - gel/contact mask 输出
  - `downsample_by_dist_ratio`
  - zero-mean
  - 最终 `sc_input / gs_input` 的采样与拼接规则

## 实现计划
- 保持 `visual_chain_mode=保留原链路`：
  - `adapter` 先从 env `raw_obs` 重建 VTG3D 风格的 `sc_input / gs_input`
  - frozen AdaPoinTr `return_latent=True`
  - frozen SGSNet body
  - frozen SGSNet `predict_head`
- perception 接入方式继续采用 `bridge /AdaPoinTr`，不做本地大移植。
- 设备策略固定为 `CUDA only`，不做 CPU fallback。

## 感知栈重构
- 新增一个共享的 `SGAGSNPerceptionRuntime`，负责：
  - 一次性加载 AdaPoinTr 和 SGSNet checkpoint
  - 统一持有 CUDA device、eval/no_grad 状态
  - 暴露 `forward(prepared_inputs)`，一次完成：
    - AdaPoinTr latent/coarse 输出
    - SGSNet body 输出
    - SGSNet head logit 输出
- 不再让 `FeatureExtractor` 和 `StabilityPredictor` 各自重新做 adapter 和各自触发一次 GPU 推理。
- 对 SGA-GSN 路径改为：
  - adapter 在 CPU 上把 `raw_obs` 准备成 `PreparedVTGInputs`
  - runtime 在 GPU 上一次前向，返回 `body_feature` 和 `raw_logit`
  - `ObservationBuilder` 用这一次结果同时填 `latent_feature` 和 `raw_stability_logit`
- `FeatureExtractor` 的职责改成“返回 body feature”。
- `StabilityPredictor` 的职责改成“消费已编码 body feature，返回 head logit”，不再独立读取 `raw_obs` 做二次适配。
- `contact semantics` 仍保持独立 placeholder，并继续只输出 feature，不参与几何链路。

## CPU/GPU 切换最小化方案
- 保持 env、scene、raw observation、contact semantics、RL buffer 都在 CPU。
- perception 重建输入在 CPU 完成，打包后只做一次 `CPU -> GPU`。
- AdaPoinTr + SGSNet body + SGSNet head 在 GPU 内连续完成，不在中途把 `latent_feature` 拉回 CPU。
- 最终只把 `body_feature(np.float32)` 和 `raw_logit(float)` 做一次 `GPU -> CPU` 返回给 `ObservationBuilder`。
- 在这个方案下：
  - 每个观测阶段是 `1 次上传 + 1 次下载`
  - 单个 episode 有 `before` 和 `after` 两次感知，因此是 **4 次切换**
  - 这是在不把整套 env/RL 改成 GPU-first 的前提下最简且合理的下界
- 本次不把 PPO actor-critic 迁到 GPU。
  - 原因：policy 很轻，迁过去只会额外引入 `obs CPU->GPU` 和 `action GPU->CPU`
  - 当前优先级是把重模型 perception 的切换压到最少

## Adapter 与 Geometry
- 以 `src/utils/geometry.py` 为本仓基线实现，不直接搬 `vtg3d_utils` 的 Open3D 版本进入主链。
- 在本仓补齐与 VTG3D 对齐所缺的几项：
  - tactile depth 的 `step` 采样版重建
  - gel surface mask / contact region 提取
  - `downsample_by_dist_ratio`
  - zero-mean 与应用
  - `gs_input` 的左右 tactile 合并与第 4 通道约定
  - `sc_input` 的 visual/contact 拼接规则
- 新 adapter 输出至少包含：
  - debug 用 world-frame visual object cloud
  - debug 用 left/right tactile world cloud
  - 真正推理用 `sc_input`
  - 真正推理用 `gs_input`
- 几何实现以 `vtg3d_utils` 为真值参考，保留数值对齐测试。

## Demo Script
- 新增一个可肉眼检查的 demo 脚本，建议路径：
  - `scripts/demo_live_capture_reconstruction.py`
- 脚本目标：
  - 从真实 env reset 取得 live `before`
  - 可选执行零动作，再取得 live `after`
  - 对 `before/after` 分别重建：
    - visual object world cloud
    - tactile left/right world cloud
    - `sc_input`
    - `gs_input`
  - 保存：
    - RGB/depth/seg PNG
    - 点云 `.ply`
    - 原始数组 `.npz`
  - 可选：
    - `--use-gui` 看 PyBullet live capture
    - `--show-open3d` 直接弹点云窗口
- 脚本输出需要明确打印绝对路径，方便直接打开检查。
- 脚本建议复用现有 `debug_single_rollout.py` / `sanity_check_env.py` 的 env 构建方式，不新造第二套入口。

## 测试与验收
- 增加几何对齐测试：
  - visual world cloud vs `vtg3d_utils`
  - tactile world cloud vs `vtg3d_utils`
- 增加 adapter 形状测试：
  - `sc_input`
  - `gs_input`
  - 左右 tactile 点数
  - zero-mean 后 shape/finite 检查
- 增加 runtime smoke test：
  - AdaPoinTr 和 SGSNet 能在 CUDA 加载
  - checkpoint split 正常
  - body/head 输出维度稳定
- 增加 observation builder smoke test：
  - 一次 reset/step 后 `latent_feature`、`raw_logit` 都有限
  - 同一个 `raw_obs` 只触发一次 GPU perception forward
- 增加 demo 脚本冒烟测试：
  - 至少能成功导出一组 PNG + PLY + NPZ

## 假设
- `v3.md` 仍作为 env 运行事实优先来源；`env_side_adapter_contract.md` 中关于 `before` 来自旧数据集的描述视为过时。
- 本次只支持当前兼容的 PPCT checkpoint 族，不接旧的 `woCompletion` 训练分支。
- 本次不强行把 latent 压回 `32` 维；RL 侧输入维度跟随真实 body feature 维度调整。
