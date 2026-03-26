# Env Side Adapter Contract

本文档只描述 **env / scene 这一侧当前实际会输出什么**，以及 adapter / perception 当前需要自行补什么。

本文档不讨论：
- backbone 结构
- reward、PPO、训练细节
- 未来真实点云恢复算法的具体实现

目标是让任何 agent 或工程师在阅读后都能明确：
- env 当前提供哪些原始观测
- 这些字段的单位、坐标系、shape 是什么
- `before / after` 各自来自哪里
- 哪些能力已经从 env 侧移出，转到 adapter / perception 侧占位实现

## 1. 观测阶段定义

当前 single-step env 有两个观测阶段：

- `before`
  - 来源：数据集 `/Datasets/GraspNet-1billion/tactile-extended`
  - 含义：旧采集流程中已经完成闭合/施力、但尚未 release 的抓持状态
  - 当前实现中，`before` 直接读取旧数据集图像与 metadata

- `after`
  - 来源：当前仿真实时采集
  - 含义：对 `before` grasp pose 施加一次 delta action 后，重新执行一次“手张开 -> 手位姿变换 -> 位置闭合 -> 力闭合”得到的 release 前状态
  - 如果动作语义失败，但 after 数值有效，仍然输出完整 `after`

重要说明：
- 当前 env 不把 `after` 作为下一轮的 `before`
- 每个 episode 始终 `reset -> 单步 action -> done=True`

## 2. 顶层结构

env / scene 输出的是 `RawSensorObservation`，顶层固定分为三块：

- `visual_data`
- `tactile_data`
- `grasp_metadata`

当前 env **不再直接输出**：

- `visual_data.point_cloud`
- `visual_data.distance_to_edge`

这两个量已经从 env 侧移出：

- `point_cloud` 目前在 [adapters.py](/rl-grasp-refine/src/perception/adapters.py#L27) 中用 dummy array 占位
- `edge_proximity` 目前在 [contact_semantics.py](/rl-grasp-refine/src/perception/contact_semantics.py#L44) 中用 tactile-only placeholder 占位

## 3. visual_data

`visual_data` 当前包含以下字段：

- `rgb`
  - 类型：`uint8`
  - shape：`[H, W, 3]`
  - 语义：外部视角 RGB 图像

- `depth`
  - 类型：`float32`
  - shape：`[H, W]`
  - 单位：米
  - 语义：外部视角深度图

- `seg`
  - 类型：`int16`
  - shape：`[H, W]`
  - 语义：分割图，背景通常为 `-1`

- `view_matrix`
  - 类型：`float32`
  - shape：`[4, 4]`
  - 语义：外部相机的 view matrix

- `proj_matrix`
  - 类型：`float32`
  - shape：`[4, 4]`
  - 语义：外部相机的 projection matrix

## 4. tactile_data

`tactile_data` 当前包含以下字段：

- `rgb`
  - 类型：`uint8`
  - shape：`[2, H, W, 3]`
  - 语义：左右 gel 的 RGB 图像

- `depth`
  - 类型：`float32`
  - shape：`[2, H, W]`
  - 单位：米
  - 语义：左右 gel 的深度图

- `proj_matrix`
  - 类型：`float32`
  - shape：`[4, 4]`
  - 语义：TACTO 传感器投影矩阵

- `sensor_poses_world`
  - 类型：`dict`
  - 结构：
    - `left.position`
    - `left.quaternion`
    - `right.position`
    - `right.quaternion`
  - 语义：左右 gel 在世界坐标系下的位姿

- `camera_distance_to_gel_m`
  - 类型：`float`
  - 单位：米
  - 语义：TACTO 相机到 gel 参考面的固定距离

- `contact_map`
  - 类型：`float32`
  - shape：`[2, H, W]`
  - 范围：归一化到 `[0, 1]`
  - 语义：当前实现中的兼容字段，由 tactile depth 归一化得到
  - 用途：给现有占位版 adapter / contact semantics 提供最小可运行输入

- `contact_force`
  - 类型：`float`
  - 语义：当前实现中的兼容标量，约等于 `contact_map` 的均值

## 5. grasp_metadata

`grasp_metadata` 当前包含以下字段：

- `grasp_pose`
  - 类型：`GraspPose`
  - 结构：`position[3] + rotation(rotvec)[3]`
  - 语义：policy 当前作用的 grasp pose

- `object_pose_world`
  - 类型：`dict`
  - 结构：
    - `position[3]`
    - `quaternion[4]`
  - 语义：物体在世界系下的当前位姿

- `source_object_id`
  - 类型：`int`
  - 语义：来源数据集的物体编号

- `source_global_id`
  - 类型：`int`
  - 语义：来源数据集中的全局 sample id

- `observation_stage`
  - 类型：`str`
  - 取值：`before` / `after`

- `segmentation_ids`
  - 类型：`dict`
  - 结构：
    - `object`
    - `hand`
  - 语义：当前 visual segmentation 中 object / hand 对应的 body id

- `gel_pose_world`
  - 类型：`dict`
  - 结构：
    - `left.position`
    - `left.quaternion`
    - `right.position`
    - `right.quaternion`
  - 语义：左右 gel 在世界系下的位姿

- `observation_valid`
  - 类型：`bool`
  - 语义：该观测是否是数值上有效的学习样本

## 6. 坐标系与单位约定

- 世界坐标系
  - env 输出里的 `object_pose_world`、`gel_pose_world` 都在世界坐标系下解释

- visual depth
  - 单位固定为米
  - 数据集读入时：`uint16 / 1000.0`
  - 仿真实时采集时：由 PyBullet depth buffer 转换成米

- tactile depth
  - 单位固定为米
  - 数据集读入时：`uint8 / 10000.0`
  - 仿真实时采集时：直接使用 TACTO 输出的米制 depth

- segmentation
  - 数据集 `before`：
    - 原始 PNG 读入后减 `1`
    - 背景通常恢复为 `-1`
  - 仿真 `after`：
    - 直接使用当前 PyBullet 的 body id

- pose 表达
  - `grasp_pose` 用 `rotation vector`
  - `object_pose_world / gel_pose_world` 用 `quaternion`

## 7. 当前 adapter / contact semantics 责任边界

当前代码中的明确边界如下：

- env 不直接输出世界系点云
- env 不负责输出 `distance_to_edge`
- env 只负责提供：
  - 原始视觉图像
  - 原始触觉图像
  - 深度图
  - 相机矩阵
  - hand / object / gel pose
  - trial 结果与状态位

对应地：

- adapter 当前需要自己产出 `point_cloud`
  - 现状：在 [adapters.py](/rl-grasp-refine/src/perception/adapters.py#L33) 里先返回 dummy `zeros((1, 3))`
  - TODO：未来应根据 `visual_data.depth + seg + view_matrix + proj_matrix` 恢复世界系点云

- `ContactSemanticsExtractor` 当前需要自己产出 `coverage_ratio` 和 `edge_proximity`
  - 现状：在 [contact_semantics.py](/rl-grasp-refine/src/perception/contact_semantics.py#L32) 和 [contact_semantics.py](/rl-grasp-refine/src/perception/contact_semantics.py#L44) 中，两个量都只从 tactile 信号计算占位值
  - TODO：未来应改为基于 tac RGB / tac depth 的真实公式

## 8. trial 输出约定

除 `RawSensorObservation` 外，scene 还会返回 `trial_result`，其中 `trial_metadata` 至少包含：

- `trial_status`
  - 当前可能值：
    - `success`
    - `failure_interference`
    - `failure_contact_lost`
    - `failure_effort_timeout`
    - `failure_pre_release_drop`
    - `failure_release_drop`
    - `system_invalid_observation`
    - `system_sim_error`

- `release_executed`
  - `bool`
  - 表示该 step 是否真的执行了 release

- `valid_for_learning`
  - `bool`
  - `false` 时，该 step 不应进入 PPO buffer，也不应进入 calibrator update

- `failure_reason`
  - `str | null`

- `runtime_counters`
  - `dict`
  - 用于记录 `reset/refine/release` 阶段的 close / effort / release 步数等运行时诊断信息

另外，顶层 `trial_result` 还包含：

- `drop_success`
  - `0 / 1`
  - 语义：
    - 正常 trial：由 release 后是否仍保持抓持决定
    - 动作语义失败：不执行 release，直接置 `0`

## 9. 有效观测与系统故障

以下情况仍视为有效观测：

- 触觉很弱
- 接触区域很小
- 动作语义失败，但 after 图像数值正常

以下情况视为系统故障：

- 渲染失败
- 数组 shape 错误
- 数组为空
- 出现 NaN / Inf
- 仿真异常导致 after 无法正常采样

系统故障的输出约定：

- 仍返回一个结构完整的占位 `after observation`
- `grasp_metadata.observation_valid = false`
- `trial_result.trial_metadata.valid_for_learning = false`
- `drop_success = 0`
