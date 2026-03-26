# Env 迁移结果

本文档记录当前 `src/envs` 迁移后的**已落地结果**，不是计划草案。

## 1. 当前系统形态

当前环境已经固定为 **dataset-backed single-step grasp refinement**：

- 每个 episode 仍是 `reset -> 一次 delta action -> trial -> done=True`
- 下一轮始终重新从数据集采一个新的 `before`
- 不把当前 `after` 作为下一轮 `before`

样本来源已经切到 `/Datasets/GraspNet-1billion/tactile-extended`：

- `before` 原始观测直接读取旧数据集图像与 metadata
- `after` 原始观测由当前仿真实时采集
- 数据集 entry 采样不按旧 `isPositive` 过滤
- 旧 `isPositive` 只保留为来源元数据

## 2. 运行时外部依赖状态

`src/envs` 运行时已经不再依赖 `/tac-sim-sys` 目录。

当前做法是：

- hand Python 逻辑 vendoring 到 [gsmini_panda_hand.py](/rl-grasp-refine/src/envs/hand/gsmini_panda_hand.py)
- hand / tacto 资产 vendoring 到 [assets](/rl-grasp-refine/src/envs/assets)
- 统一由 [asset_paths.py](/rl-grasp-refine/src/envs/asset_paths.py#L23) 解析资产路径

当前 env 仍然允许依赖外部数据资产：

- 数据集图像与 `_metadata.json`
- `src/envs/object_model` 软链接指向的 object / table URDF 资产

## 3. Scene 资产与真实物体导入

当前 scene 已经使用真实物体导入，不再使用 box 近似：

- table 路径固定为 [object_model/table/table.urdf](/rl-grasp-refine/src/envs/object_model/table/table.urdf)
- object 路径固定为 `src/envs/object_model/model/{object_id:03d}/object.urdf`
- 解析逻辑在 [asset_paths.py](/rl-grasp-refine/src/envs/asset_paths.py#L35)
- 实际加载逻辑在 [scene_assets.py](/rl-grasp-refine/src/envs/scene_assets.py#L20)

物体 pose 直接使用数据集给出的 `pre_grasp.object_pose_world`，不再做额外 z 偏移修正。

## 4. `PyBulletScene` 的当前职责

[PyBulletScene](/rl-grasp-refine/src/envs/pybullet_scene.py#L29) 当前是单个 episode 的物理场景编排器，负责：

- scene lifecycle
- reset / refine / release trial 时序
- before / after raw observation 出口
- trial metadata 出口

辅助逻辑已经拆到独立 helper：

- 资产路径解析：[asset_paths.py](/rl-grasp-refine/src/envs/asset_paths.py)
- 资产实例化：[scene_assets.py](/rl-grasp-refine/src/envs/scene_assets.py)
- contact / force 工具：[scene_contact.py](/rl-grasp-refine/src/envs/scene_contact.py)
- 原始观测采集：[scene_observation.py](/rl-grasp-refine/src/envs/scene_observation.py)

因此，`PyBulletScene` 本身现在保留的是“编排层”职责，不再承载全部资产与观测细节。

## 5. Reset 与 Refine 的物理语义

当前 `reset_scene()` 和 `apply_refinement()` 已共享同一套抓持重建流程 [pybullet_scene.py](/rl-grasp-refine/src/envs/pybullet_scene.py#L291)：

1. `hand.reset()`，让手回到全开状态
2. 将物体重置到指定 object pose
3. 重建 object constraint
4. 将手 base 放到目标 hand pose
5. 执行位置闭合，直到接触成立或 timeout
6. 执行力闭合，直到达到目标力或失败
7. 根据接触保持情况与干涉情况给出结果

这意味着：

- `reset` 后场景处于 pre-release 的稳定抓持状态
- `step` 不再是“已抓住状态上的瞬移”
- `apply_refinement()` 的语义已经修正为：
  - 手张开
  - 物体重置到数据集初始 pose
  - 手移动到 refined pose
  - 重新执行位置闭合 + 力闭合

因此，当前 `after` 表示的是**refined grasp reconstruction 完成后、release 之前**的状态。

## 6. Raw observation 边界

当前 env 输出侧固定提供：

- `visual_data.rgb`
- `visual_data.depth`
- `visual_data.seg`
- `visual_data.view_matrix`
- `visual_data.proj_matrix`
- `tactile_data.rgb`
- `tactile_data.depth`
- `tactile_data.proj_matrix`
- `tactile_data.sensor_poses_world`
- `tactile_data.camera_distance_to_gel_m`
- `tactile_data.contact_map`
- `tactile_data.contact_force`
- `grasp_metadata.{grasp_pose, object_pose_world, source_object_id, source_global_id, observation_stage, segmentation_ids, gel_pose_world, observation_valid}`

相较于上一版计划，现在有一个重要修正：

- env 已经**不再**输出 `visual_data.point_cloud`
- env 已经**不再**输出 `visual_data.distance_to_edge`

当前对应责任已经转移为：

- `point_cloud` 由 [adapters.py](/rl-grasp-refine/src/perception/adapters.py#L27) 在本地用 dummy array 占位
- `coverage_ratio / edge_proximity` 由 [contact_semantics.py](/rl-grasp-refine/src/perception/contact_semantics.py#L32) 以 tactile-only placeholder 方式产出

更详细的 env 输出契约见 [env_side_adapter_contract.md](/rl-grasp-refine/docs/env_side_adapter_contract.md)。

## 7. Failure 与 Trial 语义

当前 `run_grasp_trial()` 行为固定如下 [pybullet_scene.py](/rl-grasp-refine/src/envs/pybullet_scene.py#L157)：

- 如果 refine 后 after 数值无效或仿真异常：
  - `trial_status` 为 `system_invalid_observation` 或 `system_sim_error`
  - `valid_for_learning=False`
  - 不进入 PPO / calibrator

- 如果 refine 是动作语义失败，但 after 有效：
  - 不执行 release
  - `drop_success=0`
  - `valid_for_learning=True`

- 如果 refine 成功：
  - 执行 release
  - 根据 release 后是否仍接触给 `drop_success`

当前 `trial_status` 包括：

- `success`
- `failure_interference`
- `failure_contact_lost`
- `failure_effort_timeout`
- `failure_pre_release_drop`
- `failure_release_drop`
- `system_invalid_observation`
- `system_sim_error`

## 8. 调试与人工检查入口

当前环境支持手工 GUI 检查：

- 脚本：[sanity_check_env.py](/rl-grasp-refine/scripts/sanity_check_env.py)
- 支持 `--gui --interactive`
- 启动后自动 `reset()`
- `r` 执行随机 step
- `n` 强制新一轮 `reset()`
- `q` 退出

脚本会打印：

- 样本来源 id
- 资产路径
- before / after 读取状态
- reset / refine close 与 effort 步数
- `trial_status`
- `release_executed`
- `valid_for_learning`
- `drop_success`
- reward breakdown

## 9. 已知折中与后续 TODO

当前已知折中有三点：

- `before` 来自旧数据集图像，`after` 来自新仿真，二者仍存在域差异
- adapter 侧 `point_cloud` 目前仍是 dummy array，占位保证 train loop 可运行
- `contact_semantics` 目前是 tactile-only placeholder，不是最终公式

下一步更合理的方向是：

- 在 adapter 中根据 `visual depth + seg + 矩阵` 恢复世界系视觉点云
- 在 perception 中根据 tac RGB / tac depth 定义真实 `coverage_ratio / edge_proximity`
- 然后再清理 `tactile_data.contact_map` 这类兼容字段
