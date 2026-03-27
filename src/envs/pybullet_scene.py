from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as pb
import pybulletX as px

from src.envs.asset_paths import resolve_object_urdf, resolve_scene_asset_paths
from src.envs.scene_assets import (
    attach_object_to_tacto_sensor,
    create_tacto_sensor,
    remove_object_body,
    remove_object_from_tacto_sensor,
    spawn_hand,
    spawn_object,
)
from src.envs.scene_contact import check_contact, check_target_force, has_any_contact
from src.envs.scene_observation import capture_scene_observation, make_invalid_after_observation
from src.structures.action import GraspPose
from src.structures.observation import RawSensorObservation
from src.utils.geometry import rotvec_to_quaternion

try:
    import pybullet_data
except ImportError:  # pragma: no cover - pybullet wheels usually vendor this.
    pybullet_data = None


GRIPPER_CLOSE_WIDTH = 0.8


class PyBulletScene:
    """Dataset-backed single-step scene with live before/after observations."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.asset_paths = resolve_scene_asset_paths()
        self.client_id: int | None = None
        self.sample_cfg: dict[str, Any] | None = None
        self.current_grasp_pose: GraspPose | None = None
        self.initial_grasp_pose: GraspPose | None = None
        self.before_raw_obs: RawSensorObservation | None = None
        self.hand = None
        self.object_body = None
        self.tacto_sensor = None
        self.object_constraint_id: int | None = None
        self._after_observation_ready = False
        self._pending_trial_status: str | None = None
        self._last_after_raw_obs: RawSensorObservation | None = None
        self._last_runtime_counters: dict[str, int] = {}
        self._baseline_undesired_contact_count = 0
        self._last_reset_debug: dict[str, Any] = {}
        self._last_refine_debug: dict[str, Any] = {}
        self._loaded_source_object_id: int | None = None
        self._last_object_action: str | None = None
        self._last_tacto_action: str | None = None
        self._connect()

    def _connect(self) -> None:
        if self.client_id is not None:
            return
        mode = pb.GUI if self.cfg.get("use_gui", False) else pb.DIRECT
        px.init(mode=mode)
        self.client_id = 0
        if pybullet_data is not None:
            pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        pb.setTimeStep(float(self.cfg.get("time_step", 1.0 / 200.0)), physicsClientId=self.client_id)
        if bool(self.cfg.get("use_gui", False)):
            pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)
            pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1, physicsClientId=self.client_id)
            pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1, physicsClientId=self.client_id)

    @staticmethod
    def _dict_to_pose(pose_dict: dict) -> GraspPose:
        return GraspPose(position=pose_dict["position"], rotation=pose_dict["rotation"])

    def reset_scene(self, sample_cfg: dict) -> None:
        self.sample_cfg = sample_cfg
        self.initial_grasp_pose = self._dict_to_pose(sample_cfg["initial_grasp_pose"])
        self.current_grasp_pose = self.initial_grasp_pose
        self._pending_trial_status = None
        self._after_observation_ready = False
        self._last_after_raw_obs = None
        self._last_runtime_counters = {}
        self._last_reset_debug = {}
        self._last_refine_debug = {}
        self.before_raw_obs = None

        tacto_action = self._ensure_static_scene_assets()
        self._restore_reset_runtime_state()
        object_action = self._ensure_sample_object()

        reset_result = self._run_grasp_reconstruction(
            stage="reset",
            hand_pose_world=self.sample_cfg["pre_grasp"]["hand_pose_world"],
            object_pose_world=self.sample_cfg["pre_grasp"]["object_pose_world"],
            raise_on_failure=True,
        )
        reset_result["object_action"] = object_action
        reset_result["tacto_action"] = tacto_action
        reset_result["hand_body_id"] = None if self.hand is None else int(self.hand.id)
        reset_result["object_body_id"] = None if self.object_body is None else int(self.object_body.id)
        self._last_reset_debug = reset_result
        self._baseline_undesired_contact_count = int(reset_result["undesired_contact_count"])
        self.before_raw_obs = capture_scene_observation(
            sample_cfg=self.sample_cfg,
            scene_cfg=self.cfg,
            tacto_sensor=self.tacto_sensor,
            hand=self.hand,
            object_body=self.object_body,
            client_id=self.client_id,
            current_grasp_pose=self.current_grasp_pose,
            stage="before",
        )
        self._request_gui_render()

    def set_initial_grasp(self, grasp_pose) -> None:
        self.initial_grasp_pose = grasp_pose
        self.current_grasp_pose = grasp_pose

    def apply_refinement(self, refined_pose) -> None:
        self.current_grasp_pose = refined_pose
        self._after_observation_ready = True
        self._pending_trial_status = None
        self._last_after_raw_obs = None
        refined_hand_pose = {
            "position": refined_pose.position.tolist(),
            "quaternion": rotvec_to_quaternion(refined_pose.rotation).tolist(),
        }
        try:
            refine_result = self._run_grasp_reconstruction(
                stage="refine",
                hand_pose_world=refined_hand_pose,
                object_pose_world=self.sample_cfg["pre_grasp"]["object_pose_world"],
                raise_on_failure=False,
            )
            self._last_refine_debug = refine_result
            if refine_result["trial_status"] != "success":
                self._pending_trial_status = refine_result["trial_status"]
            if refine_result["trial_status"] == "system_sim_error":
                self._last_after_raw_obs = make_invalid_after_observation(self.sample_cfg, self.cfg, self.current_grasp_pose)
        except Exception:
            self._pending_trial_status = "system_sim_error"
            self._last_refine_debug = {
                "stage": "refine",
                "trial_status": "system_sim_error",
                "failure_reason": "simulation_error",
            }
            self._last_after_raw_obs = make_invalid_after_observation(self.sample_cfg, self.cfg, self.current_grasp_pose)

    def get_raw_observation(self):
        if not self._after_observation_ready:
            if self.before_raw_obs is None:
                raise RuntimeError("reset_scene() must be called before requesting before observation.")
            return self.before_raw_obs

        if self._last_after_raw_obs is not None and not bool(
            self._last_after_raw_obs.grasp_metadata.get("observation_valid", True)
        ):
            return self._last_after_raw_obs

        try:
            self._last_after_raw_obs = capture_scene_observation(
                sample_cfg=self.sample_cfg,
                scene_cfg=self.cfg,
                tacto_sensor=self.tacto_sensor,
                hand=self.hand,
                object_body=self.object_body,
                client_id=self.client_id,
                current_grasp_pose=self.current_grasp_pose,
                stage="after",
            )
        except Exception:
            self._pending_trial_status = "system_sim_error"
            self._last_after_raw_obs = make_invalid_after_observation(self.sample_cfg, self.cfg, self.current_grasp_pose)
        return self._last_after_raw_obs

    def run_grasp_trial(self) -> dict:
        runtime_counters = dict(self._last_runtime_counters)
        release_executed = False

        if self._pending_trial_status == "system_sim_error":
            return self._trial_result(
                drop_success=0,
                trial_status="system_sim_error",
                release_executed=release_executed,
                valid_for_learning=False,
                failure_reason="simulation_error",
                runtime_counters=runtime_counters,
            )

        if self._last_after_raw_obs is None:
            return self._trial_result(
                drop_success=0,
                trial_status="system_invalid_observation",
                release_executed=release_executed,
                valid_for_learning=False,
                failure_reason="missing_after_observation",
                runtime_counters=runtime_counters,
            )

        observation_valid = bool(self._last_after_raw_obs.grasp_metadata.get("observation_valid", True))
        if not observation_valid:
            return self._trial_result(
                drop_success=0,
                trial_status="system_invalid_observation",
                release_executed=release_executed,
                valid_for_learning=False,
                failure_reason="invalid_after_observation",
                runtime_counters=runtime_counters,
            )

        if self._pending_trial_status is not None:
            return self._trial_result(
                drop_success=0,
                trial_status=self._pending_trial_status,
                release_executed=release_executed,
                valid_for_learning=True,
                failure_reason=self._pending_trial_status,
                runtime_counters=runtime_counters,
            )

        try:
            release_executed = True
            if self.object_constraint_id is not None:
                pb.removeConstraint(self.object_constraint_id, physicsClientId=self.client_id)
                self.object_constraint_id = None
            pb.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)

            time_step = float(self._runtime_cfg().get("time_step", 1.0 / 200.0))
            release_duration = float(self._runtime_cfg().get("release_duration_s", 2.0))
            release_interval = int(self._runtime_cfg().get("release_check_interval_steps", 10))
            total_steps = max(int(release_duration / max(time_step, 1e-6)), 1)

            for step in range(total_steps):
                pb.stepSimulation(physicsClientId=self.client_id)
                if (step + 1) % max(release_interval, 1) == 0 and not has_any_contact(self.hand.id, self.object_body.id):
                    self._last_runtime_counters["release_steps"] = step + 1
                    runtime_counters["release_steps"] = step + 1
                    return self._trial_result(
                        drop_success=0,
                        trial_status="failure_release_drop",
                        release_executed=release_executed,
                        valid_for_learning=True,
                        failure_reason="contact_lost_after_release",
                        runtime_counters=runtime_counters,
                    )

            self._last_runtime_counters["release_steps"] = total_steps
            runtime_counters["release_steps"] = total_steps
            drop_success = int(has_any_contact(self.hand.id, self.object_body.id))
            return self._trial_result(
                drop_success=drop_success,
                trial_status="success" if drop_success == 1 else "failure_release_drop",
                release_executed=release_executed,
                valid_for_learning=True,
                failure_reason=None if drop_success == 1 else "contact_lost_after_release",
                runtime_counters=runtime_counters,
            )
        except Exception:
            return self._trial_result(
                drop_success=0,
                trial_status="system_sim_error",
                release_executed=release_executed,
                valid_for_learning=False,
                failure_reason="simulation_error_during_release",
                runtime_counters=runtime_counters,
            )

    def close(self) -> None:
        if self.client_id is not None:
            pb.disconnect(self.client_id)
        self.client_id = None
        self.hand = None
        self.object_body = None
        self.tacto_sensor = None
        self.object_constraint_id = None
        self._loaded_source_object_id = None

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "source_object_id": None if self.sample_cfg is None else int(self.sample_cfg["source"]["object_id"]),
            "source_global_id": None if self.sample_cfg is None else int(self.sample_cfg["source"]["global_id"]),
            "hand_body_id": None if self.hand is None else int(self.hand.id),
            "current_object_body_id": None if self.object_body is None else int(self.object_body.id),
            "last_object_action": self._last_object_action,
            "last_tacto_action": self._last_tacto_action,
            "asset_paths": {
                "hand_python": str(self.asset_paths.hand_python),
                "hand_urdf": str(self.asset_paths.hand_urdf),
                "tacto_background": str(self.asset_paths.tacto_background),
                "tacto_config": str(self.asset_paths.tacto_config),
                "object_urdf": None
                if self.sample_cfg is None
                else str(resolve_object_urdf(self.sample_cfg["source"]["object_id"])),
            },
            "pre_grasp": None if self.sample_cfg is None else self.sample_cfg.get("pre_grasp"),
            "runtime_counters": dict(self._last_runtime_counters),
            "pending_trial_status": self._pending_trial_status,
            "reset_debug": dict(self._last_reset_debug),
            "refine_debug": dict(self._last_refine_debug),
            "before_shapes": self._raw_observation_shapes(self.before_raw_obs),
            "after_shapes": self._raw_observation_shapes(self._last_after_raw_obs),
            "after_observation_valid": None
            if self._last_after_raw_obs is None
            else bool(self._last_after_raw_obs.grasp_metadata.get("observation_valid", True)),
        }

    def _runtime_cfg(self) -> dict[str, Any]:
        return (self.sample_cfg or {}).get("runtime", {})

    @staticmethod
    def _staging_hand_pose_world() -> dict[str, list[float]]:
        return {
            "position": [0.0, 0.0, 2.0],
            "quaternion": [0.0, 0.0, 0.0, 1.0],
        }

    def _ensure_static_scene_assets(self) -> str:
        tacto_action = "reuse"
        if self.hand is None:
            self.hand = spawn_hand(self.asset_paths, self._staging_hand_pose_world())
        if self.tacto_sensor is None:
            self.tacto_sensor = create_tacto_sensor(self.asset_paths, self.cfg, self.hand, self.client_id)
            tacto_action = "created"
        self._last_tacto_action = tacto_action
        return tacto_action

    def _set_gui_rendering_enabled(self, enabled: bool) -> None:
        if not bool(self.cfg.get("use_gui", False)) or self.client_id is None:
            return
        pb.configureDebugVisualizer(
            pb.COV_ENABLE_RENDERING,
            1 if enabled else 0,
            physicsClientId=self.client_id,
        )

    def _request_gui_render(self) -> None:
        if not bool(self.cfg.get("use_gui", False)) or self.client_id is None:
            return
        pb.configureDebugVisualizer(
            pb.COV_ENABLE_SINGLE_STEP_RENDERING,
            1,
            physicsClientId=self.client_id,
        )

    def _restore_reset_runtime_state(self) -> None:
        if pybullet_data is not None:
            pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0.0, 0.0, 0.0, physicsClientId=self.client_id)
        pb.setTimeStep(float(self._runtime_cfg().get("time_step", 1.0 / 200.0)), physicsClientId=self.client_id)
        self._clear_object_constraint()

    def _ensure_sample_object(self) -> str:
        target_object_id = int(self.sample_cfg["source"]["object_id"])
        object_action = "reuse"

        if self.object_body is None:
            self._set_gui_rendering_enabled(False)
            try:
                self.object_body = spawn_object(
                    self.asset_paths,
                    object_id=target_object_id,
                    object_pose_world=self.sample_cfg["pre_grasp"]["object_pose_world"],
                )
                attach_object_to_tacto_sensor(self.tacto_sensor, self.object_body)
                self._loaded_source_object_id = target_object_id
                object_action = "spawn"
            finally:
                self._set_gui_rendering_enabled(True)
                self._request_gui_render()
        elif self._loaded_source_object_id != target_object_id:
            self._set_gui_rendering_enabled(False)
            try:
                remove_object_from_tacto_sensor(self.tacto_sensor, self.object_body.id)
                remove_object_body(self.object_body.id, self.client_id)
                self.object_body = spawn_object(
                    self.asset_paths,
                    object_id=target_object_id,
                    object_pose_world=self.sample_cfg["pre_grasp"]["object_pose_world"],
                )
                attach_object_to_tacto_sensor(self.tacto_sensor, self.object_body)
                self._loaded_source_object_id = target_object_id
                object_action = "swap"
            finally:
                self._set_gui_rendering_enabled(True)
                self._request_gui_render()

        self._last_object_action = object_action
        return object_action

    def _run_grasp_reconstruction(
        self,
        stage: str,
        hand_pose_world: dict[str, Any],
        object_pose_world: dict[str, Any],
        raise_on_failure: bool,
    ) -> dict[str, Any]:
        stage_prefix = str(stage)
        time_step = float(self._runtime_cfg().get("time_step", 1.0 / 200.0))
        close_timeout_s = float(self._runtime_cfg().get("close_timeout_s", 1.0))
        effort_timeout_s = float(self._runtime_cfg().get("effort_timeout_s", 1.0))
        grip_force = float(self._runtime_cfg().get("grip_force", 30.0))
        settle_steps = int(self._runtime_cfg().get("post_refine_settle_steps", 8)) if stage == "refine" else 0
        close_steps = max(int(close_timeout_s / max(time_step, 1e-6)), 1)
        effort_steps = max(int(effort_timeout_s / max(time_step, 1e-6)), 1)
        result = {
            "stage": stage_prefix,
            "trial_status": "success",
            "failure_reason": None,
            "close_steps": close_steps,
            "effort_steps": effort_steps,
            "undesired_contact_count": 0,
            "hand_pose_world": hand_pose_world,
            "object_pose_world": object_pose_world,
        }

        try:
            self.hand.reset()
            self._set_object_pose(object_pose_world)
            self._recreate_object_constraint(object_pose_world["position"])
            self._set_hand_pose(hand_pose_world)
            pb.stepSimulation(physicsClientId=self.client_id)
            self.hand.finger_position_control(GRIPPER_CLOSE_WIDTH)
            contact_reached = False
            for step in range(close_steps):
                pb.stepSimulation(physicsClientId=self.client_id)
                contact, undesired = check_contact(self.hand.id, self.object_body.id, self.hand.gsmini_joint_ids)
                if contact:
                    result["close_steps"] = step + 1
                    result["undesired_contact_count"] = int(undesired)
                    contact_reached = True
                    break

            if not contact_reached:
                return self._finalize_reconstruction_failure(
                    result=result,
                    stage=stage_prefix,
                    trial_status="failure_contact_lost",
                    failure_reason="contact_not_reached",
                    raise_on_failure=raise_on_failure,
                    runtime_message="Failed to reconstruct grasp contact during reset.",
                )
            self.hand.finger_position_control(GRIPPER_CLOSE_WIDTH, grip_force)
            effort_reached = False
            for step in range(effort_steps):
                pb.stepSimulation(physicsClientId=self.client_id)
                contact, undesired = check_contact(self.hand.id, self.object_body.id, self.hand.gsmini_joint_ids)
                result["effort_steps"] = step + 1
                result["undesired_contact_count"] = int(undesired)
                if not contact:
                    return self._finalize_reconstruction_failure(
                        result=result,
                        stage=stage_prefix,
                        trial_status="failure_pre_release_drop",
                        failure_reason="contact_lost_during_effort_close",
                        raise_on_failure=raise_on_failure,
                        runtime_message="Failed to maintain grasp contact during reset.",
                    )
                if check_target_force(self.hand.id, self.hand.gripper_joint_ids, grip_force):
                    effort_reached = True
                    break

            if not effort_reached:
                return self._finalize_reconstruction_failure(
                    result=result,
                    stage=stage_prefix,
                    trial_status="failure_effort_timeout",
                    failure_reason="target_force_not_reached",
                    raise_on_failure=raise_on_failure,
                    runtime_message="Failed to reach target grip force during reset.",
                )

            for _ in range(max(settle_steps, 0)):
                pb.stepSimulation(physicsClientId=self.client_id)

            contact, undesired = check_contact(self.hand.id, self.object_body.id, self.hand.gsmini_joint_ids)
            result["undesired_contact_count"] = int(undesired)
            if not contact:
                return self._finalize_reconstruction_failure(
                    result=result,
                    stage=stage_prefix,
                    trial_status="failure_pre_release_drop",
                    failure_reason="contact_lost_after_settle",
                    raise_on_failure=raise_on_failure,
                    runtime_message="Failed to keep object in gripper during reset.",
                )

            if stage == "refine" and undesired > self._baseline_undesired_contact_count:
                return self._finalize_reconstruction_failure(
                    result=result,
                    stage=stage_prefix,
                    trial_status="failure_interference",
                    failure_reason="undesired_contact_increased",
                    raise_on_failure=False,
                    runtime_message="Unexpected interference during refinement.",
                )

            self._last_runtime_counters.update(
                {
                    f"{stage_prefix}_close_steps": int(result["close_steps"]),
                    f"{stage_prefix}_effort_steps": int(result["effort_steps"]),
                }
            )
            self._request_gui_render()
            return result
        except Exception as exc:
            if raise_on_failure:
                raise RuntimeError(str(exc)) from exc
            result["trial_status"] = "system_sim_error"
            result["failure_reason"] = "simulation_error"
            self._request_gui_render()
            return result

    def _finalize_reconstruction_failure(
        self,
        result: dict[str, Any],
        stage: str,
        trial_status: str,
        failure_reason: str,
        raise_on_failure: bool,
        runtime_message: str,
    ) -> dict[str, Any]:
        result["trial_status"] = trial_status
        result["failure_reason"] = failure_reason
        self._last_runtime_counters.update(
            {
                f"{stage}_close_steps": int(result["close_steps"]),
                f"{stage}_effort_steps": int(result["effort_steps"]),
            }
        )
        if raise_on_failure:
            raise RuntimeError(runtime_message)
        return result

    def _set_object_pose(self, object_pose_world: dict[str, Any]) -> None:
        pb.resetBasePositionAndOrientation(
            self.object_body.id,
            object_pose_world["position"],
            object_pose_world["quaternion"],
            physicsClientId=self.client_id,
        )
        pb.resetBaseVelocity(
            self.object_body.id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.client_id,
        )

    def _set_hand_pose(self, hand_pose_world: dict[str, Any]) -> None:
        pb.resetBasePositionAndOrientation(
            self.hand.id,
            hand_pose_world["position"],
            hand_pose_world["quaternion"],
            physicsClientId=self.client_id,
        )

    def _clear_object_constraint(self) -> None:
        if self.object_constraint_id is None:
            return
        pb.removeConstraint(self.object_constraint_id, physicsClientId=self.client_id)
        self.object_constraint_id = None

    def _recreate_object_constraint(self, object_position: list[float]) -> None:
        self._clear_object_constraint()
        self.object_constraint_id = pb.createConstraint(
            self.object_body.id,
            -1,
            -1,
            -1,
            jointType=pb.JOINT_POINT2POINT,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=object_position,
            physicsClientId=self.client_id,
        )

    def _trial_result(
        self,
        drop_success: int,
        trial_status: str,
        release_executed: bool,
        valid_for_learning: bool,
        failure_reason: str | None,
        runtime_counters: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "drop_success": int(drop_success),
            "trial_metadata": {
                "trial_status": trial_status,
                "release_executed": bool(release_executed),
                "valid_for_learning": bool(valid_for_learning),
                "failure_reason": failure_reason,
                "runtime_counters": runtime_counters,
            },
        }

    @staticmethod
    def _raw_observation_shapes(raw_obs: RawSensorObservation | None) -> dict[str, Any] | None:
        if raw_obs is None:
            return None
        return {
            "visual_rgb": tuple(np.asarray(raw_obs.visual_data["rgb"]).shape),
            "visual_depth": tuple(np.asarray(raw_obs.visual_data["depth"]).shape),
            "visual_seg": tuple(np.asarray(raw_obs.visual_data["seg"]).shape),
            "tactile_rgb": tuple(np.asarray(raw_obs.tactile_data["rgb"]).shape),
            "tactile_depth": tuple(np.asarray(raw_obs.tactile_data["depth"]).shape),
        }
