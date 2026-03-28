from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.structures.action import GraspPose, NormalizedAction
from src.structures.info import StepInfo
from src.structures.observation import Observation

from src.envs.pybullet_scene import PyBulletScene
from src.envs.observation_builder import ObservationBuilder


class GraspRefineEnv:
    """Single-step grasp refinement environment."""

    def __init__(
        self,
        cfg: dict,
        scene,
        action_executor,
        observation_builder,
        reward_manager,
        calibrator,
        termination,
        sample_provider=None,
    ):
        self.cfg = cfg
        self.scene: PyBulletScene = scene
        self.action_executor = action_executor
        self.observation_builder: ObservationBuilder = observation_builder
        self.reward_manager = reward_manager
        self.calibrator = calibrator
        self.termination = termination
        self.sample_provider = sample_provider
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.obs_before: Observation | None = None
        self.grasp_pose_before: GraspPose | None = None
        self.sample_cfg: dict | None = None
        self.raw_obs_before = None
        self.raw_obs_after = None
        self.max_reset_attempts = int(cfg.get("max_reset_attempts", 32))

    def reset(self) -> Observation:
        last_error: Exception | None = None
        for _ in range(self.max_reset_attempts):
            try:
                self.sample_cfg = self._sample_initial_state()
                self.scene.reset_scene(self.sample_cfg)
                self.grasp_pose_before = self._get_initial_grasp_pose(self.sample_cfg)
                self.scene.set_initial_grasp(self.grasp_pose_before)
                raw_obs_before = self.scene.get_raw_observation()
                self.raw_obs_before = raw_obs_before
                self.raw_obs_after = None
                self.obs_before = self.observation_builder.build(raw_obs_before, self.grasp_pose_before)
                return self.obs_before
            except Exception as exc:
                # print(f"[GraspRefineEnv.reset] reset attempt failed: {exc}")
                last_error = exc
                continue

        raise RuntimeError(f"Failed to reset environment after {self.max_reset_attempts} attempts: {last_error}")

    def step(self, action: NormalizedAction):
        if self.obs_before is None or self.grasp_pose_before is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if not isinstance(action, NormalizedAction):
            action = NormalizedAction(value=np.asarray(action, dtype=np.float32))

        physical_action = self.action_executor.decode(action)
        refined_pose = self.action_executor.apply_to_pose(self.grasp_pose_before, physical_action)
        self.scene.apply_refinement(refined_pose)
        raw_obs_after = self.scene.get_raw_observation()
        self.raw_obs_after = raw_obs_after
        obs_after = self.observation_builder.build(raw_obs_after, refined_pose)
        trial_result = self.scene.run_grasp_trial()

        calibrated_before = self.calibrator.predict(self.obs_before.raw_stability_logit)
        calibrated_after = self.calibrator.predict(obs_after.raw_stability_logit)
        posterior_trace = self.calibrator.posterior_trace()
        reward_breakdown = self.reward_manager.compute(
            drop_success=trial_result["drop_success"],
            calibrated_before=calibrated_before,
            calibrated_after=calibrated_after,
            posterior_trace=posterior_trace,
            contact_after=obs_after.contact_semantic,
        )
        done = self.termination.is_done()
        source_cfg = dict((self.sample_cfg or {}).get("source", {}))
        info = self._build_step_info(
            drop_success=trial_result["drop_success"],
            reward_breakdown=reward_breakdown,
            calibrated_before=calibrated_before,
            calibrated_after=calibrated_after,
            posterior_trace=posterior_trace,
            extra={
                "reward_breakdown": reward_breakdown,
                "raw_logit_before": self.obs_before.raw_stability_logit,
                "raw_logit_after": obs_after.raw_stability_logit,
                "posterior_trace": posterior_trace,
                "trial_metadata": trial_result.get("trial_metadata", {}),
                "legacy_drop_success_before": source_cfg.get("legacy_drop_success"),
                "source_object_id": source_cfg.get("object_id"),
                "source_global_id": source_cfg.get("global_id"),
            },
        )
        self.obs_before = obs_after
        self.raw_obs_before = raw_obs_after
        self.grasp_pose_before = refined_pose
        return obs_after, reward_breakdown.total, done, info

    def _sample_initial_state(self) -> dict:
        if self.sample_provider is not None:
            return self.sample_provider.sample()

        sample_cfg = deepcopy(self.cfg.get("default_sample_cfg", {}))
        if not sample_cfg:
            raise RuntimeError(
                "No dataset sample provider is configured and env cfg has no default_sample_cfg. "
                "Use a dataset-backed env config for formal training, or provide a debug fallback env config."
            )
        target_pose = sample_cfg["target_grasp_pose"]
        sampling_cfg = self.cfg.get("sampling", {})
        position_noise = np.asarray(sampling_cfg.get("position_noise", [0.015, 0.015, 0.015]), dtype=np.float32)
        rotation_noise = np.asarray(sampling_cfg.get("rotation_noise", [0.12, 0.12, 0.12]), dtype=np.float32)
        initial_position = np.asarray(target_pose["position"], dtype=np.float32) + self.rng.uniform(
            low=-position_noise,
            high=position_noise,
            size=3,
        )
        initial_rotation = np.asarray(target_pose["rotation"], dtype=np.float32) + self.rng.uniform(
            low=-rotation_noise,
            high=rotation_noise,
            size=3,
        )
        sample_cfg["initial_grasp_pose"] = {
            "position": initial_position.tolist(),
            "rotation": initial_rotation.tolist(),
        }
        return sample_cfg

    def _get_initial_grasp_pose(self, sample_cfg: dict) -> GraspPose:
        initial_cfg = sample_cfg["initial_grasp_pose"]
        return GraspPose(position=initial_cfg["position"], rotation=initial_cfg["rotation"])

    def _build_step_info(
        self,
        drop_success: int,
        reward_breakdown,
        calibrated_before: float,
        calibrated_after: float,
        posterior_trace: float,
        extra: dict,
    ) -> StepInfo:
        return StepInfo(
            drop_success=int(drop_success),
            calibrated_stability_before=float(calibrated_before),
            calibrated_stability_after=float(calibrated_after),
            posterior_trace=float(posterior_trace),
            reward_drop=float(reward_breakdown.drop),
            reward_stability=float(reward_breakdown.stability),
            reward_contact=float(reward_breakdown.contact),
            extra=extra,
        )

    def close(self) -> None:
        close_fn = getattr(self.scene, "close", None)
        if callable(close_fn):
            close_fn()

    def sync_calibrator(self, state: dict) -> None:
        load_state = getattr(self.calibrator, "load_state", None)
        if callable(load_state):
            load_state(state)

    def get_debug_snapshot(self) -> dict:
        scene_debug = {}
        scene_getter = getattr(self.scene, "get_debug_snapshot", None)
        if callable(scene_getter):
            scene_debug = scene_getter()
        return {
            "sample_source": None if self.sample_cfg is None else self.sample_cfg.get("source"),
            "grasp_pose_before": None if self.grasp_pose_before is None else self.grasp_pose_before.as_array().tolist(),
            "obs_before_logit": None if self.obs_before is None else float(self.obs_before.raw_stability_logit),
            "scene": scene_debug,
        }
