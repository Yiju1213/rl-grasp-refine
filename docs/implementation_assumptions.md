# Implementation Assumptions

This file records implementation details that were not fully specified in `docs/v1.md`.
Each item is chosen to keep the v1 pipeline runnable without changing the public module boundaries.

## Data Contracts

- `Observation.grasp_pose` is implemented as `GraspPose`, not a raw numpy array.
- `Transition.reward` stores the scalar PPO reward. The structured reward breakdown lives in `StepInfo.extra["reward_breakdown"]`.
- `StepInfo.extra` always carries `reward_breakdown`, `raw_logit_before`, `raw_logit_after`, and `trial_metadata`.

## Action and Pose Handling

- The normalized action is ordered as `[dx, dy, dz, rx, ry, rz]`.
- Translation deltas are bounded per axis. Rotation deltas use a rotation-vector representation and are composed with the current pose via `scipy.spatial.transform.Rotation`.
- The actor samples from a Gaussian policy and clips the sampled action to `[-1, 1]` before dispatch. Log-probabilities are evaluated on the clipped action for consistency in the stored rollout.

## Environment and Scene

- `PyBulletScene` uses a lightweight, self-contained scene setup. It connects to PyBullet when available, but the grasp success signal is still a controlled heuristic based on distance to a target grasp pose so the project remains runnable without external robot assets.
- The minimum `sample_cfg` schema is:
  - `object_name`
  - `object_pose.position`
  - `object_pose.rotation`
  - `target_grasp_pose.position`
  - `target_grasp_pose.rotation`
  - optional `initial_grasp_pose`
  - optional `trial.max_position_error`
  - optional `trial.max_rotation_error`
- The default environment samples initial grasps by adding bounded noise around the target pose.

## Reward and Calibration

- Reward defaults to:
  - `R_total = w_drop * R_drop + w_stability * R_stability + w_contact * R_contact`
  - `R_drop(success=1) = +1.0`, `R_drop(success=0) = -1.0`
  - `R_stability = clip((p_after - p_before) - alpha * max(0, u_after - u_before), -1.0, 1.0)`
  - `R_contact = clip((coverage_after - coverage_before) - beta * (edge_after - edge_before), -0.25, 0.25)`
- The online calibrator updates only on `raw_logit_after -> drop_success`.
- Calibrator uncertainty is approximated from a two-parameter logistic calibration fit and a running posterior covariance estimate.

## Parallel Environments

- `DummyVecEnvWrapper` is synchronous and resets all wrapped envs together.
- `Trainer.collect_rollout()` supports both a single env and the dummy vector wrapper, but v1 defaults to `num_envs=1`.

## Testing

- Automated tests use fake scenes and do not depend on real PyBullet behavior.
- Real PyBullet usage is limited to smoke/debug scripts.
