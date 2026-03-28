from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from copy import deepcopy
from types import MethodType

import cv2
import numpy as np
import pybullet as pb

from _common import build_env, load_experiment_bundle
from src.envs.scene_assets import (
    attach_object_to_tacto_sensor,
    remove_object_from_tacto_sensor,
    set_object_body_collision_enabled,
    spawn_object,
)
from src.structures.action import NormalizedAction
from src.envs.grasp_refine_env import GraspRefineEnv
from src.utils.seed import set_seed


def _build_env_from_args(
    experiment_path: str,
    *,
    use_gui: bool,
    visualize_tacto_gui: bool,
    visualize_debug_windows: bool,
):
    experiment_cfg, config_bundle = load_experiment_bundle(experiment_path)
    set_seed(int(experiment_cfg.get("seed", 0)))
    env_cfg = deepcopy(config_bundle["env"])
    perception_cfg = config_bundle["perception"]
    calibration_cfg = config_bundle["calibration"]
    env_cfg.setdefault("scene", {})
    env_cfg["scene"]["use_gui"] = bool(use_gui)
    env_cfg["scene"]["visualize_tacto_gui"] = bool(visualize_tacto_gui)
    env_cfg["scene"]["visualize_debug_windows"] = bool(visualize_debug_windows)
    return build_env(env_cfg, perception_cfg, calibration_cfg)[0]


def _print_payload(title: str, payload: dict) -> None:
    print(f"\n[{title}]")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _apply_gui_env(display: str | None, xauthority: str | None) -> None:
    if display:
        os.environ["DISPLAY"] = display
    if xauthority:
        os.environ["XAUTHORITY"] = os.path.expanduser(xauthority)


def _gui_env_payload() -> dict:
    return {
        "DISPLAY": os.environ.get("DISPLAY"),
        "XAUTHORITY": os.environ.get("XAUTHORITY"),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
        "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
    }


def _preload_slot_pose(slot_index: int) -> dict[str, list[float]]:
    cols = 10
    row, col = divmod(int(slot_index), cols)
    return {
        "position": [1.5 + 0.35 * float(col), -2.5 - 0.35 * float(row), -1.5],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
    }


def _set_body_pose(body_id: int, pose_world: dict, client_id: int) -> None:
    pb.resetBasePositionAndOrientation(
        int(body_id),
        pose_world["position"],
        pose_world["quaternion"],
        physicsClientId=int(client_id),
    )
    pb.resetBaseVelocity(
        int(body_id),
        linearVelocity=[0.0, 0.0, 0.0],
        angularVelocity=[0.0, 0.0, 0.0],
        physicsClientId=int(client_id),
    )


def _stash_preloaded_body(scene, object_id: int) -> None:
    body = scene._sanity_preloaded_object_bodies[int(object_id)]
    stash_pose = scene._sanity_preloaded_object_stash_poses[int(object_id)]
    _set_body_pose(body.id, stash_pose, scene.client_id)
    set_object_body_collision_enabled(body.id, scene.client_id, enabled=False)


def _activate_preloaded_body(scene, object_body, object_pose_world: dict) -> None:
    _set_body_pose(object_body.id, object_pose_world, scene.client_id)
    set_object_body_collision_enabled(object_body.id, scene.client_id, enabled=True)


def _install_preload_all_objects_mode(env: GraspRefineEnv) -> None:
    scene = getattr(env, "scene", None)
    provider = getattr(env, "sample_provider", None)
    if scene is None:
        raise RuntimeError("Environment has no scene; cannot enable preload-all-objects mode.")
    if provider is None or not hasattr(provider, "_object_entries"):
        raise RuntimeError("Sample provider does not expose object entries; cannot preload objects from script.")

    preload_object_ids = sorted(int(object_id) for object_id in provider._object_entries.keys())
    if not preload_object_ids:
        raise RuntimeError("No object ids available for preload-all-objects mode.")

    def _preload_all_objects(self) -> None:
        if getattr(self, "_sanity_preloaded_object_bodies", None) is not None:
            return

        start_time = time.perf_counter()
        self._sanity_preloaded_object_bodies = {}
        self._sanity_preloaded_object_stash_poses = {}
        self._sanity_preloaded_object_ids = list(preload_object_ids)

        self._set_gui_rendering_enabled(False)
        try:
            for slot_index, object_id in enumerate(preload_object_ids):
                stash_pose = _preload_slot_pose(slot_index)
                body = spawn_object(self.asset_paths, object_id=object_id, object_pose_world=stash_pose)
                set_object_body_collision_enabled(body.id, self.client_id, enabled=False)
                self._sanity_preloaded_object_bodies[int(object_id)] = body
                self._sanity_preloaded_object_stash_poses[int(object_id)] = stash_pose
        finally:
            self._set_gui_rendering_enabled(True)
            self._request_gui_render()

        elapsed_s = time.perf_counter() - start_time
        self._sanity_preload_stats = {
            "enabled": True,
            "object_count": len(preload_object_ids),
            "elapsed_s": float(elapsed_s),
            "stash_mode": "offscreen_no_alpha_toggle_no_rgba_override",
        }
        print(f"[PRELOAD] loaded {len(preload_object_ids)} objects in {elapsed_s:.2f}s")

    def _ensure_sample_object_preloaded(self):
        self._preload_all_objects()

        target_object_id = int(self.sample_cfg["source"]["object_id"])
        target_body = self._sanity_preloaded_object_bodies[target_object_id]
        target_pose = self.sample_cfg["pre_grasp"]["object_pose_world"]
        current_object_id = self._loaded_source_object_id
        object_action = "reuse"

        self._set_gui_rendering_enabled(False)
        try:
            if self.object_body is None:
                _activate_preloaded_body(self, target_body, target_pose)
                attach_object_to_tacto_sensor(self.tacto_sensor, target_body)
                self.object_body = target_body
                self._loaded_source_object_id = target_object_id
                object_action = "spawn"
            elif current_object_id != target_object_id:
                remove_object_from_tacto_sensor(self.tacto_sensor, self.object_body.id)
                _stash_preloaded_body(self, current_object_id)
                _activate_preloaded_body(self, target_body, target_pose)
                attach_object_to_tacto_sensor(self.tacto_sensor, target_body)
                self.object_body = target_body
                self._loaded_source_object_id = target_object_id
                object_action = "swap"
            else:
                _activate_preloaded_body(self, target_body, target_pose)
                self.object_body = target_body
        finally:
            self._set_gui_rendering_enabled(True)
            self._request_gui_render()

        self._last_object_action = object_action
        return object_action

    scene._preload_all_objects = MethodType(_preload_all_objects, scene)
    scene._ensure_sample_object = MethodType(_ensure_sample_object_preloaded, scene)
    scene._sanity_preload_mode = "all_objects"
    scene._sanity_preloaded_object_bodies = None
    scene._sanity_preloaded_object_stash_poses = None
    scene._sanity_preload_stats = {
        "enabled": True,
        "object_count": len(preload_object_ids),
        "elapsed_s": None,
        "stash_mode": "offscreen_no_alpha_toggle_no_rgba_override",
    }


def _tick_pybullet_gui(env: GraspRefineEnv) -> None:
    scene = getattr(env, "scene", None)
    if scene is None or not bool(getattr(scene, "cfg", {}).get("use_gui", False)):
        return
    client_id = getattr(scene, "client_id", None)
    if client_id is not None and getattr(pb, "COV_ENABLE_SINGLE_STEP_RENDERING", None) is not None:
        pb.configureDebugVisualizer(
            pb.COV_ENABLE_SINGLE_STEP_RENDERING,
            1,
            physicsClientId=client_id,
        )


def _focus_pybullet_camera(env: GraspRefineEnv) -> None:
    scene = getattr(env, "scene", None)
    if scene is None or not bool(getattr(scene, "cfg", {}).get("use_gui", False)):
        return
    hand = getattr(scene, "hand", None)
    object_body = getattr(scene, "object_body", None)
    client_id = getattr(scene, "client_id", None)
    if hand is None or object_body is None or client_id is None:
        return

    hand_pos, _ = pb.getBasePositionAndOrientation(hand.id, physicsClientId=client_id)
    object_pos, _ = pb.getBasePositionAndOrientation(object_body.id, physicsClientId=client_id)
    target = [
        0.5 * (float(hand_pos[0]) + float(object_pos[0])),
        0.5 * (float(hand_pos[1]) + float(object_pos[1])),
        0.5 * (float(hand_pos[2]) + float(object_pos[2])),
    ]
    pb.resetDebugVisualizerCamera(
        cameraDistance=0.35,
        cameraYaw=45.0,
        cameraPitch=-35.0,
        cameraTargetPosition=target,
        physicsClientId=client_id,
    )


def _clear_object_overlay(scene) -> None:
    debug_ids = getattr(scene, "_sanity_debug_item_ids", [])
    client_id = getattr(scene, "client_id", None)
    for debug_id in debug_ids:
        pb.removeUserDebugItem(int(debug_id), physicsClientId=client_id)
    scene._sanity_debug_item_ids = []


def _draw_object_overlay(env: GraspRefineEnv) -> None:
    scene = getattr(env, "scene", None)
    if scene is None or not bool(getattr(scene, "cfg", {}).get("use_gui", False)):
        return
    object_body = getattr(scene, "object_body", None)
    client_id = getattr(scene, "client_id", None)
    if object_body is None or client_id is None:
        return

    _clear_object_overlay(scene)
    aabb_min, aabb_max = pb.getAABB(object_body.id, physicsClientId=client_id)
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    debug_ids = []
    for start_idx, end_idx in edges:
        debug_ids.append(
            pb.addUserDebugLine(
                corners[start_idx],
                corners[end_idx],
                lineColorRGB=[1.0, 0.1, 0.1],
                lineWidth=2.0,
                lifeTime=0.0,
                physicsClientId=client_id,
            )
        )
    label_pos = [(x0 + x1) * 0.5, (y0 + y1) * 0.5, z1 + 0.03]
    debug_ids.append(
        pb.addUserDebugText(
            text=f"obj {object_body.id}",
            textPosition=label_pos,
            textColorRGB=[1.0, 0.1, 0.1],
            textSize=1.2,
            lifeTime=0.0,
            physicsClientId=client_id,
        )
    )
    scene._sanity_debug_item_ids = debug_ids


def _show_visual_observation(env: GraspRefineEnv) -> None:
    scene = getattr(env, "scene", None)
    if scene is None or not bool(getattr(scene, "cfg", {}).get("visualize_debug_windows", False)):
        return
    raw_obs_getter = getattr(scene, "get_raw_observation", None)
    if not callable(raw_obs_getter):
        return

    raw_obs = raw_obs_getter()
    visual_data = getattr(raw_obs, "visual_data", None)
    tactile_data = getattr(raw_obs, "tactile_data", None)
    if not isinstance(visual_data, dict) or not isinstance(tactile_data, dict):
        return

    visual_rgb = np.asarray(visual_data.get("rgb"))
    visual_depth = np.asarray(visual_data.get("depth"))
    visual_seg = np.asarray(visual_data.get("seg"))
    tactile_rgb = np.asarray(tactile_data.get("rgb"))
    tactile_depth = np.asarray(tactile_data.get("depth"))
    if visual_rgb.size == 0 or visual_depth.size == 0 or visual_seg.size == 0:
        return
    if tactile_rgb.size == 0 or tactile_depth.size == 0:
        return
    if visual_rgb.ndim != 3 or visual_rgb.shape[-1] != 3:
        return
    if tactile_rgb.ndim != 4 or tactile_rgb.shape[0] < 2 or tactile_rgb.shape[-1] != 3:
        return
    if tactile_depth.ndim != 3 or tactile_depth.shape[0] < 2:
        return

    def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
        panel = np.asarray(image, dtype=np.uint8).copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (20, 20, 20), thickness=-1)
        cv2.putText(
            panel,
            label,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _normalize_depth_for_display(depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        min_depth = float(np.min(depth[valid]))
        max_depth = float(np.max(depth[valid]))
        if max_depth - min_depth < 1e-8:
            normalized = np.zeros_like(depth, dtype=np.uint8)
            normalized[valid] = 255
        else:
            normalized = np.zeros_like(depth, dtype=np.uint8)
            normalized[valid] = np.round((depth[valid] - min_depth) * 255.0 / (max_depth - min_depth)).astype(
                np.uint8
            )
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return colored

    def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        height, width = image.shape[:2]
        if height == target_height:
            return image
        scale = float(target_height) / max(float(height), 1.0)
        target_width = max(int(round(width * scale)), 1)
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    seg_vis = np.zeros((*visual_seg.shape, 3), dtype=np.uint8)
    grasp_metadata = getattr(raw_obs, "grasp_metadata", {}) or {}
    segmentation_ids = grasp_metadata.get("segmentation_ids", {})
    hand_seg_id = segmentation_ids.get("hand", -9999)
    object_seg_id = segmentation_ids.get("object", -9999)
    seg_vis[visual_seg == int(hand_seg_id)] = np.array([40, 80, 255], dtype=np.uint8)
    seg_vis[visual_seg == int(object_seg_id)] = np.array([255, 80, 40], dtype=np.uint8)

    visual_panels = [
        _label_panel(cv2.cvtColor(visual_rgb, cv2.COLOR_RGB2BGR), "visual rgb"),
        _label_panel(_normalize_depth_for_display(visual_depth), "visual depth"),
        _label_panel(seg_vis, "visual seg"),
    ]
    visual_target_height = max(panel.shape[0] for panel in visual_panels)
    visual_window = np.concatenate(
        [_resize_to_height(panel, visual_target_height) for panel in visual_panels],
        axis=1,
    )

    tactile_panels = [
        _label_panel(cv2.cvtColor(tactile_rgb[0], cv2.COLOR_RGB2BGR), "tactile left rgb"),
        _label_panel(_normalize_depth_for_display(tactile_depth[0]), "tactile left depth"),
        _label_panel(cv2.cvtColor(tactile_rgb[1], cv2.COLOR_RGB2BGR), "tactile right rgb"),
        _label_panel(_normalize_depth_for_display(tactile_depth[1]), "tactile right depth"),
    ]
    tactile_target_height = max(panel.shape[0] for panel in tactile_panels)
    tactile_window = np.concatenate(
        [_resize_to_height(panel, tactile_target_height) for panel in tactile_panels],
        axis=1,
    )

    cv2.imshow("raw_obs_visual", visual_window)
    cv2.imshow("raw_obs_tactile", tactile_window)


def _refresh_gui_windows(env: GraspRefineEnv) -> None:
    scene = getattr(env, "scene", None)
    if scene is None:
        return
    if bool(getattr(scene, "cfg", {}).get("use_gui", False)):
        _focus_pybullet_camera(env)
        _draw_object_overlay(env)
    _show_visual_observation(env)
    tacto_sensor = getattr(scene, "tacto_sensor", None)
    if tacto_sensor is not None and bool(getattr(scene, "cfg", {}).get("visualize_tacto_gui", False)):
        gels_color, gels_depth = tacto_sensor.render()
        tacto_sensor.updateGUI(gels_color, gels_depth)
    _tick_pybullet_gui(env)


def _poll_interactive_command(env: GraspRefineEnv) -> str | None:
    scene = getattr(env, "scene", None)
    if scene is not None and bool(getattr(scene, "cfg", {}).get("use_gui", False)):
        events = pb.getKeyboardEvents()
        for key, command in ((ord("q"), "q"), (ord("n"), "n"), (ord("r"), "r")):
            if key in events and events[key] & pb.KEY_WAS_TRIGGERED:
                return command

    if scene is not None and bool(getattr(scene, "cfg", {}).get("visualize_debug_windows", False)):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("n"), ord("r")):
            return chr(key)

    if sys.stdin.isatty():
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            line = sys.stdin.readline().strip().lower()
            if line in {"q", "n", "r"}:
                return line
            if line:
                print(f"Ignoring unknown command from stdin: {line!r}")
    return None


def _reset_payload(env, obs) -> dict:
    debug = env.get_debug_snapshot()
    scene_debug = debug.get("scene", {})
    scene = getattr(env, "scene", None)
    return {
        "source_object_id": scene_debug.get("source_object_id"),
        "source_global_id": scene_debug.get("source_global_id"),
        "asset_paths": scene_debug.get("asset_paths"),
        "before_shapes": scene_debug.get("before_shapes"),
        "pre_grasp": scene_debug.get("pre_grasp"),
        "reset_debug": scene_debug.get("reset_debug"),
        "runtime_counters": scene_debug.get("runtime_counters"),
        "obs_before_logit": float(obs.raw_stability_logit),
        "reset_ready": scene_debug.get("reset_debug", {}).get("trial_status") == "success",
        "preload_debug": None if scene is None else getattr(scene, "_sanity_preload_stats", None),
    }


def _step_payload(env, action, obs_before, obs_after, reward, done, info) -> dict:
    debug = env.get_debug_snapshot()
    scene_debug = debug.get("scene", {})
    return {
        "action": np.asarray(action.value, dtype=np.float32).tolist(),
        "refined_grasp_pose": None if obs_after.grasp_pose is None else obs_after.grasp_pose.as_array().tolist(),
        "refine_debug": scene_debug.get("refine_debug"),
        "after_shapes": scene_debug.get("after_shapes"),
        "after_observation_valid": scene_debug.get("after_observation_valid"),
        "trial_status": info.extra["trial_metadata"].get("trial_status"),
        "release_executed": info.extra["trial_metadata"].get("release_executed"),
        "valid_for_learning": info.extra["trial_metadata"].get("valid_for_learning"),
        "drop_success": info.drop_success,
        "reward": float(reward),
        "reward_breakdown": info.extra["reward_breakdown"].as_dict(),
        "raw_logit_before": float(obs_before.raw_stability_logit),
        "raw_logit_after": float(obs_after.raw_stability_logit),
        "calibrated_before": float(info.calibrated_stability_before),
        "calibrated_after": float(info.calibrated_stability_after),
        "posterior_trace": float(info.posterior_trace),
        "done": bool(done),
    }


def _run_single_rollout(env: GraspRefineEnv) -> None:
    obs = env.reset()
    _refresh_gui_windows(env)
    _print_payload("RESET", _reset_payload(env, obs))
    action = NormalizedAction(value=np.random.uniform(-1.0, 1.0, size=6).astype(np.float32))
    next_obs, reward, done, info = env.step(action)
    _refresh_gui_windows(env)
    _print_payload("STEP", _step_payload(env, action, obs, next_obs, reward, done, info))


def _interactive_loop(env: GraspRefineEnv) -> None:
    obs = env.reset()
    episode_done = False
    _refresh_gui_windows(env)
    _print_payload("RESET", _reset_payload(env, obs))
    print("Interactive controls: press r/n/q in PyBullet or debug image windows, or type r/n/q then Enter in terminal.")

    while True:
        command = _poll_interactive_command(env)
        _tick_pybullet_gui(env)

        if command == "q":
            break
        if command == "n":
            obs = env.reset()
            _refresh_gui_windows(env)
            episode_done = False
            _print_payload("RESET", _reset_payload(env, obs))
        if command == "r":
            if episode_done:
                obs = env.reset()
                _refresh_gui_windows(env)
                episode_done = False
                _print_payload("RESET", _reset_payload(env, obs))
                print("Episode was done. Reset completed; press 'r' again to step.")
            else:
                action = NormalizedAction(value=np.random.uniform(-1.0, 1.0, size=6).astype(np.float32))
                next_obs, reward, done, info = env.step(action)
                _refresh_gui_windows(env)
                _print_payload("STEP", _step_payload(env, action, obs, next_obs, reward, done, info))
                obs = next_obs
                episode_done = bool(done)
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description="Run a random action through the environment.")
    parser.add_argument("--experiment", default="configs/experiment/exp_debug.yaml")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI.")
    parser.add_argument("--interactive", action="store_true", help="Enable keyboard-driven manual inspection.")
    parser.add_argument(
        "--preload-all-objects",
        action="store_true",
        help="For debugging only: preload all dataset objects once and swap by pose/collision instead of remove/spawn.",
    )
    parser.add_argument("--display", default=":1", help="Override DISPLAY before opening the PyBullet GUI.")
    parser.add_argument("--xauthority", help="Override XAUTHORITY before opening the PyBullet GUI.")
    args = parser.parse_args()

    use_pybullet_gui = bool(args.gui)
    visualize_tacto_gui = bool(args.gui)
    visualize_debug_windows = bool(args.gui or args.interactive)
    if visualize_debug_windows:
        _apply_gui_env(args.display, args.xauthority)
        _print_payload("GUI_ENV", _gui_env_payload())
    env = _build_env_from_args(
        args.experiment,
        use_gui=use_pybullet_gui,
        visualize_tacto_gui=visualize_tacto_gui,
        visualize_debug_windows=visualize_debug_windows,
    )
    if args.preload_all_objects:
        _install_preload_all_objects_mode(env)
    try:
        if args.interactive:
            _interactive_loop(env)
        else:
            _run_single_rollout(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
