from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from _common import build_env, load_experiment_bundle


def _write_point_cloud_ply(points: np.ndarray, path: Path) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def _write_depth_png(path: Path, depth_m: np.ndarray, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_uint16 = np.clip(np.asarray(depth_m, dtype=np.float32) * scale, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), depth_uint16)


def _write_seg_png(path: Path, seg: np.ndarray) -> None:
    seg = np.asarray(seg, dtype=np.int32)
    seg_offset = seg - int(seg.min())
    seg_norm = np.zeros_like(seg_offset, dtype=np.uint8)
    max_value = int(seg_offset.max()) if seg_offset.size else 0
    if max_value > 0:
        seg_norm = np.round(seg_offset.astype(np.float32) * (255.0 / max_value)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.applyColorMap(seg_norm, cv2.COLORMAP_TURBO))


def _export_stage(stage_dir: Path, raw_obs, prepared_inputs) -> dict[str, str]:
    stage_dir.mkdir(parents=True, exist_ok=True)

    visual = raw_obs.visual_data
    tactile = raw_obs.tactile_data

    _write_rgb(stage_dir / "visual_rgb.png", visual["rgb"])
    _write_depth_png(stage_dir / "visual_depth_mm.png", visual["depth"], scale=1000.0)
    _write_seg_png(stage_dir / "visual_seg.png", visual["seg"])

    tactile_rgb = np.asarray(tactile["rgb"], dtype=np.uint8)
    tactile_depth = np.asarray(tactile["depth"], dtype=np.float32)
    for index, side in enumerate(("left", "right")):
        _write_rgb(stage_dir / f"tactile_{side}_rgb.png", tactile_rgb[index])
        _write_depth_png(stage_dir / f"tactile_{side}_depth_1e4m.png", tactile_depth[index], scale=10000.0)

    _write_point_cloud_ply(prepared_inputs.debug_visual_world_points, stage_dir / "visual_world_points.ply")
    _write_point_cloud_ply(prepared_inputs.debug_tactile_left_world_points, stage_dir / "tactile_left_world_points.ply")
    _write_point_cloud_ply(prepared_inputs.debug_tactile_right_world_points, stage_dir / "tactile_right_world_points.ply")
    _write_point_cloud_ply(
        prepared_inputs.debug_tactile_left_contact_world_points,
        stage_dir / "tactile_left_contact_world_points.ply",
    )
    _write_point_cloud_ply(
        prepared_inputs.debug_tactile_right_contact_world_points,
        stage_dir / "tactile_right_contact_world_points.ply",
    )
    _write_point_cloud_ply(prepared_inputs.sc_input, stage_dir / "sc_input.ply")
    _write_point_cloud_ply(prepared_inputs.gs_input[:, :3], stage_dir / "gs_input_xyz.ply")

    np.savez(
        stage_dir / "capture_and_reconstruction.npz",
        visual_rgb=np.asarray(visual["rgb"], dtype=np.uint8),
        visual_depth=np.asarray(visual["depth"], dtype=np.float32),
        visual_seg=np.asarray(visual["seg"]),
        tactile_rgb=np.asarray(tactile["rgb"], dtype=np.uint8),
        tactile_depth=np.asarray(tactile["depth"], dtype=np.float32),
        sc_input=prepared_inputs.sc_input,
        gs_input=prepared_inputs.gs_input,
        zero_mean=prepared_inputs.zero_mean,
        visual_world_points=prepared_inputs.debug_visual_world_points,
        tactile_left_world_points=prepared_inputs.debug_tactile_left_world_points,
        tactile_right_world_points=prepared_inputs.debug_tactile_right_world_points,
        tactile_left_contact_world_points=prepared_inputs.debug_tactile_left_contact_world_points,
        tactile_right_contact_world_points=prepared_inputs.debug_tactile_right_contact_world_points,
        tactile_left_gel_mask=prepared_inputs.debug_tactile_left_gel_mask,
        tactile_right_gel_mask=prepared_inputs.debug_tactile_right_gel_mask,
    )

    return {
        "stage_dir": str(stage_dir.resolve()),
        "npz": str((stage_dir / "capture_and_reconstruction.npz").resolve()),
        "visual_ply": str((stage_dir / "visual_world_points.ply").resolve()),
        "sc_input_ply": str((stage_dir / "sc_input.ply").resolve()),
        "gs_input_ply": str((stage_dir / "gs_input_xyz.ply").resolve()),
    }


def _maybe_show_open3d(prepared_inputs_list) -> None:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(f"open3d is required for --show-open3d, but import failed: {exc}") from exc

    geometries = []
    colors = [
        np.asarray([0.9, 0.2, 0.2], dtype=np.float64),
        np.asarray([0.2, 0.6, 0.9], dtype=np.float64),
        np.asarray([0.2, 0.8, 0.4], dtype=np.float64),
    ]
    for prepared_inputs in prepared_inputs_list:
        for points, color in (
            (prepared_inputs.debug_visual_world_points, colors[0]),
            (prepared_inputs.debug_tactile_left_contact_world_points, colors[1]),
            (prepared_inputs.debug_tactile_right_contact_world_points, colors[2]),
        ):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            cloud.paint_uniform_color(color.tolist())
            geometries.append(cloud)
    o3d.visualization.draw_geometries(geometries)


def _build_env_for_demo(experiment_path: str, use_gui: bool):
    _, bundle = load_experiment_bundle(experiment_path)
    env_cfg = deepcopy(bundle["env"])
    perception_cfg = bundle["perception"]
    calibration_cfg = bundle["calibration"]
    env_cfg.setdefault("scene", {})
    env_cfg["scene"]["use_gui"] = bool(use_gui)
    env_cfg["scene"]["visualize_tacto_gui"] = bool(use_gui)
    return build_env(env_cfg, perception_cfg, calibration_cfg)[0]


def _require_sga_runtime(env):
    feature_extractor = env.observation_builder.feature_extractor
    runtime = getattr(feature_extractor, "runtime", None)
    if runtime is None:
        raise RuntimeError("Current perception config is not using the bridged SGA-GSN runtime.")
    return runtime, feature_extractor.adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture live env observations and export SGA-GSN reconstructions.")
    parser.add_argument("--experiment", default="configs/experiment/exp_debug.yaml")
    parser.add_argument("--output-dir", default="outputs/live_capture_reconstruction")
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--capture-after", action="store_true")
    parser.add_argument("--show-open3d", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    env = _build_env_for_demo(args.experiment, use_gui=args.use_gui)
    runtime, adapter = _require_sga_runtime(env)

    exported = {}
    prepared_inputs_list = []
    try:
        obs_before = env.reset()
        raw_before = env.raw_obs_before
        before_result = runtime.infer(raw_before, adapter)
        prepared_inputs_list.append(before_result.prepared_inputs)
        exported["before"] = {
            "raw_logit": float(obs_before.raw_stability_logit),
            **_export_stage(output_dir / "before", raw_before, before_result.prepared_inputs),
        }

        if args.capture_after:
            zero_action = np.zeros(6, dtype=np.float32)
            obs_after, reward, done, info = env.step(zero_action)
            raw_after = env.raw_obs_after
            after_result = runtime.infer(raw_after, adapter)
            prepared_inputs_list.append(after_result.prepared_inputs)
            exported["after"] = {
                "raw_logit": float(obs_after.raw_stability_logit),
                "reward": float(reward),
                "done": bool(done),
                "drop_success": int(info.drop_success),
                **_export_stage(output_dir / "after", raw_after, after_result.prepared_inputs),
            }
    finally:
        env.close()

    if args.show_open3d:
        _maybe_show_open3d(prepared_inputs_list)

    print(json.dumps(exported, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
