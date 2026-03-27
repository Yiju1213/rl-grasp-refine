from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from _common import build_env, load_experiment_bundle


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = float(target_height) / max(float(height), 1.0)
    target_width = max(int(round(width * scale)), 1)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


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
        normalized[valid] = np.round((depth[valid] - min_depth) * 255.0 / (max_depth - min_depth)).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _seg_to_display(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.int32)
    seg_offset = seg - int(seg.min())
    seg_norm = np.zeros_like(seg_offset, dtype=np.uint8)
    max_value = int(seg_offset.max()) if seg_offset.size else 0
    if max_value > 0:
        seg_norm = np.round(seg_offset.astype(np.float32) * (255.0 / max_value)).astype(np.uint8)
    return cv2.applyColorMap(seg_norm, cv2.COLORMAP_TURBO)


def _build_raw_obs_panel(stage_label: str, raw_obs) -> np.ndarray:
    visual = raw_obs.visual_data
    tactile = raw_obs.tactile_data

    visual_panels = [
        _label_panel(cv2.cvtColor(np.asarray(visual["rgb"], dtype=np.uint8), cv2.COLOR_RGB2BGR), f"{stage_label} visual rgb"),
        _label_panel(_normalize_depth_for_display(visual["depth"]), f"{stage_label} visual depth"),
        _label_panel(_seg_to_display(visual["seg"]), f"{stage_label} visual seg"),
    ]
    tactile_rgb = np.asarray(tactile["rgb"], dtype=np.uint8)
    tactile_depth = np.asarray(tactile["depth"], dtype=np.float32)
    tactile_panels = [
        _label_panel(cv2.cvtColor(tactile_rgb[0], cv2.COLOR_RGB2BGR), f"{stage_label} tactile left rgb"),
        _label_panel(_normalize_depth_for_display(tactile_depth[0]), f"{stage_label} tactile left depth"),
        _label_panel(cv2.cvtColor(tactile_rgb[1], cv2.COLOR_RGB2BGR), f"{stage_label} tactile right rgb"),
        _label_panel(_normalize_depth_for_display(tactile_depth[1]), f"{stage_label} tactile right depth"),
    ]

    top_height = max(panel.shape[0] for panel in visual_panels)
    bottom_height = max(panel.shape[0] for panel in tactile_panels)
    top_row = np.concatenate([_resize_to_height(panel, top_height) for panel in visual_panels], axis=1)
    bottom_row = np.concatenate([_resize_to_height(panel, bottom_height) for panel in tactile_panels], axis=1)
    target_width = max(top_row.shape[1], bottom_row.shape[1])
    if top_row.shape[1] != target_width:
        top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, target_width - top_row.shape[1], cv2.BORDER_CONSTANT, value=0)
    if bottom_row.shape[1] != target_width:
        bottom_row = cv2.copyMakeBorder(
            bottom_row,
            0,
            0,
            0,
            target_width - bottom_row.shape[1],
            cv2.BORDER_CONSTANT,
            value=0,
        )
    return np.concatenate([top_row, bottom_row], axis=0)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _seg_to_display(seg))


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
    raw_obs_panel = _build_raw_obs_panel(stage_dir.name, raw_obs)
    cv2.imwrite(str(stage_dir / "raw_obs_panel.png"), raw_obs_panel)

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
        "raw_obs_panel": str((stage_dir / "raw_obs_panel.png").resolve()),
    }


def _maybe_show_open3d(prepared_inputs_list) -> None:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(f"open3d is required for --show-open3d, but import failed: {exc}") from exc

    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0.0, 0.0, 0.0])]
    colors = {
        "visual": np.asarray([0.85, 0.20, 0.20], dtype=np.float64),
        "left_full": np.asarray([0.55, 0.72, 0.95], dtype=np.float64),
        "right_full": np.asarray([0.55, 0.90, 0.65], dtype=np.float64),
        "left_contact": np.asarray([0.10, 0.35, 0.90], dtype=np.float64),
        "right_contact": np.asarray([0.05, 0.55, 0.20], dtype=np.float64),
    }
    for prepared_inputs in prepared_inputs_list:
        for points, color in (
            (prepared_inputs.debug_visual_world_points, colors["visual"]),
            (prepared_inputs.debug_tactile_left_world_points, colors["left_full"]),
            (prepared_inputs.debug_tactile_right_world_points, colors["right_full"]),
            (prepared_inputs.debug_tactile_left_contact_world_points, colors["left_contact"]),
            (prepared_inputs.debug_tactile_right_contact_world_points, colors["right_contact"]),
        ):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            cloud.paint_uniform_color(color.tolist())
            geometries.append(cloud)
    o3d.visualization.draw_geometries(geometries)


def _maybe_show_raw_obs(raw_obs_entries: list[tuple[str, object]]) -> None:
    try:
        for stage_label, raw_obs in raw_obs_entries:
            panel = _build_raw_obs_panel(stage_label, raw_obs)
            window_name = f"raw_obs_{stage_label}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, panel)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


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


def _capture_round(
    env,
    runtime,
    adapter,
    round_dir: Path,
    *,
    capture_after: bool,
) -> tuple[dict, list[tuple[str, object]], list[object]]:
    exported = {}
    prepared_inputs_list = []
    raw_obs_entries: list[tuple[str, object]] = []

    obs_before = env.reset()
    raw_before = env.raw_obs_before
    before_result = runtime.infer(raw_before, adapter)
    prepared_inputs_list.append(before_result.prepared_inputs)
    raw_obs_entries.append(("before", raw_before))
    exported["before"] = {
        "raw_logit": float(obs_before.raw_stability_logit),
        **_export_stage(round_dir / "before", raw_before, before_result.prepared_inputs),
    }

    if capture_after:
        zero_action = np.zeros(6, dtype=np.float32)
        obs_after, reward, done, info = env.step(zero_action)
        raw_after = env.raw_obs_after
        after_result = runtime.infer(raw_after, adapter)
        prepared_inputs_list.append(after_result.prepared_inputs)
        raw_obs_entries.append(("after", raw_after))
        exported["after"] = {
            "raw_logit": float(obs_after.raw_stability_logit),
            "reward": float(reward),
            "done": bool(done),
            "drop_success": int(info.drop_success),
            **_export_stage(round_dir / "after", raw_after, after_result.prepared_inputs),
        }

    return exported, raw_obs_entries, prepared_inputs_list


def _show_requested_views(
    *,
    show_raw_obs: bool,
    show_open3d: bool,
    raw_obs_entries: list[tuple[str, object]],
    prepared_inputs_list: list[object],
) -> None:
    if show_raw_obs:
        _maybe_show_raw_obs(raw_obs_entries)
    if show_open3d:
        _maybe_show_open3d(prepared_inputs_list)


def _prompt_next_round() -> bool:
    while True:
        command = input("Next round? [Enter/n=continue, q=quit]: ").strip().lower()
        if command in {"", "n"}:
            return True
        if command == "q":
            return False
        print(f"Unknown command: {command!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture live env observations and export SGA-GSN reconstructions.")
    parser.add_argument("--experiment", default="configs/experiment/exp_debug.yaml")
    parser.add_argument("--output-dir", default="outputs/live_capture_reconstruction")
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--capture-after", action="store_true")
    parser.add_argument("--show-raw-obs", action="store_true")
    parser.add_argument("--show-open3d", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Run multiple rounds and prompt after each round.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    env = _build_env_for_demo(args.experiment, use_gui=args.use_gui)
    runtime, adapter = _require_sga_runtime(env)

    exported = {}
    try:
        if args.interactive:
            round_index = 0
            while True:
                round_dir = output_dir / f"round_{round_index:03d}"
                round_payload, raw_obs_entries, prepared_inputs_list = _capture_round(
                    env,
                    runtime,
                    adapter,
                    round_dir,
                    capture_after=args.capture_after,
                )
                round_key = f"round_{round_index:03d}"
                exported[round_key] = round_payload
                _show_requested_views(
                    show_raw_obs=args.show_raw_obs,
                    show_open3d=args.show_open3d,
                    raw_obs_entries=raw_obs_entries,
                    prepared_inputs_list=prepared_inputs_list,
                )
                print(json.dumps({round_key: round_payload}, indent=2, sort_keys=True))
                if not _prompt_next_round():
                    break
                round_index += 1
        else:
            exported, raw_obs_entries, prepared_inputs_list = _capture_round(
                env,
                runtime,
                adapter,
                output_dir,
                capture_after=args.capture_after,
            )
            _show_requested_views(
                show_raw_obs=args.show_raw_obs,
                show_open3d=args.show_open3d,
                raw_obs_entries=raw_obs_entries,
                prepared_inputs_list=prepared_inputs_list,
            )
    finally:
        env.close()

    print(json.dumps(exported, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
