from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

PLOT_SCRIPTS_DIR = Path("/rl-grasp-refine/plot_scripts")
if str(PLOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_SCRIPTS_DIR))

import fig01_main_overall_performance as fig01
import fig02_main_risk_return as fig02
import fig03_risk_return_scatter as fig03
import fig04_mechanism_triplet as fig04
import fig06_object_stability_boxplot as fig06
import fig07_object_stability_bar as fig07
import fig08_per_object_rank_curve as fig08
import fig09_per_run_overlay as fig09
import fig10_full_training_curves as fig10
import plot_common
import plot_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = (1001, 1002, 1003)
OBJECT_IDS = (75, 76, 77)


def _summary_row(label: str, index: int) -> dict[str, float | str]:
    base = 0.04 * (index + 1)
    pos_drop = 0.12 + 0.01 * index
    neg_hold = 0.46 + 0.02 * index
    t_cover = 0.05 + 0.01 * index
    t_edge = 0.14 + 0.01 * index
    prob_delta = 0.02 + 0.005 * index
    return {
        "experiment_name": label,
        "evaluation_wall_minutes": 1.0 + index,
        "macro_success_lift_mean": base,
        "macro_success_lift_std": 0.01,
        "macro_success_lift_ci95_low": base - 0.02,
        "macro_success_lift_ci95_high": base + 0.02,
        "pos_drop_mean": pos_drop,
        "pos_drop_std": 0.01,
        "pos_drop_ci95_low": pos_drop - 0.02,
        "pos_drop_ci95_high": pos_drop + 0.02,
        "neg_hold_mean": neg_hold,
        "neg_hold_std": 0.01,
        "neg_hold_ci95_low": neg_hold - 0.02,
        "neg_hold_ci95_high": neg_hold + 0.02,
        "across_object_lift_std_mean": 0.08 + 0.01 * index,
        "across_object_lift_std_std": 0.01,
        "across_object_lift_iqr_mean": 0.11 + 0.01 * index,
        "across_object_lift_iqr_std": 0.01,
        "t_cover_delta_mean": t_cover,
        "t_cover_delta_std": 0.01,
        "t_cover_delta_ci95_low": t_cover - 0.01,
        "t_cover_delta_ci95_high": t_cover + 0.01,
        "t_edge_delta_mean": t_edge,
        "t_edge_delta_std": 0.01,
        "t_edge_delta_ci95_low": t_edge - 0.01,
        "t_edge_delta_ci95_high": t_edge + 0.01,
        "prob_delta_mean_mean": prob_delta,
        "prob_delta_mean_std": 0.005,
        "prob_delta_mean_ci95_low": prob_delta - 0.01,
        "prob_delta_mean_ci95_high": prob_delta + 0.01,
    }


def _per_run_rows(label: str, index: int) -> list[dict[str, float | int | str]]:
    base = 0.04 * (index + 1)
    rows = []
    for seed_index, seed in enumerate(TEST_SEEDS):
        rows.append(
            {
                "experiment_name": label,
                "test_seed": seed,
                "macro_success_lift": base + 0.01 * (seed_index - 1),
                "pos_drop_rate": 0.12 + 0.01 * index + 0.005 * seed_index,
                "neg_hold_rate": 0.46 + 0.02 * index + 0.01 * seed_index,
                "across_object_lift_std": 0.08 + 0.01 * index,
                "across_object_lift_iqr": 0.11 + 0.01 * index,
                "t_cover_delta_mean": 0.05 + 0.01 * index,
                "t_edge_delta_mean": 0.14 + 0.01 * index,
                "prob_delta_mean": 0.02 + 0.005 * index + 0.002 * seed_index,
                "num_objects": len(OBJECT_IDS),
                "total_episodes": len(OBJECT_IDS) * 100,
            }
        )
    return rows


def _per_object_rows(label: str, index: int) -> list[dict[str, float | int | str]]:
    base = 0.04 * (index + 1)
    object_offsets = {75: -0.02, 76: 0.0, 77: 0.02}
    rows = []
    for seed_index, seed in enumerate(TEST_SEEDS):
        seed_offset = 0.01 * (seed_index - 1)
        for object_id in OBJECT_IDS:
            rows.append(
                {
                    "experiment_name": label,
                    "test_seed": seed,
                    "object_id": object_id,
                    "success_lift_vs_dataset": base + object_offsets[object_id] + seed_offset,
                    "positive_drop_rate": 0.10 + 0.01 * index + 0.01 * seed_index,
                    "negative_hold_rate": 0.45 + 0.01 * index + 0.01 * seed_index,
                    "t_cover_delta_mean": 0.05 + 0.01 * index,
                    "t_edge_delta_mean": 0.14 + 0.01 * index,
                    "prob_delta_mean": 0.02 + 0.005 * index + 0.002 * seed_index,
                    "num_episodes": 100,
                    "positive_count": 40,
                    "negative_count": 60,
                    "positive_drop_count": 5 + seed_index,
                    "negative_hold_count": 25 + seed_index,
                }
            )
    return rows


def write_plot_fixture(root: Path, labels: tuple[str, ...] | list[str] | None = None) -> Path:
    selected_labels = tuple(labels or plot_config.ORDERED_LABELS)
    root.mkdir(parents=True, exist_ok=True)
    for index, label in enumerate(selected_labels):
        experiment_dir = root / label
        experiment_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([_summary_row(label, index)]).to_csv(experiment_dir / "summary.csv", index=False)
        pd.DataFrame(_per_run_rows(label, index)).to_csv(experiment_dir / "per_run_summary.csv", index=False)
        pd.DataFrame(_per_object_rows(label, index)).to_csv(experiment_dir / "per_object_summary.csv", index=False)
    return root


def write_metrics_fixture(
    run_dir: Path,
    *,
    steps: range,
    offset: float,
    validation_every: int = 2,
    include_validation: bool = True,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as handle:
        for step in steps:
            stats = {
                "outcome/success_lift_vs_dataset": offset + 0.1 * step,
            }
            if include_validation and step % validation_every == 0:
                stats["validation/outcome/success_lift_vs_dataset"] = offset + 0.2 * step
            handle.write(json.dumps({"timestamp": "2026-04-13T00:00:00+00:00", "step": step, "stats": stats}) + "\n")
    return run_dir


class TestPlotCommon(unittest.TestCase):
    def test_normalize_cli_args_appends_group_to_output_dir(self):
        parser = fig01.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = plot_common.normalize_cli_args(
                parser.parse_args(
                    [
                        "--root",
                        str(Path(tmpdir) / "formal"),
                        "--out-dir",
                        str(Path(tmpdir) / "generated"),
                        "--group",
                        "ablation",
                    ]
                )
            )
            self.assertEqual(args.out_dir, (Path(tmpdir) / "generated" / "ablation").resolve())

    def test_discover_experiment_dirs_finds_summary_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(
                Path(tmpdir),
                labels=("drop-only-latent-only-128-epi", "full-latefus-128-epi"),
            )
            discovered = plot_common.discover_experiment_dirs(root)
            self.assertEqual(set(discovered.keys()), {"drop-only-latent-only-128-epi", "full-latefus-128-epi"})

    def test_resolve_selected_labels_warns_and_orders_group_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(
                Path(tmpdir),
                labels=("full-latefus-128-epi", "drop-only-latent-only-128-epi"),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                selected = plot_common.resolve_selected_labels(root, group="main", labels=None)
            self.assertEqual(selected, ["drop-only-latent-only-128-epi", "full-latefus-128-epi"])
            self.assertTrue(caught)

    def test_explicit_labels_override_group_and_follow_config_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(
                Path(tmpdir),
                labels=("full-latefus-128-epi", "drop-only-latent-only-128-epi"),
            )
            selected = plot_common.resolve_selected_labels(
                root,
                group="main",
                labels=["full-latefus-128-epi", "drop-only-latent-only-128-epi"],
            )
            self.assertEqual(selected, ["drop-only-latent-only-128-epi", "full-latefus-128-epi"])

    def test_load_table_for_labels_reads_all_supported_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(
                Path(tmpdir),
                labels=("drop-only-latent-only-128-epi", "full-latefus-128-epi"),
            )
            labels = ["drop-only-latent-only-128-epi", "full-latefus-128-epi"]
            summary_frame = plot_common.load_table_for_labels(root, "summary.csv", labels)
            per_run_frame = plot_common.load_table_for_labels(root, "per_run_summary.csv", labels)
            per_object_frame = plot_common.load_table_for_labels(root, "per_object_summary.csv", labels)
            self.assertEqual(len(summary_frame), 2)
            self.assertEqual(len(per_run_frame), 6)
            self.assertEqual(len(per_object_frame), 18)

    def test_average_object_metric_across_seeds_groups_by_label_and_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir), labels=("drop-only-latent-only-128-epi",))
            per_object_frame = plot_common.load_table_for_labels(
                root,
                "per_object_summary.csv",
                ["drop-only-latent-only-128-epi"],
            )
            averaged = plot_common.average_object_metric_across_seeds(
                per_object_frame,
                labels=["drop-only-latent-only-128-epi"],
                value_column="success_lift_vs_dataset",
            )
            self.assertEqual(len(averaged), len(OBJECT_IDS))
            object_75 = averaged.loc[averaged["object_id"] == 75, "seed_avg_value"].iloc[0]
            self.assertAlmostEqual(object_75, 0.02)

    def test_load_per_object_table_with_baseline_auto_loads_no_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir), labels=("no-action", "full-latefus-128-epi"))
            frame = plot_common.load_per_object_table_with_baseline(root, ["full-latefus-128-epi"])
            self.assertEqual(set(frame["label"].unique().tolist()), {"no-action", "full-latefus-128-epi"})

    def test_adjusted_values_are_paired_by_seed_and_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir), labels=plot_config.ORDERED_LABELS)
            frame = plot_common.load_per_object_table_with_baseline(root, ["full-latefus-128-epi"])
            adjusted = plot_common.compute_adjusted_per_object_values(
                frame,
                ["full-latefus-128-epi"],
                metric_key="success_gain",
            )
            self.assertEqual(len(adjusted), len(TEST_SEEDS) * len(OBJECT_IDS))
            self.assertTrue((adjusted["adjusted_value"].round(10) == 0.12).all())

    def test_no_action_adjusted_values_are_zero_when_plotted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir), labels=("no-action", "full-latefus-128-epi"))
            frame = plot_common.load_per_object_table_with_baseline(root, ["no-action", "full-latefus-128-epi"])
            adjusted = plot_common.compute_adjusted_per_object_values(frame, ["no-action"], metric_key="success_gain")
            self.assertTrue((adjusted["adjusted_value"].abs() < 1e-12).all())

    def test_missing_no_action_baseline_errors_for_adjusted_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir), labels=("full-latefus-128-epi",))
            with self.assertRaisesRegex(ValueError, "baseline"):
                plot_common.load_per_object_table_with_baseline(root, ["full-latefus-128-epi"])

    def test_adjusted_bootstrap_ci_does_not_reuse_summary_ci(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir), labels=plot_config.ORDERED_LABELS)
            labels = ["full-latefus-128-epi"]
            per_object_frame = plot_common.load_per_object_table_with_baseline(root, labels)
            plot_frame = fig01.prepare_data(per_object_frame, labels)
            self.assertAlmostEqual(float(plot_frame["adjusted_mean"].iloc[0]), 0.12)
            self.assertAlmostEqual(float(plot_frame["adjusted_ci95_low"].iloc[0]), 0.12)
            self.assertAlmostEqual(float(plot_frame["adjusted_ci95_high"].iloc[0]), 0.12)
            self.assertEqual(int(plot_frame["num_objects"].iloc[0]), len(OBJECT_IDS))


class TestPlotPreparation(unittest.TestCase):
    def test_adjusted_prepare_functions_map_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir))
            labels = list(plot_config.ORDERED_LABELS)
            per_object_frame = plot_common.load_per_object_table_with_baseline(root, labels)

            fig01_data = fig01.prepare_data(per_object_frame, labels)
            self.assertIn("adjusted_mean", fig01_data.columns)
            self.assertEqual(set(fig01_data["label"].tolist()), set(labels))

            fig02_data = fig02.prepare_data(per_object_frame, labels)
            self.assertIn("degradation_mean", fig02_data.columns)
            self.assertIn("recovery_mean", fig02_data.columns)
            self.assertNotIn("no-action", fig02.plot_labels_without_baseline(["no-action", "full-latefus-128-epi"]))
            self.assertEqual(fig02.plot_labels_without_baseline(["no-action", "full-latefus-128-epi"]), ["full-latefus-128-epi"])

            fig03_data = fig03.prepare_data(per_object_frame, labels)
            self.assertIn("degradation_mean", fig03_data.columns)
            self.assertIn("recovery_mean", fig03_data.columns)

            fig04_data = fig04.prepare_data(per_object_frame, labels)
            self.assertIn("excess_probability_delta", fig04_data)
            self.assertIn("adjusted_mean", fig04_data["excess_probability_delta"].columns)

            fig06_data = fig06.prepare_data(per_object_frame, labels)
            self.assertIn("seed_avg_value", fig06_data.columns)

            fig07_data = fig07.prepare_data(per_object_frame, labels, "iqr")
            self.assertIn("adjusted_stability", fig07_data.columns)

            fig08_data = fig08.prepare_data(per_object_frame, labels)
            self.assertIn("rank", fig08_data.columns)

    def test_per_run_overlay_prepare_data_uses_adjusted_object_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir))
            labels = list(plot_config.ORDERED_LABELS)
            per_object_frame = plot_common.load_per_object_table_with_baseline(root, labels)
            prepared_summary, prepared_runs = fig09.prepare_data(per_object_frame, labels)
            self.assertEqual(len(prepared_summary), len(plot_config.ORDERED_LABELS))
            self.assertEqual(len(prepared_runs), len(plot_config.ORDERED_LABELS) * len(TEST_SEEDS))
            self.assertEqual(sorted(prepared_runs["test_seed"].unique().tolist()), list(TEST_SEEDS))
            no_action_runs = prepared_runs.loc[prepared_runs["label"] == "no-action", "adjusted_run_mean"]
            self.assertTrue((no_action_runs.abs() < 1e-12).all())


class TestTrainingCurvePreparation(unittest.TestCase):
    def test_metrics_jsonl_loader_extracts_train_and_validation_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_metrics_fixture(Path(tmpdir) / "seed7", steps=range(4), offset=0.0)
            frame = fig10.read_metrics_jsonl(
                run_dir,
                seed_label="seed7",
                metrics=fig10.DEFAULT_METRICS,
            )
            self.assertEqual(set(frame["metric"].unique().tolist()), set(fig10.DEFAULT_METRICS))
            train_rows = frame.loc[frame["metric"] == "outcome/success_lift_vs_dataset"]
            validation_rows = frame.loc[frame["metric"] == "validation/outcome/success_lift_vs_dataset"]
            self.assertEqual(train_rows["step"].tolist(), [0, 1, 2, 3])
            self.assertEqual(validation_rows["step"].tolist(), [0, 2])

    def test_missing_metric_warns_without_crashing_when_other_metrics_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = write_metrics_fixture(
                Path(tmpdir) / "seed7",
                steps=range(3),
                offset=0.0,
                include_validation=False,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                frame = fig10.read_metrics_jsonl(
                    run_dir,
                    seed_label="seed7",
                    metrics=fig10.DEFAULT_METRICS,
                )
            self.assertTrue(caught)
            self.assertEqual(set(frame["metric"].unique().tolist()), {"outcome/success_lift_vs_dataset"})

    def test_common_alignment_keeps_only_steps_shared_by_all_seeds_per_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dirs = [
                write_metrics_fixture(Path(tmpdir) / "seed7", steps=range(5), offset=0.0),
                write_metrics_fixture(Path(tmpdir) / "seed8", steps=range(4), offset=1.0),
                write_metrics_fixture(Path(tmpdir) / "seed9", steps=range(3), offset=2.0),
            ]
            prepared = fig10.prepare_data(
                run_dirs,
                ["seed7", "seed8", "seed9"],
                ["outcome/success_lift_vs_dataset"],
                align="common",
                smooth_window=1,
            )
            self.assertEqual(prepared["step"].tolist(), [0, 1, 2])
            self.assertTrue((prepared["num_seeds"] == 3).all())

    def test_available_alignment_keeps_partial_steps_and_counts_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dirs = [
                write_metrics_fixture(Path(tmpdir) / "seed7", steps=range(5), offset=0.0),
                write_metrics_fixture(Path(tmpdir) / "seed8", steps=range(4), offset=1.0),
                write_metrics_fixture(Path(tmpdir) / "seed9", steps=range(3), offset=2.0),
            ]
            prepared = fig10.prepare_data(
                run_dirs,
                ["seed7", "seed8", "seed9"],
                ["outcome/success_lift_vs_dataset"],
                align="available",
                smooth_window=1,
            )
            self.assertEqual(prepared["step"].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(prepared["num_seeds"].tolist(), [3, 3, 3, 2, 1])

    def test_smooth_window_one_keeps_raw_seed_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dirs = [
                write_metrics_fixture(Path(tmpdir) / "seed7", steps=range(3), offset=0.0),
                write_metrics_fixture(Path(tmpdir) / "seed8", steps=range(3), offset=0.0),
                write_metrics_fixture(Path(tmpdir) / "seed9", steps=range(3), offset=0.0),
            ]
            prepared = fig10.prepare_data(
                run_dirs,
                ["seed7", "seed8", "seed9"],
                ["outcome/success_lift_vs_dataset"],
                align="common",
                smooth_window=1,
            )
            self.assertEqual(prepared["raw_seed7"].tolist(), prepared["smooth_seed7"].tolist())

    def test_smoothing_is_applied_per_seed_before_mean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dirs = [
                write_metrics_fixture(Path(tmpdir) / "seed7", steps=range(3), offset=0.0),
                write_metrics_fixture(Path(tmpdir) / "seed8", steps=range(3), offset=1.0),
                write_metrics_fixture(Path(tmpdir) / "seed9", steps=range(3), offset=2.0),
            ]
            prepared = fig10.prepare_data(
                run_dirs,
                ["seed7", "seed8", "seed9"],
                ["outcome/success_lift_vs_dataset"],
                align="common",
                smooth_window=2,
            )
            step_1 = prepared.loc[prepared["step"] == 1].iloc[0]
            self.assertAlmostEqual(float(step_1["smooth_seed7"]), 0.05)
            self.assertAlmostEqual(float(step_1["smooth_seed8"]), 1.05)
            self.assertAlmostEqual(float(step_1["smooth_seed9"]), 2.05)
            self.assertAlmostEqual(float(step_1["mean"]), 1.05)


class TestPlotScriptSmoke(unittest.TestCase):
    def test_all_plot_scripts_run_with_temp_fixture(self):
        scripts = [
            "fig01_main_overall_performance.py",
            "fig02_main_risk_return.py",
            "fig03_risk_return_scatter.py",
            "fig04_mechanism_triplet.py",
            "fig06_object_stability_boxplot.py",
            "fig07_object_stability_bar.py",
            "fig08_per_object_rank_curve.py",
            "fig09_per_run_overlay.py",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir) / "formal")
            output_dir = Path(tmpdir) / "generated"
            for script_name in scripts:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(PLOT_SCRIPTS_DIR / script_name),
                        "--root",
                        str(root),
                        "--out-dir",
                        str(output_dir),
                        "--no-print-data",
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise AssertionError(
                        f"{script_name} failed.\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                    )
                expected_group = "main" if script_name in {
                    "fig01_main_overall_performance.py",
                    "fig02_main_risk_return.py",
                    "fig03_risk_return_scatter.py",
                    "fig09_per_run_overlay.py",
                } else "ablation"
                expected_path = output_dir / expected_group / f"{Path(script_name).stem}.png"
                self.assertTrue(expected_path.exists(), msg=f"Missing plot output: {expected_path}")
                self.assertFalse((output_dir / expected_group / f"{Path(script_name).stem}.pdf").exists())

    def test_default_cli_outputs_final_plot_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir) / "formal")
            output_dir = Path(tmpdir) / "generated"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPTS_DIR / "fig01_main_overall_performance.py"),
                    "--root",
                    str(root),
                    "--out-dir",
                    str(output_dir),
                    "--print-data-format",
                    "csv",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(f"fig01 print-data failed.\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}")
            self.assertIn("[plot-data]", completed.stdout)
            self.assertIn("adjusted_mean", completed.stdout)

    def test_no_print_data_cli_suppresses_final_plot_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = write_plot_fixture(Path(tmpdir) / "formal")
            output_dir = Path(tmpdir) / "generated"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPTS_DIR / "fig01_main_overall_performance.py"),
                    "--root",
                    str(root),
                    "--out-dir",
                    str(output_dir),
                    "--no-print-data",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(f"fig01 no-print-data failed.\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}")
            self.assertNotIn("[plot-data]", completed.stdout)

    def test_training_curve_script_runs_with_temp_metrics_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dirs = [
                write_metrics_fixture(Path(tmpdir) / "seed7", steps=range(5), offset=0.0),
                write_metrics_fixture(Path(tmpdir) / "seed8", steps=range(5), offset=1.0),
                write_metrics_fixture(Path(tmpdir) / "seed9", steps=range(3), offset=2.0),
            ]
            output_dir = Path(tmpdir) / "generated" / "training"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPTS_DIR / "fig10_full_training_curves.py"),
                    "--run-dirs",
                    *(str(path) for path in run_dirs),
                    "--labels",
                    "seed7",
                    "seed8",
                    "seed9",
                    "--out-dir",
                    str(output_dir),
                    "--smooth-window",
                    "1",
                    "--align",
                    "common",
                    "--no-print-data",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise AssertionError(
                    f"fig10 smoke failed.\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )
            self.assertTrue((output_dir / "fig10_full_training_curves.png").exists())
            self.assertTrue((output_dir / "fig10_full_training_curves_data.csv").exists())
            self.assertFalse((output_dir / "fig10_full_training_curves.pdf").exists())


if __name__ == "__main__":
    unittest.main()
