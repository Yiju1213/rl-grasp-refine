from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.calibration.online_logit_calibrator import OnlineLogitCalibrator
from src.evaluation.best_checkpoint_pipeline import (
    ObjectEvaluationBudget,
    aggregate_object_episode_records,
    aggregate_run_object_rows,
    load_evaluation_manifest,
    run_best_checkpoint_evaluation,
)
from src.runtime.builders import build_actor_critic
from src.utils.checkpoint import save_checkpoint
from tests.fakes import (
    build_evaluation_async_env_for_worker,
    make_actor_critic_cfg,
    make_calibration_cfg,
    make_env_cfg,
    make_perception_cfg,
    make_rl_cfg,
)

_PER_OBJECT_HEADER = [
    "experiment_name",
    "test_seed",
    "object_id",
    "success_lift_vs_dataset",
    "positive_drop_rate",
    "negative_hold_rate",
    "t_cover_delta_mean",
    "t_edge_delta_mean",
    "prob_delta_mean",
    "num_episodes",
    "positive_count",
    "negative_count",
    "positive_drop_count",
    "negative_hold_count",
]

_PER_RUN_HEADER = [
    "experiment_name",
    "test_seed",
    "macro_success_lift",
    "pos_drop_rate",
    "neg_hold_rate",
    "across_object_lift_std",
    "across_object_lift_iqr",
    "t_cover_delta_mean",
    "t_edge_delta_mean",
    "prob_delta_mean",
    "num_objects",
    "total_episodes",
]

_SUMMARY_HEADER = [
    "experiment_name",
    "evaluation_wall_minutes",
    "macro_success_lift_mean",
    "macro_success_lift_std",
    "macro_success_lift_ci95_low",
    "macro_success_lift_ci95_high",
    "pos_drop_mean",
    "pos_drop_std",
    "pos_drop_ci95_low",
    "pos_drop_ci95_high",
    "neg_hold_mean",
    "neg_hold_std",
    "neg_hold_ci95_low",
    "neg_hold_ci95_high",
    "across_object_lift_std_mean",
    "across_object_lift_std_std",
    "across_object_lift_iqr_mean",
    "across_object_lift_iqr_std",
    "t_cover_delta_mean",
    "t_cover_delta_std",
    "t_cover_delta_ci95_low",
    "t_cover_delta_ci95_high",
    "t_edge_delta_mean",
    "t_edge_delta_std",
    "t_edge_delta_ci95_low",
    "t_edge_delta_ci95_high",
    "prob_delta_mean_mean",
    "prob_delta_mean_std",
    "prob_delta_mean_ci95_low",
    "prob_delta_mean_ci95_high",
]


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cell_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def _stub_object_budget_resolver(
    *,
    dataset_cfg: dict,
    object_id: int,
    test_seed: int,
    requested_num_workers: int,
) -> ObjectEvaluationBudget:
    del dataset_cfg, object_id, test_seed
    effective_num_workers = max(1, min(int(requested_num_workers), 1))
    return ObjectEvaluationBudget(
        available_samples=3,
        effective_num_workers=effective_num_workers,
        per_worker_dispatch_limits={worker_id: 3 for worker_id in range(effective_num_workers)},
        total_block_count=3,
    )


def _make_experiment_dir(root: Path, label: str) -> Path:
    experiment_dir = root / label
    configs_dir = experiment_dir / "configs"
    checkpoints_dir = experiment_dir / "checkpoints"

    env_cfg = make_env_cfg(seed=11)
    env_cfg["dataset"] = {"enabled": True}
    perception_cfg = make_perception_cfg()
    calibration_cfg = make_calibration_cfg()
    actor_critic_cfg = make_actor_critic_cfg()
    actor_critic_cfg["policy_observation"] = {"preset": "paper"}
    rl_cfg = make_rl_cfg()
    rl_cfg["batch_episodes"] = 2
    rl_cfg["num_envs"] = 1

    _write_yaml(
        configs_dir / "experiment.yaml",
        {
            "name": label,
            "seed": 11,
            "logging": {"experiment_name": label},
            "configs": {
                "env": "env.yaml",
                "perception": "perception.yaml",
                "calibration": "calibration.yaml",
                "rl": "rl.yaml",
                "actor_critic": "actor_critic.yaml",
            },
        },
    )
    _write_yaml(configs_dir / "env.yaml", env_cfg)
    _write_yaml(configs_dir / "perception.yaml", perception_cfg)
    _write_yaml(configs_dir / "calibration.yaml", calibration_cfg)
    _write_yaml(configs_dir / "rl.yaml", rl_cfg)
    _write_yaml(configs_dir / "actor_critic.yaml", actor_critic_cfg)

    actor_critic = build_actor_critic(perception_cfg, actor_critic_cfg)
    calibrator = OnlineLogitCalibrator(calibration_cfg)
    calibrator.load_state(
        {
            "a": 1.35,
            "b": -0.15,
            "posterior_cov": np.asarray([[0.7, 0.0], [0.0, 0.9]], dtype=np.float64),
        }
    )
    save_checkpoint(
        checkpoints_dir / "best.pt",
        {
            "actor_critic": actor_critic.state_dict(),
            "calibrator": calibrator.get_state(),
            "experiment_cfg": {"name": label, "seed": 11},
            "best_metric_name": "validation/outcome/success_lift_vs_dataset",
        },
    )
    return experiment_dir


class TestBestCheckpointPipeline(unittest.TestCase):
    def test_object_and_run_aggregators_use_macro_semantics_and_preserve_na(self):
        positive_only_rows = aggregate_object_episode_records(
            experiment_name="demo",
            test_seed=101,
            object_id=3,
            episode_records=[
                {
                    "experiment_name": "demo",
                    "test_seed": 101,
                    "object_id": 3,
                    "drop_success": 1,
                    "legacy_drop_success_before": 1.0,
                    "t_cover_delta": 0.3,
                    "t_edge_delta": 0.2,
                    "prob_delta": 0.05,
                },
                {
                    "experiment_name": "demo",
                    "test_seed": 101,
                    "object_id": 3,
                    "drop_success": 0,
                    "legacy_drop_success_before": 1.0,
                    "t_cover_delta": 0.5,
                    "t_edge_delta": 0.4,
                    "prob_delta": 0.15,
                },
            ],
        )
        negative_only_rows = aggregate_object_episode_records(
            experiment_name="demo",
            test_seed=101,
            object_id=4,
            episode_records=[
                {
                    "experiment_name": "demo",
                    "test_seed": 101,
                    "object_id": 4,
                    "drop_success": 0,
                    "legacy_drop_success_before": 0.0,
                    "t_cover_delta": 0.4,
                    "t_edge_delta": 0.3,
                    "prob_delta": 0.25,
                },
                {
                    "experiment_name": "demo",
                    "test_seed": 101,
                    "object_id": 4,
                    "drop_success": 1,
                    "legacy_drop_success_before": 0.0,
                    "t_cover_delta": 0.6,
                    "t_edge_delta": 0.5,
                    "prob_delta": 0.35,
                },
            ],
        )

        self.assertEqual(positive_only_rows["negative_hold_rate"], None)
        self.assertEqual(negative_only_rows["positive_drop_rate"], None)

        run_row = aggregate_run_object_rows(
            experiment_name="demo",
            test_seed=101,
            object_rows=[positive_only_rows, negative_only_rows],
        )

        self.assertAlmostEqual(run_row["macro_success_lift"], 0.0, places=7)
        self.assertAlmostEqual(run_row["pos_drop_rate"], 0.5, places=7)
        self.assertAlmostEqual(run_row["neg_hold_rate"], 0.5, places=7)
        self.assertAlmostEqual(run_row["t_cover_delta_mean"], 0.45, places=7)
        self.assertAlmostEqual(run_row["t_edge_delta_mean"], 0.35, places=7)
        self.assertAlmostEqual(run_row["prob_delta_mean"], 0.20, places=7)

    def test_load_evaluation_manifest_validates_duplicate_labels_and_seed_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.yaml"
            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "dup", "experiment_dir": "exp_a"},
                        {"label": "dup", "experiment_dir": "exp_b"},
                    ],
                    "protocol": {
                        "test_object_ids": [3, 4],
                        "test_seeds": [1, 2, 3],
                        "episodes_per_object": 2,
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "Duplicate experiment label"):
                load_evaluation_manifest(manifest_path)

            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "exp_a", "experiment_dir": "exp_a"},
                    ],
                    "protocol": {
                        "test_object_ids": [3, 4],
                        "test_seeds": [1, 2],
                        "episodes_per_object": 2,
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "exactly 3 seeds"):
                load_evaluation_manifest(manifest_path)

    def test_load_evaluation_manifest_parses_policy_modes_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.yaml"
            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "legacy", "experiment_dir": "exp_a"},
                        {"label": "no-action", "experiment_dir": "exp_b", "policy_mode": "zero_action"},
                        {
                            "label": "rand-action",
                            "experiment_dir": "exp_c",
                            "policy_mode": "random_uniform",
                            "action_seed": 17,
                        },
                    ],
                    "protocol": {
                        "test_object_ids": [3, 4],
                        "test_seeds": [1, 2, 3],
                        "episodes_per_object": 2,
                    },
                },
            )

            manifest = load_evaluation_manifest(manifest_path)

            self.assertEqual(manifest.experiments[0].policy_mode, "learned_best")
            self.assertEqual(manifest.experiments[0].action_seed, 0)
            self.assertEqual(manifest.experiments[1].policy_mode, "zero_action")
            self.assertEqual(manifest.experiments[1].action_seed, 0)
            self.assertEqual(manifest.experiments[2].policy_mode, "random_uniform")
            self.assertEqual(manifest.experiments[2].action_seed, 17)

            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "bad", "experiment_dir": "exp_a", "policy_mode": "not_a_policy"},
                    ],
                    "protocol": {
                        "test_object_ids": [3, 4],
                        "test_seeds": [1, 2, 3],
                        "episodes_per_object": 2,
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "Unsupported rollout policy_mode"):
                load_evaluation_manifest(manifest_path)

    def test_run_best_checkpoint_evaluation_raises_when_all_experiments_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing_dir = root / "missing_exp"
            (missing_dir / "configs").mkdir(parents=True, exist_ok=True)
            manifest_path = root / "manifest.yaml"
            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "missing", "experiment_dir": str(missing_dir)},
                    ],
                    "protocol": {
                        "test_object_ids": [3],
                        "test_seeds": [11, 12, 13],
                        "episodes_per_object": 2,
                    },
                },
            )

            with self.assertRaisesRegex(RuntimeError, "All experiments failed"):
                run_best_checkpoint_evaluation(
                    manifest_path,
                    env_factory=build_evaluation_async_env_for_worker,
                    object_budget_resolver=_stub_object_budget_resolver,
                )

    def test_run_best_checkpoint_evaluation_continues_after_one_experiment_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiment_ok = _make_experiment_dir(root, "baseline")
            missing_dir = root / "missing_exp"
            (missing_dir / "configs").mkdir(parents=True, exist_ok=True)
            manifest_path = root / "manifest.yaml"
            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "baseline", "experiment_dir": str(experiment_ok)},
                        {"label": "missing", "experiment_dir": str(missing_dir)},
                    ],
                    "protocol": {
                        "test_object_ids": [3, 4],
                        "test_seeds": [101, 102, 103],
                        "episodes_per_object": 3,
                        "bootstrap_iterations": 50,
                        "confidence_level": 0.95,
                    },
                },
            )

            output_paths = run_best_checkpoint_evaluation(
                manifest_path,
                env_factory=build_evaluation_async_env_for_worker,
                object_budget_resolver=_stub_object_budget_resolver,
            )

            self.assertEqual(set(output_paths["experiments"].keys()), {"baseline"})
            self.assertEqual(set(output_paths["failed_experiments"].keys()), {"missing"})
            self.assertIn("Missing best checkpoint", output_paths["failed_experiments"]["missing"]["error"])
            self.assertTrue((output_paths["output_dir"] / "baseline" / "summary.csv").exists())
            self.assertFalse((output_paths["output_dir"] / "missing").exists())

    def test_run_best_checkpoint_evaluation_writes_expected_outputs_and_macro_rollups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiment_a = _make_experiment_dir(root, "baseline")
            experiment_b = _make_experiment_dir(root, "wo-stb-rwd")
            manifest_path = root / "manifest.yaml"
            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "baseline", "experiment_dir": str(experiment_a)},
                        {"label": "wo-stb-rwd", "experiment_dir": str(experiment_b)},
                    ],
                    "protocol": {
                        "test_object_ids": [3, 4],
                        "test_seeds": [101, 102, 103],
                        "episodes_per_object": 3,
                        "bootstrap_iterations": 200,
                        "confidence_level": 0.95,
                    },
                    "collector": {
                        "batch_episodes": 2,
                    },
                },
            )

            output_paths = run_best_checkpoint_evaluation(
                manifest_path,
                env_factory=build_evaluation_async_env_for_worker,
                object_budget_resolver=_stub_object_budget_resolver,
            )

            output_dir = output_paths["output_dir"]
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"baseline", "wo-stb-rwd"},
            )
            self.assertEqual(set(output_paths["experiments"].keys()), {"baseline", "wo-stb-rwd"})
            self.assertEqual(output_paths["failed_experiments"], {})

            for experiment_name, experiment_output_paths in output_paths["experiments"].items():
                experiment_output_dir = experiment_output_paths["directory"]
                self.assertEqual(
                    {path.name for path in experiment_output_dir.iterdir()},
                    {"metadata.json", "per_object_summary.csv", "per_run_summary.csv", "summary.csv"},
                )
                self.assertEqual(
                    (experiment_output_dir / "per_object_summary.csv").read_text(encoding="utf-8").splitlines()[0].split(","),
                    _PER_OBJECT_HEADER,
                )
                self.assertEqual(
                    (experiment_output_dir / "per_run_summary.csv").read_text(encoding="utf-8").splitlines()[0].split(","),
                    _PER_RUN_HEADER,
                )
                self.assertEqual(
                    (experiment_output_dir / "summary.csv").read_text(encoding="utf-8").splitlines()[0].split(","),
                    _SUMMARY_HEADER,
                )

                per_object_rows = _read_csv_rows(experiment_output_paths["per_object_summary"])
                per_run_rows = _read_csv_rows(experiment_output_paths["per_run_summary"])
                summary_rows = _read_csv_rows(experiment_output_paths["summary"])
                metadata = json.loads(experiment_output_paths["metadata"].read_text(encoding="utf-8"))

                self.assertEqual(len(per_object_rows), 6)
                self.assertEqual(len(per_run_rows), 3)
                self.assertEqual(len(summary_rows), 1)
                self.assertEqual(metadata["policy_mode"], "learned_best")
                self.assertEqual(metadata["action_mode"], "deterministic_mean")
                self.assertEqual(metadata["action_seed"], 0)
                self.assertEqual(metadata["protocol_notes"]["policy_mode"], "learned_best")
                self.assertEqual(metadata["protocol_notes"]["action_mode"], "deterministic_mean")
                self.assertFalse(metadata["protocol_notes"]["episode_records_persisted"])
                self.assertEqual(int(metadata["collector"]["num_workers"]), 1)
                self.assertEqual(metadata["experiment_name"], experiment_name)
                self.assertGreaterEqual(float(metadata["evaluation_wall_minutes"]), 0.0)

                grouped_object_rows: dict[int, list[dict[str, str]]] = {}
                for row in per_object_rows:
                    self.assertEqual(row["experiment_name"], experiment_name)
                    grouped_object_rows.setdefault(int(row["test_seed"]), []).append(row)

                for run_row in per_run_rows:
                    self.assertEqual(run_row["experiment_name"], experiment_name)
                    object_rows = grouped_object_rows[int(run_row["test_seed"])]
                    macro_success_lift = float(
                        np.mean([float(item["success_lift_vs_dataset"]) for item in object_rows])
                    )
                    pos_drop_rate = float(
                        np.mean([float(item["positive_drop_rate"]) for item in object_rows if item["positive_drop_rate"] != ""])
                    )
                    neg_hold_rate = float(
                        np.mean([float(item["negative_hold_rate"]) for item in object_rows if item["negative_hold_rate"] != ""])
                    )
                    t_cover_delta_mean = float(np.mean([float(item["t_cover_delta_mean"]) for item in object_rows]))
                    t_edge_delta_mean = float(np.mean([float(item["t_edge_delta_mean"]) for item in object_rows]))
                    prob_delta_mean = float(np.mean([float(item["prob_delta_mean"]) for item in object_rows]))

                    self.assertAlmostEqual(float(run_row["macro_success_lift"]), macro_success_lift, places=7)
                    self.assertAlmostEqual(float(run_row["pos_drop_rate"]), pos_drop_rate, places=7)
                    self.assertAlmostEqual(float(run_row["neg_hold_rate"]), neg_hold_rate, places=7)
                    self.assertAlmostEqual(float(run_row["t_cover_delta_mean"]), t_cover_delta_mean, places=7)
                    self.assertAlmostEqual(float(run_row["t_edge_delta_mean"]), t_edge_delta_mean, places=7)
                    self.assertAlmostEqual(float(run_row["prob_delta_mean"]), prob_delta_mean, places=7)

                summary_row = summary_rows[0]
                self.assertEqual(summary_row["experiment_name"], experiment_name)
                self.assertGreaterEqual(float(summary_row["evaluation_wall_minutes"]), 0.0)
                macro_mean = float(np.mean([float(item["macro_success_lift"]) for item in per_run_rows]))
                pos_mean = float(np.mean([float(item["pos_drop_rate"]) for item in per_run_rows]))
                neg_mean = float(np.mean([float(item["neg_hold_rate"]) for item in per_run_rows]))
                self.assertAlmostEqual(float(summary_row["macro_success_lift_mean"]), macro_mean, places=7)
                self.assertAlmostEqual(float(summary_row["pos_drop_mean"]), pos_mean, places=7)
                self.assertAlmostEqual(float(summary_row["neg_hold_mean"]), neg_mean, places=7)
                for prefix in (
                    "macro_success_lift",
                    "pos_drop",
                    "neg_hold",
                    "t_cover_delta",
                    "t_edge_delta",
                    "prob_delta_mean",
                ):
                    mean = _cell_float(summary_row, f"{prefix}_mean")
                    ci_low = _cell_float(summary_row, f"{prefix}_ci95_low")
                    ci_high = _cell_float(summary_row, f"{prefix}_ci95_high")
                    self.assertIsNotNone(mean)
                    self.assertIsNotNone(ci_low)
                    self.assertIsNotNone(ci_high)
                    self.assertLessEqual(float(ci_low), float(mean))
                    self.assertLessEqual(float(mean), float(ci_high))

    def test_run_best_checkpoint_evaluation_writes_policy_mode_metadata_for_fixed_action_baselines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            experiment_dir = _make_experiment_dir(root, "full")
            manifest_path = root / "manifest.yaml"
            _write_yaml(
                manifest_path,
                {
                    "output_dir": "outputs",
                    "experiments": [
                        {"label": "learned", "experiment_dir": str(experiment_dir)},
                        {"label": "no-action", "experiment_dir": str(experiment_dir), "policy_mode": "zero_action"},
                        {
                            "label": "rand-action",
                            "experiment_dir": str(experiment_dir),
                            "policy_mode": "random_uniform",
                            "action_seed": 23,
                        },
                    ],
                    "protocol": {
                        "test_object_ids": [3],
                        "test_seeds": [101, 102, 103],
                        "episodes_per_object": 2,
                        "bootstrap_iterations": 20,
                        "confidence_level": 0.95,
                    },
                    "collector": {
                        "batch_episodes": 1,
                    },
                },
            )

            output_paths = run_best_checkpoint_evaluation(
                manifest_path,
                env_factory=build_evaluation_async_env_for_worker,
                object_budget_resolver=_stub_object_budget_resolver,
            )

            self.assertEqual(set(output_paths["experiments"].keys()), {"learned", "no-action", "rand-action"})
            expected = {
                "learned": ("learned_best", "deterministic_mean", 0),
                "no-action": ("zero_action", "zero_action", 0),
                "rand-action": ("random_uniform", "random_uniform", 23),
            }
            for label, (policy_mode, action_mode, action_seed) in expected.items():
                paths = output_paths["experiments"][label]
                metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
                self.assertEqual(metadata["policy_mode"], policy_mode)
                self.assertEqual(metadata["action_mode"], action_mode)
                self.assertEqual(metadata["action_seed"], action_seed)
                self.assertEqual(metadata["protocol_notes"]["policy_mode"], policy_mode)
                self.assertEqual(metadata["protocol_notes"]["action_mode"], action_mode)
                self.assertEqual(metadata["protocol_notes"]["action_seed"], action_seed)
                self.assertEqual(
                    paths["summary"].read_text(encoding="utf-8").splitlines()[0].split(","),
                    _SUMMARY_HEADER,
                )
                self.assertEqual(
                    paths["per_run_summary"].read_text(encoding="utf-8").splitlines()[0].split(","),
                    _PER_RUN_HEADER,
                )
                self.assertEqual(
                    paths["per_object_summary"].read_text(encoding="utf-8").splitlines()[0].split(","),
                    _PER_OBJECT_HEADER,
                )


if __name__ == "__main__":
    unittest.main()
