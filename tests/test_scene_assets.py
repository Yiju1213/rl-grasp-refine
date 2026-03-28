from __future__ import annotations

import unittest
from unittest.mock import patch

from src.envs.scene_assets import destroy_tacto_sensor, remove_object_from_tacto_sensor


class _FakeScene:
    def __init__(self):
        self.removed_nodes = []

    def remove_node(self, node):
        self.removed_nodes.append(node)


class _FakeOffscreenRenderer:
    def __init__(self):
        self.delete_calls = 0

    def delete(self):
        self.delete_calls += 1


class _FakeRenderer:
    def __init__(self):
        self.scene = _FakeScene()
        self.current_object_nodes = {}
        self.object_nodes = {}
        self.render_calls = []
        self.r = _FakeOffscreenRenderer()

    def render(self, object_poses=None, normal_forces=None, noise=True, calibration=True):
        self.render_calls.append(
            {
                "object_poses": object_poses,
                "normal_forces": normal_forces,
                "noise": noise,
                "calibration": calibration,
            }
        )
        return [], []


class _FakeSensor:
    def __init__(self):
        self.renderer = _FakeRenderer()
        self.objects = {}
        self.object_poses = {}
        self.normal_forces = {}
        self._static = ("cached", "cached")


class TestSceneAssets(unittest.TestCase):
    def test_remove_object_from_tacto_sensor_clears_refs_and_flushes_renderer(self):
        sensor = _FakeSensor()
        current_node = object()
        object_node = object()
        other_node = object()
        sensor.objects = {
            "42_-1": object(),
            "42_0": object(),
            "7_-1": object(),
        }
        sensor.object_poses = {
            "42_-1": (None, None),
            "42_0": (None, None),
            "7_-1": (None, None),
        }
        sensor.renderer.current_object_nodes = {
            "42_-1": current_node,
            "42_0": current_node,
            "7_-1": other_node,
        }
        sensor.renderer.object_nodes = {
            "42_-1": object_node,
            "42_0": object_node,
            "7_-1": other_node,
        }
        sensor.normal_forces = {
            "cam0": {"42_-1": 1.0, "7_-1": 2.0},
            "cam1": {"42_0": 3.0},
        }

        with patch("src.envs.scene_assets._best_effort_trim_process_heap") as trim_mock:
            remove_object_from_tacto_sensor(sensor, 42)

        self.assertEqual(set(sensor.objects.keys()), {"7_-1"})
        self.assertEqual(set(sensor.object_poses.keys()), {"7_-1"})
        self.assertEqual(set(sensor.renderer.current_object_nodes.keys()), {"7_-1"})
        self.assertEqual(set(sensor.renderer.object_nodes.keys()), {"7_-1"})
        self.assertEqual(sensor.normal_forces["cam0"], {"7_-1": 2.0})
        self.assertEqual(sensor.normal_forces["cam1"], {})
        self.assertIsNone(sensor._static)
        self.assertEqual(sensor.renderer.scene.removed_nodes, [current_node, object_node])
        self.assertEqual(
            sensor.renderer.render_calls,
            [
                {
                    "object_poses": None,
                    "normal_forces": None,
                    "noise": False,
                    "calibration": False,
                }
            ],
        )
        trim_mock.assert_called_once_with()

    def test_destroy_tacto_sensor_deletes_renderer_and_is_idempotent(self):
        sensor = _FakeSensor()
        sensor.objects = {"42_-1": object()}
        sensor.object_poses = {"42_-1": (None, None)}
        sensor.normal_forces = {"cam0": {"42_-1": 1.0}}
        offscreen_renderer = sensor.renderer.r

        with patch("src.envs.scene_assets._best_effort_trim_process_heap") as trim_mock:
            destroy_tacto_sensor(sensor)
            destroy_tacto_sensor(sensor)

        self.assertEqual(offscreen_renderer.delete_calls, 1)
        self.assertEqual(sensor.renderer.r, None)
        self.assertEqual(sensor.objects, {})
        self.assertEqual(sensor.object_poses, {})
        self.assertEqual(sensor.normal_forces["cam0"], {})
        self.assertEqual(trim_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
