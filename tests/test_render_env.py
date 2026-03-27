from __future__ import annotations

import os
import unittest

from src.runtime.render_env import configure_render_environment


class TestRenderEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = os.environ.get("PYOPENGL_PLATFORM")

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        else:
            os.environ["PYOPENGL_PLATFORM"] = self._previous

    def test_sets_egl_for_headless_scene(self):
        os.environ.pop("PYOPENGL_PLATFORM", None)
        payload = configure_render_environment({"use_gui": False})
        self.assertEqual(os.environ.get("PYOPENGL_PLATFORM"), "egl")
        self.assertEqual(payload, {"PYOPENGL_PLATFORM": "egl"})

    def test_does_not_override_existing_backend(self):
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        payload = configure_render_environment({"use_gui": False})
        self.assertEqual(os.environ.get("PYOPENGL_PLATFORM"), "osmesa")
        self.assertEqual(payload, {"PYOPENGL_PLATFORM": "osmesa"})

    def test_does_not_set_backend_for_gui_scene(self):
        os.environ.pop("PYOPENGL_PLATFORM", None)
        payload = configure_render_environment({"use_gui": True})
        self.assertIsNone(os.environ.get("PYOPENGL_PLATFORM"))
        self.assertEqual(payload, {})


if __name__ == "__main__":
    unittest.main()
