from __future__ import annotations

import unittest
from pathlib import Path

from src.envs.asset_paths import resolve_object_urdf, resolve_scene_asset_paths


class TestEnvAssetPaths(unittest.TestCase):
    def test_scene_assets_are_repo_local(self):
        repo_root = Path(__file__).resolve().parents[1]
        asset_paths = resolve_scene_asset_paths()
        object_urdf = resolve_object_urdf(0)

        self.assertTrue(asset_paths.hand_python.exists())
        self.assertTrue(asset_paths.hand_urdf.exists())
        self.assertTrue(asset_paths.tacto_background.exists())
        self.assertTrue(asset_paths.tacto_config.exists())
        self.assertTrue(object_urdf.exists())

        self.assertTrue(str(asset_paths.hand_python).startswith(str(repo_root)))
        self.assertTrue(str(asset_paths.hand_urdf).startswith(str(repo_root)))
        self.assertTrue(str(asset_paths.tacto_background).startswith(str(repo_root)))
        self.assertTrue(str(asset_paths.tacto_config).startswith(str(repo_root)))
        self.assertTrue(str(object_urdf).startswith(str(repo_root)))

    def test_env_python_modules_do_not_reference_tac_sim_root(self):
        env_root = Path(__file__).resolve().parents[1] / "src" / "envs"
        for path in env_root.rglob("*.py"):
            if "object_model" in path.parts:
                continue
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("/tac-sim-sys", content, msg=f"Unexpected external dependency in {path}")


if __name__ == "__main__":
    unittest.main()
