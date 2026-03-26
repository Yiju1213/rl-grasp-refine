from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ENVS_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = ENVS_ROOT / "assets"
OBJECT_MODEL_ROOT = ENVS_ROOT / "object_model"

#TODO 整个文件可以移到scene_assets.py里，因为其实都是它在调用，且该文件功能较单一

@dataclass(frozen=True)
class SceneAssetPaths:
    hand_python: Path
    hand_package_root: Path
    hand_urdf: Path
    tacto_background: Path
    tacto_config: Path
    object_model_root: Path


def resolve_scene_asset_paths() -> SceneAssetPaths:
    return SceneAssetPaths(
        hand_python=ENVS_ROOT / "hand" / "gsmini_panda_hand.py",
        hand_package_root=ASSETS_ROOT / "gsmini_panda_hand",
        hand_urdf=ASSETS_ROOT / "gsmini_panda_hand" / "urdf" / "hand_gsmini.urdf",
        tacto_background=ASSETS_ROOT / "tacto" / "bg_gsmini_240_320.jpeg",
        tacto_config=ASSETS_ROOT / "tacto" / "config_gsmini.yml",
        object_model_root=OBJECT_MODEL_ROOT,
    )


def resolve_object_urdf(object_id: int) -> Path:
    return OBJECT_MODEL_ROOT / "model" / f"{int(object_id):03d}" / "object.urdf"
