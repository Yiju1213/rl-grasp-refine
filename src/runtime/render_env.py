from __future__ import annotations

import os


def configure_render_environment(scene_cfg: dict | None) -> dict[str, str]:
    scene_cfg = scene_cfg or {}
    if bool(scene_cfg.get("use_gui", False)):
        return {}
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return {
        "PYOPENGL_PLATFORM": os.environ["PYOPENGL_PLATFORM"],
    }
