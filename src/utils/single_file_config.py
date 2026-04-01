from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.runtime.experiment_config import apply_experiment_overrides
from src.utils.config import load_config


def _resolve_path(path_str: str | Path, *, base_dir: str | Path | None = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.expanduser().resolve()
    base = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
    return (base / path).resolve()


def _find_enclosing_configs_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if candidate.name == "configs":
            return candidate
    return None


def _resolve_config_reference(path_str: str | Path, *, experiment_path: Path, bundle_base: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.expanduser().resolve()
    if path.parts and path.parts[0] == "configs":
        return _resolve_path(path, base_dir=bundle_base)
    return _resolve_path(path, base_dir=experiment_path.parent)


def _looks_like_experiment_config(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
        return False
    try:
        payload = load_config(path)
    except Exception:
        return False
    configs = payload.get("configs")
    return isinstance(configs, dict) and bool(configs)


def _find_experiment_candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    candidates = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    return [path for path in candidates if _looks_like_experiment_config(path)]


def discover_experiment_config(input_path: str | Path) -> Path:
    resolved = _resolve_path(input_path)
    if resolved.is_file() and _looks_like_experiment_config(resolved):
        return resolved

    search_roots = [resolved] if resolved.is_dir() else [resolved.parent, *resolved.parents]
    for root in search_roots:
        if not root.exists():
            continue
        candidates = _find_experiment_candidates(root)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            candidate_list = ", ".join(str(path) for path in candidates[:5])
            raise ValueError(
                f"Found multiple experiment configs under {root}: {candidate_list}. "
                "Please pass the experiment yaml path explicitly."
            )

    raise FileNotFoundError(f"Could not locate an experiment config from input: {resolved}")


def load_experiment_bundle_from_input(input_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    experiment_path = discover_experiment_config(input_path)
    experiment_cfg = load_config(experiment_path)
    configs_root = _find_enclosing_configs_root(experiment_path)
    bundle_base = configs_root.parent if configs_root is not None else experiment_path.parent
    bundle = {
        key: load_config(
            _resolve_config_reference(relative_path, experiment_path=experiment_path, bundle_base=bundle_base)
        )
        for key, relative_path in experiment_cfg.get("configs", {}).items()
    }
    experiment_cfg_resolved, bundle_resolved = apply_experiment_overrides(experiment_cfg, bundle)
    return experiment_cfg_resolved, bundle_resolved, experiment_path


def build_single_file_config(input_path: str | Path) -> dict[str, Any]:
    experiment_cfg, bundle, _ = load_experiment_bundle_from_input(input_path)
    experiment_section = deepcopy(experiment_cfg)
    experiment_section.pop("configs", None)

    single_file_config: dict[str, Any] = {"experiment": experiment_section}
    for key, value in bundle.items():
        single_file_config[str(key)] = deepcopy(value)
    return single_file_config


def dump_single_file_config(input_path: str | Path, output_path: str | Path) -> Path:
    payload = build_single_file_config(input_path)
    output_path_resolved = _resolve_path(output_path)
    output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    with output_path_resolved.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return output_path_resolved
