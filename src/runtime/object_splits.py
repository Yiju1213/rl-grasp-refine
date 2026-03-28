from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any


@dataclass(frozen=True)
class ObjectSplit:
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    split_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "test_ids": list(self.test_ids),
            "split_seed": int(self.split_seed),
        }


def _resolve_inclusive_range(raw_value: Any, *, field_name: str) -> list[int]:
    if raw_value is None:
        return []
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
        raise ValueError(f"{field_name} must be a two-element inclusive range like [start, end].")
    start = int(raw_value[0])
    end = int(raw_value[1])
    if end < start:
        raise ValueError(f"{field_name} must satisfy end >= start. Got {raw_value!r}.")
    return list(range(start, end + 1))


def _resolve_val_count(*, holdout_count: int, raw_count: Any, raw_ratio: Any) -> int:
    if raw_count is not None:
        return max(min(int(raw_count), holdout_count), 0)
    if raw_ratio is None:
        return 0
    ratio = float(raw_ratio)
    if not math.isfinite(ratio):
        raise ValueError(f"val_object_ratio must be finite. Got {raw_ratio!r}.")
    ratio = min(max(ratio, 0.0), 1.0)
    if holdout_count == 0:
        return 0
    if ratio <= 0.0:
        return 0
    return min(max(int(round(float(holdout_count) * ratio)), 1), holdout_count)


def resolve_object_split(experiment_cfg: dict[str, Any]) -> ObjectSplit:
    train_ids = _resolve_inclusive_range(
        experiment_cfg.get("train_object_id_range"),
        field_name="train_object_id_range",
    )
    holdout_ids = _resolve_inclusive_range(
        experiment_cfg.get("holdout_object_id_range"),
        field_name="holdout_object_id_range",
    )
    if set(train_ids) & set(holdout_ids):
        raise ValueError("train_object_id_range and holdout_object_id_range must be disjoint.")

    split_seed = int(experiment_cfg.get("split_seed", experiment_cfg.get("seed", 0)))
    shuffled_holdout = list(holdout_ids)
    random.Random(split_seed).shuffle(shuffled_holdout)
    val_count = _resolve_val_count(
        holdout_count=len(shuffled_holdout),
        raw_count=experiment_cfg.get("val_object_count"),
        raw_ratio=experiment_cfg.get("val_object_ratio"),
    )
    val_ids = sorted(shuffled_holdout[:val_count])
    test_ids = sorted(shuffled_holdout[val_count:])
    return ObjectSplit(
        train_ids=sorted(train_ids),
        val_ids=val_ids,
        test_ids=test_ids,
        split_seed=split_seed,
    )
