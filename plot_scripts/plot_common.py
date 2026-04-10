from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import (
    COLORS,
    DEFAULT_DPI,
    DEFAULT_FORMATS,
    DEFAULT_OUT_DIR,
    DISPLAY_NAMES,
    GROUPS,
    MARKERS,
    ORDERED_LABELS,
    ROOT_DIR,
)

SUMMARY_METRIC_SPECS = {
    "macro_success_lift": {
        "mean": "macro_success_lift_mean",
        "ci_low": "macro_success_lift_ci95_low",
        "ci_high": "macro_success_lift_ci95_high",
        "ylabel": "Macro Success Lift",
    },
    "pos_drop": {
        "mean": "pos_drop_mean",
        "ci_low": "pos_drop_ci95_low",
        "ci_high": "pos_drop_ci95_high",
        "ylabel": "Positive-Sample Drop Rate",
    },
    "neg_hold": {
        "mean": "neg_hold_mean",
        "ci_low": "neg_hold_ci95_low",
        "ci_high": "neg_hold_ci95_high",
        "ylabel": "Negative-Sample Hold Rate",
    },
    "t_cover_delta": {
        "mean": "t_cover_delta_mean",
        "ci_low": "t_cover_delta_ci95_low",
        "ci_high": "t_cover_delta_ci95_high",
        "ylabel": "T-Cover Delta",
    },
    "t_edge_delta": {
        "mean": "t_edge_delta_mean",
        "ci_low": "t_edge_delta_ci95_low",
        "ci_high": "t_edge_delta_ci95_high",
        "ylabel": "T-Edge Delta",
    },
    "prob_delta_mean": {
        "mean": "prob_delta_mean_mean",
        "ci_low": "prob_delta_mean_ci95_low",
        "ci_high": "prob_delta_mean_ci95_high",
        "ylabel": "Probability Delta Mean",
    },
}

STABILITY_METRIC_SPECS = {
    "iqr": {
        "column": "across_object_lift_iqr_mean",
        "ylabel": "Across-Object Lift IQR",
    },
    "std": {
        "column": "across_object_lift_std_mean",
        "ylabel": "Across-Object Lift Std",
    },
}


def build_base_parser(
    description: str,
    *,
    default_group: str,
    style_choices: Sequence[str] | None = None,
    default_style: str | None = None,
    metric_choices: Sequence[str] | None = None,
    default_metric: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        default=str(ROOT_DIR),
        help="Root directory that contains one subdirectory per experiment.",
    )
    parser.add_argument(
        "--group",
        default=default_group,
        choices=sorted(GROUPS),
        help="Named experiment group from plot_config.py.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Explicit experiment labels. Overrides --group.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        help="Output formats, e.g. png pdf or png,pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Raster export DPI.",
    )
    if style_choices is not None:
        parser.add_argument(
            "--style",
            choices=tuple(style_choices),
            default=default_style or style_choices[0],
            help="Plot style variant.",
        )
    if metric_choices is not None:
        parser.add_argument(
            "--metric",
            choices=tuple(metric_choices),
            default=default_metric or metric_choices[0],
            help="Metric alias.",
        )
    return parser


def normalize_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    args.root = Path(args.root).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    args.formats = normalize_multi_value(args.formats)
    args.labels = normalize_multi_value(args.labels)
    return args


def normalize_multi_value(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                normalized.append(part)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in normalized:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def discover_experiment_dirs(root: Path | str) -> dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root_path}")
    discovered: dict[str, Path] = {}
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "summary.csv").exists():
            discovered[child.name] = child
    return discovered


def order_labels(labels: Sequence[str]) -> list[str]:
    labels_list = normalize_multi_value(labels)
    ordered: list[str] = []
    seen: set[str] = set()
    known = [label for label in ORDERED_LABELS if label in labels_list]
    extras = [label for label in labels_list if label not in ORDERED_LABELS]
    for label in [*known, *extras]:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def resolve_selected_labels(
    root: Path | str,
    *,
    group: str,
    labels: Sequence[str] | None,
) -> list[str]:
    available = discover_experiment_dirs(root)
    requested = order_labels(labels or GROUPS[group])
    selected: list[str] = []
    for label in requested:
        if label not in available:
            warnings.warn(
                f"Experiment label {label!r} is not available under {Path(root).expanduser().resolve()}; skipping.",
                stacklevel=2,
            )
            continue
        selected.append(label)
    if not selected:
        raise ValueError("No experiments remain after applying label selection.")
    return selected


def load_table_for_labels(root: Path | str, filename: str, labels: Sequence[str]) -> pd.DataFrame:
    root_path = Path(root).expanduser().resolve()
    experiment_dirs = discover_experiment_dirs(root_path)
    frames: list[pd.DataFrame] = []
    for label in labels:
        experiment_dir = experiment_dirs.get(label)
        if experiment_dir is None:
            warnings.warn(f"Experiment label {label!r} is missing from {root_path}; skipping.", stacklevel=2)
            continue
        csv_path = experiment_dir / filename
        if not csv_path.exists():
            warnings.warn(f"Missing {filename} for experiment {label!r}: {csv_path}", stacklevel=2)
            continue
        frame = pd.read_csv(csv_path)
        frame["label"] = label
        frame["display_name"] = display_name_for(label)
        frame["experiment_dir"] = str(experiment_dir)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No rows could be loaded from {filename}.")
    combined = pd.concat(frames, ignore_index=True)
    return sort_by_labels(combined, labels)


def sort_by_labels(frame: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    labels_list = list(labels)
    filtered = frame.loc[frame["label"].isin(labels_list)].copy()
    filtered["label"] = pd.Categorical(filtered["label"], categories=labels_list, ordered=True)
    filtered = filtered.sort_values(["label"], kind="stable")
    filtered["label"] = filtered["label"].astype(str)
    return filtered.reset_index(drop=True)


def display_name_for(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def color_for(label: str) -> str:
    return COLORS.get(label, "#4C78A8")


def marker_for(label: str) -> str:
    return MARKERS.get(label, "o")


def validate_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def average_object_metric_across_seeds(
    frame: pd.DataFrame,
    *,
    labels: Sequence[str],
    value_column: str,
) -> pd.DataFrame:
    validate_columns(
        frame,
        ("label", "display_name", "object_id", value_column),
        context="per_object_summary.csv",
    )
    averaged = (
        frame.groupby(["label", "display_name", "object_id"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "seed_avg_value"})
    )
    return sort_by_labels(averaged, labels)


def ci_yerr(frame: pd.DataFrame, *, mean_col: str, low_col: str, high_col: str) -> np.ndarray:
    lower = frame[mean_col].to_numpy(dtype=float) - frame[low_col].to_numpy(dtype=float)
    upper = frame[high_col].to_numpy(dtype=float) - frame[mean_col].to_numpy(dtype=float)
    return np.vstack([lower, upper])


def set_default_axis_style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_label_ticks(ax: plt.Axes, frame: pd.DataFrame) -> None:
    positions = np.arange(len(frame))
    ax.set_xticks(positions)
    ax.set_xticklabels(frame["display_name"].tolist(), rotation=25, ha="right")


def save_figure(
    fig: plt.Figure,
    *,
    out_dir: Path | str,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written_paths.append(path)
    return written_paths


def print_written_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        print(path)


def add_zero_reference(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="#6E6E6E", linewidth=1.0, linestyle="--", alpha=0.8)


def jitter_positions(center: float, count: int, *, spread: float = 0.10) -> np.ndarray:
    if count <= 1:
        return np.asarray([center], dtype=float)
    offsets = np.linspace(-spread, spread, num=count, dtype=float)
    return center + offsets


def draw_point_with_ci(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    mean_col: str,
    low_col: str,
    high_col: str,
) -> None:
    validate_columns(frame, ("label", mean_col, low_col, high_col), context="plot data")
    positions = np.arange(len(frame), dtype=float)
    yerr = ci_yerr(frame, mean_col=mean_col, low_col=low_col, high_col=high_col)
    for index, row in frame.reset_index(drop=True).iterrows():
        ax.errorbar(
            positions[index],
            float(row[mean_col]),
            yerr=np.asarray([[yerr[0, index]], [yerr[1, index]]], dtype=float),
            fmt=marker_for(str(row["label"])),
            color=color_for(str(row["label"])),
            markersize=7,
            elinewidth=1.4,
            capsize=4,
            linewidth=1.4,
        )


def draw_bar_with_ci(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    mean_col: str,
    low_col: str,
    high_col: str,
) -> None:
    validate_columns(frame, ("label", mean_col, low_col, high_col), context="plot data")
    positions = np.arange(len(frame), dtype=float)
    ax.bar(
        positions,
        frame[mean_col].to_numpy(dtype=float),
        color=[color_for(str(label)) for label in frame["label"]],
        width=0.72,
    )
    ax.errorbar(
        positions,
        frame[mean_col].to_numpy(dtype=float),
        yerr=ci_yerr(frame, mean_col=mean_col, low_col=low_col, high_col=high_col),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.2,
        capsize=4,
    )
