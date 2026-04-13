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
    ADJUSTED_METRIC_DISPLAY_NAMES,
    BASELINE_LABEL,
    COLORS,
    CONFIDENCE_LEVEL,
    DEFAULT_DPI,
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_FORMATS,
    DEFAULT_OUT_DIR,
    DISPLAY_NAMES,
    FONT_FAMILY,
    FONT_SIZES,
    GROUPS,
    MARKERS,
    ORDERED_LABELS,
    ROOT_DIR,
)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "legend.fontsize": FONT_SIZES["legend"],
        }
    )


apply_plot_style()

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
        "column": "adjusted_stability",
        "ylabel": "IQR of Success Gain over No-Action",
    },
    "std": {
        "column": "adjusted_stability",
        "ylabel": "Std of Success Gain over No-Action",
    },
}

ADJUSTED_METRIC_SPECS = {
    "success_gain": {
        "source": "success_lift_vs_dataset",
        "mean": "adjusted_mean",
        "ci_low": "adjusted_ci95_low",
        "ci_high": "adjusted_ci95_high",
        "ylabel": ADJUSTED_METRIC_DISPLAY_NAMES["success_gain"],
        "short_label": "Success Gain",
    },
    "excess_degradation": {
        "source": "positive_drop_rate",
        "mean": "adjusted_mean",
        "ci_low": "adjusted_ci95_low",
        "ci_high": "adjusted_ci95_high",
        "ylabel": ADJUSTED_METRIC_DISPLAY_NAMES["excess_degradation"],
        "short_label": "Excess Degradation",
    },
    "excess_recovery": {
        "source": "negative_hold_rate",
        "mean": "adjusted_mean",
        "ci_low": "adjusted_ci95_low",
        "ci_high": "adjusted_ci95_high",
        "ylabel": ADJUSTED_METRIC_DISPLAY_NAMES["excess_recovery"],
        "short_label": "Excess Recovery",
    },
    "excess_t_cover_delta": {
        "source": "t_cover_delta_mean",
        "mean": "adjusted_mean",
        "ci_low": "adjusted_ci95_low",
        "ci_high": "adjusted_ci95_high",
        "ylabel": ADJUSTED_METRIC_DISPLAY_NAMES["excess_t_cover_delta"],
        "short_label": "Excess T-Cover Delta",
    },
    "excess_t_edge_delta": {
        "source": "t_edge_delta_mean",
        "mean": "adjusted_mean",
        "ci_low": "adjusted_ci95_low",
        "ci_high": "adjusted_ci95_high",
        "ylabel": ADJUSTED_METRIC_DISPLAY_NAMES["excess_t_edge_delta"],
        "short_label": "Excess T-Edge Delta",
    },
    "excess_probability_delta": {
        "source": "prob_delta_mean",
        "mean": "adjusted_mean",
        "ci_low": "adjusted_ci95_low",
        "ci_high": "adjusted_ci95_high",
        "ylabel": ADJUSTED_METRIC_DISPLAY_NAMES["excess_probability_delta"],
        "short_label": "Excess Probability Delta",
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
        help="Base directory for generated figures. The selected --group is appended automatically.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Raster export DPI.",
    )
    print_data_group = parser.add_mutually_exclusive_group()
    print_data_group.add_argument(
        "--print-data",
        dest="print_data",
        action="store_true",
        default=True,
        help="Print the final data frames used by the plot. Enabled by default.",
    )
    print_data_group.add_argument(
        "--no-print-data",
        dest="print_data",
        action="store_false",
        help="Do not print final plot data.",
    )
    parser.add_argument(
        "--print-data-format",
        choices=("table", "csv"),
        default="table",
        help="Format used by --print-data.",
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
    args.out_dir = Path(args.out_dir).expanduser().resolve() / str(args.group)
    args.formats = list(DEFAULT_FORMATS)
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
    requested = order_labels(labels) if labels else normalize_multi_value(GROUPS[group])
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


def labels_with_baseline(
    labels: Sequence[str],
    *,
    baseline_label: str = BASELINE_LABEL,
) -> list[str]:
    labels_list = normalize_multi_value(labels)
    combined = [baseline_label]
    combined.extend(label for label in labels_list if label != baseline_label)
    return combined


def load_per_object_table_with_baseline(
    root: Path | str,
    labels: Sequence[str],
    *,
    baseline_label: str = BASELINE_LABEL,
) -> pd.DataFrame:
    root_path = Path(root).expanduser().resolve()
    experiment_dirs = discover_experiment_dirs(root_path)
    baseline_dir = experiment_dirs.get(baseline_label)
    if baseline_dir is None:
        raise ValueError(
            f"Adjusted metrics require baseline label {baseline_label!r}, "
            f"but it is missing under {root_path}."
        )
    baseline_csv = baseline_dir / "per_object_summary.csv"
    if not baseline_csv.exists():
        raise ValueError(f"Adjusted metrics require baseline per_object_summary.csv: {baseline_csv}")
    return load_table_for_labels(root_path, "per_object_summary.csv", labels_with_baseline(labels, baseline_label=baseline_label))


def adjusted_metric_source_column(metric_key: str) -> str:
    try:
        return str(ADJUSTED_METRIC_SPECS[metric_key]["source"])
    except KeyError as exc:
        raise ValueError(f"Unknown adjusted metric key: {metric_key!r}") from exc


def compute_adjusted_per_object_values(
    per_object_frame: pd.DataFrame,
    labels: Sequence[str],
    *,
    metric_key: str | None = None,
    value_column: str | None = None,
    baseline_label: str = BASELINE_LABEL,
) -> pd.DataFrame:
    if value_column is None:
        if metric_key is None:
            raise ValueError("Either metric_key or value_column must be provided.")
        value_column = adjusted_metric_source_column(metric_key)

    labels_list = normalize_multi_value(labels)
    validate_columns(
        per_object_frame,
        ("label", "display_name", "test_seed", "object_id", value_column),
        context="per_object_summary.csv",
    )
    if not labels_list:
        raise ValueError("No labels were provided for adjusted metric computation.")

    baseline_frame = per_object_frame.loc[per_object_frame["label"] == baseline_label].copy()
    if baseline_frame.empty:
        raise ValueError(f"Adjusted metrics require baseline rows for label {baseline_label!r}.")
    baseline_values = (
        baseline_frame[["test_seed", "object_id", value_column]]
        .assign(**{value_column: lambda frame: pd.to_numeric(frame[value_column], errors="coerce")})
        .groupby(["test_seed", "object_id"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "baseline_value"})
    )

    selected = per_object_frame.loc[per_object_frame["label"].isin(labels_list)].copy()
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    selected_values = (
        selected[["label", "display_name", "test_seed", "object_id", value_column]]
        .groupby(["label", "display_name", "test_seed", "object_id"], as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "observed_value"})
    )
    merged = selected_values.merge(baseline_values, on=["test_seed", "object_id"], how="left", validate="many_to_one")
    if merged["baseline_value"].isna().any():
        missing_pairs = merged.loc[merged["baseline_value"].isna(), ["test_seed", "object_id"]].drop_duplicates()
        examples = missing_pairs.head(5).to_dict("records")
        raise ValueError(f"Baseline {baseline_label!r} is missing paired object rows, examples: {examples}")
    merged["adjusted_value"] = merged["observed_value"] - merged["baseline_value"]
    return sort_by_labels(merged, labels_list)


def average_adjusted_object_metric_across_seeds(
    adjusted_frame: pd.DataFrame,
    *,
    labels: Sequence[str],
) -> pd.DataFrame:
    validate_columns(
        adjusted_frame,
        ("label", "display_name", "object_id", "adjusted_value"),
        context="adjusted per-object data",
    )
    averaged = (
        adjusted_frame.groupby(["label", "display_name", "object_id"], as_index=False)["adjusted_value"]
        .mean()
        .rename(columns={"adjusted_value": "seed_avg_value"})
    )
    return sort_by_labels(averaged, labels)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        raise ValueError("Cannot bootstrap CI from zero finite values.")
    mean = float(array.mean())
    if array.size == 1 or iterations <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(iterations, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return mean, float(low), float(high)


def summarize_adjusted_experiment(
    adjusted_frame: pd.DataFrame,
    labels: Sequence[str],
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    object_frame = average_adjusted_object_metric_across_seeds(adjusted_frame, labels=labels)
    rows: list[dict[str, float | str]] = []
    for index, label in enumerate(normalize_multi_value(labels)):
        label_values = object_frame.loc[object_frame["label"] == label, "seed_avg_value"].to_numpy(dtype=float)
        if label_values.size == 0:
            continue
        mean, low, high = bootstrap_mean_ci(
            label_values,
            iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            seed=seed + index,
        )
        rows.append(
            {
                "label": label,
                "display_name": display_name_for(label),
                "adjusted_mean": mean,
                "adjusted_ci95_low": low,
                "adjusted_ci95_high": high,
                "num_objects": int(np.isfinite(label_values).sum()),
            }
        )
    if not rows:
        raise ValueError("No adjusted experiment summaries could be computed.")
    return sort_by_labels(pd.DataFrame(rows), labels)


def summarize_adjusted_runs(adjusted_frame: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    validate_columns(
        adjusted_frame,
        ("label", "display_name", "test_seed", "adjusted_value"),
        context="adjusted per-object data",
    )
    run_frame = (
        adjusted_frame.groupby(["label", "display_name", "test_seed"], as_index=False)["adjusted_value"]
        .mean()
        .rename(columns={"adjusted_value": "adjusted_run_mean"})
    )
    return sort_by_labels(run_frame, labels)


def summarize_adjusted_object_stability(
    object_frame: pd.DataFrame,
    labels: Sequence[str],
    *,
    metric: str,
) -> pd.DataFrame:
    if metric not in STABILITY_METRIC_SPECS:
        raise ValueError(f"Unknown stability metric: {metric!r}")
    validate_columns(
        object_frame,
        ("label", "display_name", "seed_avg_value"),
        context="adjusted object-level data",
    )
    rows: list[dict[str, float | str]] = []
    for label in normalize_multi_value(labels):
        values = (
            pd.to_numeric(object_frame.loc[object_frame["label"] == label, "seed_avg_value"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if values.size == 0:
            continue
        if metric == "iqr":
            value = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        else:
            value = float(np.std(values, ddof=0))
        rows.append(
            {
                "label": label,
                "display_name": display_name_for(label),
                "adjusted_stability": value,
                "num_objects": int(values.size),
            }
        )
    if not rows:
        raise ValueError("No adjusted object-stability summaries could be computed.")
    return sort_by_labels(pd.DataFrame(rows), labels)


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


def set_default_axis_style(ax: plt.Axes, *, boxed: bool = True) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(boxed)
        spine.set_color("#4A4A4A")
        spine.set_linewidth(1.0)
    if not boxed:
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="both", color="#4A4A4A", width=1.0)


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


def _format_plot_data(frame: pd.DataFrame, *, output_format: str) -> str:
    if output_format == "csv":
        return frame.to_csv(index=False).rstrip()
    return frame.to_string(index=False)


def maybe_print_plot_data(
    args: argparse.Namespace,
    data: pd.DataFrame | dict[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
) -> None:
    if not bool(getattr(args, "print_data", False)):
        return
    output_format = str(getattr(args, "print_data_format", "table"))
    if isinstance(data, pd.DataFrame):
        print("\n[plot-data]")
        print(_format_plot_data(data, output_format=output_format))
        return
    items = data.items() if isinstance(data, dict) else data
    for name, frame in items:
        print(f"\n[plot-data:{name}]")
        print(_format_plot_data(frame, output_format=output_format))


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
