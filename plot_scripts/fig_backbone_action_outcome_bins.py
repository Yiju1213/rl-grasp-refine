from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from plot_common import (
    add_zero_reference,
    maybe_print_plot_data,
    print_written_paths,
    save_figure,
    set_default_axis_style,
    plt,
)

FIGURE_STEM = "fig_backbone_action_outcome_bins"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "outputs/unseen_test_formal_paper_seed7_episode"
DEFAULT_OUT_DIR = REPO_ROOT / "plot_scripts/generated/backbone_action_outcome"

BIN_LABELS = ("bin0", "bin1", "bin2")
BIN_EDGES = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float)

METHOD_DIRS = {
    "Rand. Action": "rand-action",
    "CNNMCA": "cnnmca-table",
    "CNNMCA-allgeom": "cnnmca-table-allgeom",
    "3D-DGCNN": "dgcnn",
    "SGA-GSN": "full-sga-gsn",
}

GROUPS = {
    "backbone": ("CNNMCA", "CNNMCA-allgeom", "3D-DGCNN", "SGA-GSN"),
    "with-rand": ("Rand. Action", "CNNMCA", "CNNMCA-allgeom", "3D-DGCNN", "SGA-GSN"),
}

METHOD_COLORS = {
    "Rand. Action": "#7A7A7A",
    "CNNMCA": "#D55E00",
    "CNNMCA-allgeom": "#009E73",
    "3D-DGCNN": "#C44E52",
    "SGA-GSN": "#1F4E79",
}

PANEL_SPECS = (
    {
        "column": "distribution_pct",
        "title": "Action-Bin Distribution",
        "ylabel": "Episode Ratio (%)",
        "lower": -5.0,
        "zero_reference": True,
    },
    {
        "column": "prob_delta_pct",
        "title": "Bin-Wise Prob Delta",
        "ylabel": "Mean Prob Delta (%)",
        "lower": -5.0,
        "zero_reference": True,
    },
    {
        "column": "success_delta_pct",
        "title": "Bin-Wise Success Delta",
        "ylabel": "Mean Success Delta (%)",
        "lower": -10.0,
        "zero_reference": True,
    },
)


def validate_episode_frame(frame: pd.DataFrame, *, path: Path) -> None:
    required = [
        *[f"action_{index}" for index in range(6)],
        "prob_delta",
        "success_delta",
    ]
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_episode_records(root: Path | str, method: str) -> pd.DataFrame:
    root_path = Path(root).expanduser().resolve()
    experiment_dir = METHOD_DIRS[method]
    path = root_path / experiment_dir / "episode_records.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing episode_records.csv for {method}: {path}")
    frame = pd.read_csv(path)
    validate_episode_frame(frame, path=path)
    return frame


def assign_6dof_bins(frame: pd.DataFrame) -> pd.Series:
    action_columns = [f"action_{index}" for index in range(6)]
    actions = frame[action_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    alpha = np.linalg.norm(actions, axis=1) / math.sqrt(6.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    bin_ids = np.digitize(alpha, BIN_EDGES[1:-1], right=False)
    return pd.Series(bin_ids.astype(int), index=frame.index, name="bin_id")


def summarize_method(frame: pd.DataFrame, *, method: str) -> pd.DataFrame:
    work = frame.copy()
    work["bin_id"] = assign_6dof_bins(work)
    total_count = int(len(work))
    if total_count <= 0:
        raise ValueError(f"No episode rows are available for {method}.")

    rows: list[dict[str, float | int | str]] = []
    for bin_id, bin_label in enumerate(BIN_LABELS):
        bin_frame = work.loc[work["bin_id"] == bin_id]
        count = int(len(bin_frame))
        row: dict[str, float | int | str] = {
            "method": method,
            "bin": bin_label,
            "bin_id": int(bin_id),
            "count": count,
            "total_count": total_count,
            "distribution_pct": 100.0 * count / total_count,
        }
        if count == 0:
            row.update(
                {
                    "prob_delta_pct": np.nan,
                    "success_delta_pct": np.nan,
                }
            )
        else:
            row.update(
                {
                    "prob_delta_pct": 100.0
                    * float(pd.to_numeric(bin_frame["prob_delta"], errors="coerce").mean()),
                    "success_delta_pct": 100.0
                    * float(pd.to_numeric(bin_frame["success_delta"], errors="coerce").mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_group_data(root: Path | str, *, group: str) -> pd.DataFrame:
    methods = GROUPS[group]
    frames: list[pd.DataFrame] = []
    for method in methods:
        episode_frame = read_episode_records(root, method)
        frames.append(summarize_method(episode_frame, method=method))
    combined = pd.concat(frames, ignore_index=True)
    combined["method_order"] = pd.Categorical(combined["method"], categories=list(methods), ordered=True)
    combined["bin_order"] = pd.Categorical(combined["bin"], categories=list(BIN_LABELS), ordered=True)
    combined = combined.sort_values(["method_order", "bin_order"], kind="stable")
    return combined.drop(columns=["method_order", "bin_order"]).reset_index(drop=True)


def finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)


def lower_limit(values: np.ndarray, *, default_lower: float) -> float:
    if values.size == 0:
        return default_lower
    min_value = float(np.nanmin(values))
    if min_value >= default_lower:
        return default_lower
    return float(math.floor(min_value / 5.0) * 5.0)


def upper_limit(values: np.ndarray, *, lower: float) -> float:
    if values.size == 0:
        return max(5.0, lower + 5.0)
    max_value = float(np.nanmax(values))
    if max_value <= lower:
        return lower + 5.0
    padding = max(2.0, 0.08 * (max_value - lower))
    return float(math.ceil((max_value + padding) / 5.0) * 5.0)


def set_axis_limits(ax: plt.Axes, frame: pd.DataFrame, *, column: str, default_lower: float) -> None:
    values = finite_values(frame, column)
    bottom = lower_limit(values, default_lower=default_lower)
    top = upper_limit(values, lower=bottom)
    ax.set_ylim(bottom, top)


def grouped_bar_geometry(methods: tuple[str, ...]) -> tuple[np.ndarray, float]:
    x = np.arange(len(BIN_LABELS), dtype=float)
    total_width = 0.84
    bar_width = total_width / max(len(methods), 1)
    return x, bar_width


def draw_grouped_bars(ax: plt.Axes, frame: pd.DataFrame, *, column: str, methods: tuple[str, ...]) -> None:
    x, bar_width = grouped_bar_geometry(methods)

    for method_index, method in enumerate(methods):
        method_frame = (
            frame.loc[frame["method"] == method]
            .set_index("bin")
            .reindex(BIN_LABELS)
            .reset_index()
        )
        values = pd.to_numeric(method_frame[column], errors="coerce").to_numpy(dtype=float)
        counts = pd.to_numeric(method_frame["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        positions = x + (method_index - (len(methods) - 1) / 2.0) * bar_width
        valid_mask = np.isfinite(values) & (counts > 0)
        if not valid_mask.any():
            warnings.warn(f"No finite values for {method} / {column}; skipping bars.", stacklevel=2)
            continue
        ax.bar(
            positions[valid_mask],
            values[valid_mask],
            width=bar_width * 0.92,
            label=method,
            color=METHOD_COLORS.get(method, "#4C78A8"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)


def missing_marker_y(ax: plt.Axes, *, column: str) -> float:
    bottom, top = ax.get_ylim()
    span = top - bottom
    if bottom <= 0.0 <= top:
        return 0.0
    return bottom + 0.08 * span


def draw_missing_markers(ax: plt.Axes, frame: pd.DataFrame, *, column: str, methods: tuple[str, ...]) -> None:
    x, bar_width = grouped_bar_geometry(methods)
    y = missing_marker_y(ax, column=column)
    for method_index, method in enumerate(methods):
        method_frame = (
            frame.loc[frame["method"] == method]
            .set_index("bin")
            .reindex(BIN_LABELS)
            .reset_index()
        )
        counts = pd.to_numeric(method_frame["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        missing_mask = counts <= 0
        if not missing_mask.any():
            continue
        positions = x + (method_index - (len(methods) - 1) / 2.0) * bar_width
        ax.scatter(
            positions[missing_mask],
            np.full(int(missing_mask.sum()), y, dtype=float),
            marker="x",
            s=92,
            linewidths=2.8,
            color=METHOD_COLORS.get(method, "#4C78A8"),
            zorder=8,
        )


def plot_group(plot_frame: pd.DataFrame, *, group: str, out_dir: Path | str, dpi: int) -> list[Path]:
    methods = GROUPS[group]
    fig, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(12.0, 3.8), sharex=False)

    for ax, spec in zip(axes, PANEL_SPECS, strict=True):
        column = str(spec["column"])
        draw_grouped_bars(ax, plot_frame, column=column, methods=methods)
        set_default_axis_style(ax)
        if bool(spec["zero_reference"]):
            add_zero_reference(ax)
        set_axis_limits(ax, plot_frame, column=column, default_lower=float(spec["lower"]))
        draw_missing_markers(ax, plot_frame, column=column, methods=methods)
        ax.set_title(str(spec["title"]), pad=8)
        ax.set_ylabel(str(spec["ylabel"]))
        ax.set_xlabel("6-DoF Action Bin")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(len(methods), 5),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    written = save_figure(fig, out_dir=out_dir, stem=f"{FIGURE_STEM}_{slug_for_group(group)}", formats=("png",), dpi=dpi)
    plt.close(fig)
    return written


def slug_for_group(group: str) -> str:
    return group.replace("-", "_")


def write_group_data(plot_frame: pd.DataFrame, *, group: str, out_dir: Path | str) -> Path:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{FIGURE_STEM}_{slug_for_group(group)}_data.csv"
    plot_frame.to_csv(path, index=False)
    return path


def selected_groups(group: str) -> tuple[str, ...]:
    if group == "all":
        return ("backbone", "with-rand")
    return (group,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot 6-DoF action-bin outcome diagnostics for formal backbone evaluations.",
    )
    parser.add_argument(
        "--group",
        choices=("all", "backbone", "with-rand"),
        default="all",
        help="Which built-in method group to plot.",
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Formal-test output root containing one directory per method.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for the combined PNG figure.",
    )
    parser.add_argument("--dpi", type=int, default=330, help="Raster export DPI.")
    print_group = parser.add_mutually_exclusive_group()
    print_group.add_argument(
        "--print-data",
        dest="print_data",
        action="store_true",
        default=True,
        help="Print the final data used by the plot. Enabled by default.",
    )
    print_group.add_argument(
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    written_paths: list[Path] = []
    data_by_group: dict[str, pd.DataFrame] = {}
    for group in selected_groups(str(args.group)):
        plot_frame = prepare_group_data(root, group=group)
        data_by_group[group] = plot_frame
        written_paths.extend(plot_group(plot_frame, group=group, out_dir=out_dir, dpi=int(args.dpi)))
        written_paths.append(write_group_data(plot_frame, group=group, out_dir=out_dir))

    print_written_paths(written_paths)
    maybe_print_plot_data(args, data_by_group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
