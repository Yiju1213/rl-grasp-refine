from __future__ import annotations

import argparse
import math
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

FIGURE_STEM = "fig_backbone_action_outcome_bins_table"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "plot_scripts/generated/backbone_action_outcome_table"

BIN_LABELS = ("bin0", "bin1", "bin2")
BIN_DISPLAY_NAMES = ("Low", "Mid", "High")
METHODS = ("CNNMCA", "CNNMCA-allgeom", "3D-DGCNN", "SGA-GSN")

METHOD_COLORS = {
    "CNNMCA": "#D55E00",
    "CNNMCA-allgeom": "#009E73",
    "3D-DGCNN": "#C44E52",
    "SGA-GSN": "#1F4E79",
}

TABLE_ROWS = (
    ("CNNMCA", "bin0", None, None, None, None),
    ("CNNMCA", "bin1", None, None, None, None),
    ("CNNMCA", "bin2", 3900, 100.0, +1.01, -8.36),
    ("CNNMCA-allgeom", "bin0", None, None, None, None),
    ("CNNMCA-allgeom", "bin1", None, None, None, None),
    ("CNNMCA-allgeom", "bin2", 3900, 100.0, +1.13, -7.79),
    ("3D-DGCNN", "bin0", 1358, 34.8, -0.71, -4.79),
    ("3D-DGCNN", "bin1", 1431, 36.7, +5.74, +7.27),
    ("3D-DGCNN", "bin2", 1111, 28.5, +16.25, +22.95),
    ("SGA-GSN", "bin0", 387, 9.9, -0.30, -4.13),
    ("SGA-GSN", "bin1", 3039, 77.9, +4.51, +10.99),
    ("SGA-GSN", "bin2", 474, 12.2, +9.22, +27.43),
)

PANEL_SPECS = (
    {
        "column": "distribution_pct",
        "caption": "(a) Action-Bin Distribution",
        "ylabel": "Episode Ratio (%)",
        "lower": -5.0,
        "zero_reference": True,
    },
    {
        "column": "prob_delta_pct",
        "caption": "(b) Bin-Wise Prob Delta",
        "ylabel": "Mean Prob Delta (%)",
        "lower": -5.0,
        "zero_reference": True,
    },
    {
        "column": "success_delta_pct",
        "caption": "(c) Bin-Wise Success Delta",
        "ylabel": "Mean Success Delta (%)",
        "lower": -10.0,
        "zero_reference": True,
    },
)


def prepare_table_data() -> pd.DataFrame:
    frame = pd.DataFrame(
        TABLE_ROWS,
        columns=(
            "method",
            "bin",
            "count",
            "distribution_pct",
            "prob_delta_pct",
            "success_delta_pct",
        ),
    )
    frame["bin_id"] = frame["bin"].map({label: index for index, label in enumerate(BIN_LABELS)})
    frame["bin_display"] = frame["bin"].map(dict(zip(BIN_LABELS, BIN_DISPLAY_NAMES, strict=True)))
    frame["total_count"] = 3900
    frame["method_order"] = pd.Categorical(frame["method"], categories=list(METHODS), ordered=True)
    frame["bin_order"] = pd.Categorical(frame["bin"], categories=list(BIN_LABELS), ordered=True)
    frame = frame.sort_values(["method_order", "bin_order"], kind="stable")
    return frame.drop(columns=["method_order", "bin_order"]).reset_index(drop=True)


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


def grouped_bar_geometry() -> tuple[np.ndarray, float]:
    x = np.arange(len(BIN_LABELS), dtype=float)
    total_width = 0.84
    bar_width = total_width / len(METHODS)
    return x, bar_width


def draw_grouped_bars(ax: plt.Axes, frame: pd.DataFrame, *, column: str) -> None:
    x, bar_width = grouped_bar_geometry()
    for method_index, method in enumerate(METHODS):
        method_frame = frame.loc[frame["method"] == method].set_index("bin").reindex(BIN_LABELS).reset_index()
        values = pd.to_numeric(method_frame[column], errors="coerce").to_numpy(dtype=float)
        counts = pd.to_numeric(method_frame["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        positions = x + (method_index - (len(METHODS) - 1) / 2.0) * bar_width
        valid_mask = np.isfinite(values) & (counts > 0)
        if not valid_mask.any():
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
    ax.set_xticklabels(BIN_DISPLAY_NAMES)


def missing_marker_y(ax: plt.Axes) -> float:
    bottom, top = ax.get_ylim()
    if bottom <= 0.0 <= top:
        return 0.0
    return bottom + 0.08 * (top - bottom)


def draw_missing_markers(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x, bar_width = grouped_bar_geometry()
    y = missing_marker_y(ax)
    for method_index, method in enumerate(METHODS):
        method_frame = frame.loc[frame["method"] == method].set_index("bin").reindex(BIN_LABELS).reset_index()
        counts = pd.to_numeric(method_frame["count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        missing_mask = counts <= 0
        if not missing_mask.any():
            continue
        positions = x + (method_index - (len(METHODS) - 1) / 2.0) * bar_width
        ax.scatter(
            positions[missing_mask],
            np.full(int(missing_mask.sum()), y, dtype=float),
            marker="x",
            s=92,
            linewidths=2.8,
            color=METHOD_COLORS.get(method, "#4C78A8"),
            zorder=8,
        )


def plot_table_data(plot_frame: pd.DataFrame, *, out_dir: Path | str, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(12.0, 5.5), sharex=False)
    for ax, spec in zip(axes, PANEL_SPECS, strict=True):
        column = str(spec["column"])
        draw_grouped_bars(ax, plot_frame, column=column)
        set_default_axis_style(ax)
        if bool(spec["zero_reference"]):
            add_zero_reference(ax)
        set_axis_limits(ax, plot_frame, column=column, default_lower=float(spec["lower"]))
        draw_missing_markers(ax, plot_frame)
        ax.set_ylabel(str(spec["ylabel"]))
        ax.set_xlabel("6-DoF Action Magnitude", labelpad=8)
        ax.text(
            0.5,
            -0.3,
            str(spec["caption"]),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(METHODS),
        frameon=False,
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    written = save_figure(fig, out_dir=out_dir, stem=FIGURE_STEM, formats=("png",), dpi=dpi)
    plt.close(fig)
    return written


def write_plot_data(plot_frame: pd.DataFrame, *, out_dir: Path | str) -> Path:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{FIGURE_STEM}_data.csv"
    plot_frame.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot backbone action-outcome bins directly from the paper table values.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for the table-data PNG and CSV.",
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
    out_dir = Path(args.out_dir).expanduser().resolve()
    plot_frame = prepare_table_data()
    written_paths = plot_table_data(plot_frame, out_dir=out_dir, dpi=int(args.dpi))
    data_path = write_plot_data(plot_frame, out_dir=out_dir)
    print_written_paths([*written_paths, data_path])
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
