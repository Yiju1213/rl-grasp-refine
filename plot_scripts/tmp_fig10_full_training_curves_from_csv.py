# tmp: Replot Fig. 10 from exported plot data when original seed metrics are unavailable.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fig10_full_training_curves import FIGURE_STEM as SOURCE_FIGURE_STEM
from fig10_full_training_curves import METRIC_STYLES
from plot_common import (
    FIGURE_SIZE_4_3,
    add_zero_reference,
    maybe_print_plot_data,
    print_written_paths,
    save_figure,
    set_default_axis_style,
    plt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "plot_scripts/generated/training/fig10_full_training_curves_data.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "plot_scripts/generated/training"
FIGURE_STEM = "tmp_fig10_full_training_curves_from_csv"
Y_LABEL = "Success Gain over \nDataset Legacy Grasp Outcome (%)"


def load_plot_data(path: Path | str) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"step", "metric", "metric_display", "mean", "std"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{csv_path} contains no plot rows.")
    frame = frame.copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(int)
    frame["mean"] = pd.to_numeric(frame["mean"], errors="raise")
    frame["std"] = pd.to_numeric(frame["std"], errors="raise")
    return frame


def plot_training_curves_from_csv(plot_frame: pd.DataFrame, *, dpi: int, out_dir: Path | str) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_4_3)
    for metric, metric_frame in plot_frame.groupby("metric", sort=False):
        style = METRIC_STYLES.get(
            str(metric),
            {"color": "#4C78A8", "linestyle": "-", "linewidth": 2.0, "alpha": 0.16, "zorder": 2},
        )
        ordered = metric_frame.sort_values("step")
        x = ordered["step"].to_numpy(dtype=float)
        mean = ordered["mean"].to_numpy(dtype=float)
        std = ordered["std"].to_numpy(dtype=float)
        label = str(ordered["metric_display"].iloc[0])
        ax.plot(
            x,
            mean,
            label=label,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=float(style["linewidth"]),
            zorder=int(style["zorder"]),
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=str(style["color"]),
            alpha=float(style["alpha"]),
            linewidth=0,
            zorder=int(style["zorder"]) - 1,
        )

    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(Y_LABEL)
    ax.legend(frameon=False, loc="best")
    written = save_figure(fig, out_dir=out_dir, stem=FIGURE_STEM, formats=("png",), dpi=dpi)
    plt.close(fig)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "tmp: Replot Full policy 3-seed training curves from "
            f"{SOURCE_FIGURE_STEM}_data.csv instead of metrics.jsonl."
        )
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=330)
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
    parser.add_argument("--print-data-format", choices=("table", "csv"), default="table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_frame = load_plot_data(args.input_csv)
    written = plot_training_curves_from_csv(
        plot_frame,
        dpi=int(args.dpi),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )
    print_written_paths(written)
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
