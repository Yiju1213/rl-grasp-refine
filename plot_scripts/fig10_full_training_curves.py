from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from plot_common import (
    FIGURE_SIZE_4_3,
    add_zero_reference,
    maybe_print_plot_data,
    normalize_multi_value,
    percent_label,
    print_written_paths,
    save_figure,
    set_default_axis_style,
    plt,
)

FIGURE_STEM = "fig10_full_training_curves"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIRS = (
    REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus",
    REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_seed8",
    REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_seed9",
)
DEFAULT_LABELS = ("seed7", "seed8", "seed9")
DEFAULT_METRICS = (
    "validation/outcome/success_lift_vs_dataset",
    "outcome/success_lift_vs_dataset",
)

METRIC_DISPLAY_NAMES = {
    "validation/outcome/success_lift_vs_dataset": "Validation",
    "outcome/success_lift_vs_dataset": "Training",
}
METRIC_STYLES = {
    "validation/outcome/success_lift_vs_dataset": {
        "color": "#1F4E79",
        "linestyle": "-",
        "linewidth": 2.4,
        "alpha": 0.20,
        "zorder": 3,
    },
    "outcome/success_lift_vs_dataset": {
        "color": "#6BAED6",
        "linestyle": "--",
        "linewidth": 1.9,
        "alpha": 0.14,
        "zorder": 2,
    },
}


def read_metrics_jsonl(run_dir: Path | str, *, seed_label: str, metrics: Sequence[str]) -> pd.DataFrame:
    run_path = Path(run_dir).expanduser().resolve()
    metrics_path = run_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.jsonl for {seed_label}: {metrics_path}")

    requested_metrics = normalize_multi_value(metrics)
    rows: list[dict[str, float | int | str]] = []
    seen_metrics: set[str] = set()
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {metrics_path}:{line_number}") from exc
            step = payload.get("step")
            stats = payload.get("stats", {})
            if step is None or not isinstance(stats, dict):
                continue
            for metric in requested_metrics:
                if metric not in stats:
                    continue
                value = pd.to_numeric(pd.Series([stats[metric]]), errors="coerce").iloc[0]
                rows.append(
                    {
                        "seed_label": seed_label,
                        "run_dir": str(run_path),
                        "step": int(step),
                        "metric": metric,
                        "raw_value": float(value) if pd.notna(value) else np.nan,
                    }
                )
                seen_metrics.add(metric)

    missing_metrics = [metric for metric in requested_metrics if metric not in seen_metrics]
    for metric in missing_metrics:
        warnings.warn(
            f"Metric {metric!r} was not found in {metrics_path}; it will be absent for {seed_label}.",
            stacklevel=2,
        )
    if not rows:
        raise ValueError(f"No requested metrics were found in {metrics_path}.")
    return pd.DataFrame(rows)


def load_training_metrics(
    run_dirs: Sequence[Path | str],
    labels: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    run_dir_list = list(run_dirs)
    label_list = normalize_multi_value(labels)
    if len(run_dir_list) != len(label_list):
        raise ValueError(f"--run-dirs count ({len(run_dir_list)}) must match --labels count ({len(label_list)}).")
    frames = [
        read_metrics_jsonl(run_dir, seed_label=label, metrics=metrics)
        for run_dir, label in zip(run_dir_list, label_list, strict=True)
    ]
    return pd.concat(frames, ignore_index=True)


def smooth_seed_metrics(frame: pd.DataFrame, *, smooth_window: int) -> pd.DataFrame:
    if smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1.")
    required = {"seed_label", "step", "metric", "raw_value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training metrics data is missing columns: {missing}")

    smoothed_parts: list[pd.DataFrame] = []
    for _, group in frame.sort_values(["seed_label", "metric", "step"]).groupby(["seed_label", "metric"], sort=False):
        part = group.copy()
        part["smoothed_value"] = (
            pd.to_numeric(part["raw_value"], errors="coerce")
            .rolling(window=smooth_window, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
        smoothed_parts.append(part)
    if not smoothed_parts:
        raise ValueError("No training metrics are available to smooth.")
    return pd.concat(smoothed_parts, ignore_index=True)


def aggregate_training_curves(
    smoothed_frame: pd.DataFrame,
    *,
    labels: Sequence[str],
    metrics: Sequence[str],
    align: str,
) -> pd.DataFrame:
    if align not in {"common", "available"}:
        raise ValueError(f"Unknown alignment mode: {align!r}")
    label_list = normalize_multi_value(labels)
    metric_list = normalize_multi_value(metrics)
    rows: list[pd.DataFrame] = []

    for metric in metric_list:
        metric_frame = smoothed_frame.loc[smoothed_frame["metric"] == metric].copy()
        if metric_frame.empty:
            warnings.warn(f"No rows are available for metric {metric!r}; skipping.", stacklevel=2)
            continue
        smooth_wide = metric_frame.pivot_table(
            index="step",
            columns="seed_label",
            values="smoothed_value",
            aggfunc="mean",
        )
        raw_wide = metric_frame.pivot_table(
            index="step",
            columns="seed_label",
            values="raw_value",
            aggfunc="mean",
        )
        for label in label_list:
            if label not in smooth_wide.columns:
                smooth_wide[label] = np.nan
            if label not in raw_wide.columns:
                raw_wide[label] = np.nan
        smooth_wide = smooth_wide[label_list].sort_index()
        raw_wide = raw_wide[label_list].reindex(smooth_wide.index)
        if align == "common":
            keep_mask = smooth_wide.notna().all(axis=1)
        else:
            keep_mask = smooth_wide.notna().any(axis=1)
        smooth_wide = smooth_wide.loc[keep_mask]
        raw_wide = raw_wide.loc[keep_mask]
        if smooth_wide.empty:
            warnings.warn(f"No aligned steps remain for metric {metric!r} with align={align!r}; skipping.", stacklevel=2)
            continue

        summary = pd.DataFrame(
            {
                "step": smooth_wide.index.astype(int),
                "metric": metric,
                "metric_display": METRIC_DISPLAY_NAMES.get(metric, metric),
                "mean": smooth_wide.mean(axis=1, skipna=True).to_numpy(dtype=float),
                "std": smooth_wide.std(axis=1, skipna=True, ddof=0).fillna(0.0).to_numpy(dtype=float),
                "num_seeds": smooth_wide.notna().sum(axis=1).astype(int).to_numpy(),
            }
        )
        for label in label_list:
            summary[f"raw_{label}"] = raw_wide[label].to_numpy(dtype=float)
            summary[f"smooth_{label}"] = smooth_wide[label].to_numpy(dtype=float)
        rows.append(summary)

    if not rows:
        raise ValueError("No training curve data remains after alignment.")
    combined = pd.concat(rows, ignore_index=True)
    combined["metric_order"] = pd.Categorical(combined["metric"], categories=metric_list, ordered=True)
    combined = combined.sort_values(["metric_order", "step"], kind="stable").drop(columns=["metric_order"])
    return combined.reset_index(drop=True)


def prepare_data(
    run_dirs: Sequence[Path | str],
    labels: Sequence[str],
    metrics: Sequence[str],
    *,
    align: str,
    smooth_window: int,
) -> pd.DataFrame:
    raw_frame = load_training_metrics(run_dirs, labels, metrics)
    smoothed_frame = smooth_seed_metrics(raw_frame, smooth_window=smooth_window)
    return aggregate_training_curves(smoothed_frame, labels=labels, metrics=metrics, align=align)


def plot_training_curves(plot_frame: pd.DataFrame, *, dpi: int, out_dir: Path | str) -> list[Path]:
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
    ax.set_ylabel(percent_label("Success Lift vs Dataset"))
    ax.legend(frameon=False, loc="best")
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
        description="Plot Full policy 3-seed training curves from metrics.jsonl.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_RUN_DIRS],
        help="Training run directories. Each directory must contain metrics.jsonl.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Seed labels used for exported raw/smoothed columns.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric keys to extract from stats in metrics.jsonl.",
    )
    parser.add_argument(
        "--align",
        choices=("common", "available"),
        default="common",
        help="common keeps only steps present for all seeds; available uses all steps with at least one seed.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Rolling window over available points per seed/metric. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "plot_scripts/generated/training"),
        help="Output directory for the PNG and exported plot data CSV.",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    labels = normalize_multi_value(args.labels)
    metrics = normalize_multi_value(args.metrics)
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    out_dir = Path(args.out_dir).expanduser().resolve()

    plot_frame = prepare_data(
        run_dirs,
        labels,
        metrics,
        align=args.align,
        smooth_window=args.smooth_window,
    )
    written = plot_training_curves(plot_frame, dpi=args.dpi, out_dir=out_dir)
    data_path = write_plot_data(plot_frame, out_dir=out_dir)
    print_written_paths([*written, data_path])
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
