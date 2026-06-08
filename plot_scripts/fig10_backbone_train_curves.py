from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from plot_common import (
    FIGURE_SIZE_4_3,
    add_zero_reference,
    maybe_print_plot_data,
    print_written_paths,
    save_figure,
    set_default_axis_style,
    plt,
)

FIGURE_STEM = "fig10_backbone_train_curves"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC = "outcome/success_lift_vs_dataset"
DEFAULT_SEED_LABELS = ("seed7", "seed8", "seed9")
Y_AXIS_LABEL = "Success Gain over \nDataset Legacy Grasp Outcome (%)"

DEFAULT_MODEL_RUN_DIRS = {
    "sga-gsn": (
        REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus",
        REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_seed8",
        REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_seed9",
    ),
    "dgcnn": (
        REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_dgcnn",
        REPO_ROOT / "outputs/exp_debug/seed8_rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_dgcnn",
        REPO_ROOT / "outputs/exp_debug/seed9_rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_dgcnn",
    ),
    "cnnmca-table-allgeom": (
        REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table_allgeom",
        REPO_ROOT / "outputs/exp_debug/seed8_rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table_allgeom",
        REPO_ROOT / "outputs/exp_debug/seed9_rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table_allgeom",
    ),
    "cnnmca-table": (
        REPO_ROOT / "outputs/exp_debug/rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table",
        REPO_ROOT / "outputs/exp_debug/seed8_rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table",
        REPO_ROOT / "outputs/exp_debug/seed9_rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus_cnnmca_table",
    ),
}

MODEL_DISPLAY_NAMES = {
    "sga-gsn": "SGA-GSN",
    "dgcnn": "3D-DGCNN",
    "cnnmca-table": "CNNMCA",
    "cnnmca-table-allgeom": "CNNMCA-allgeom",
}

MODEL_STYLES = {
    "sga-gsn": {
        "color": "#1F4E79",
        "linewidth": 2.4,
        "alpha": 0.20,
        "line_zorder": 14,
        "shade_zorder": 4,
    },
    "dgcnn": {
        "color": "#C44E52",
        "linewidth": 2.2,
        "alpha": 0.18,
        "line_zorder": 12,
        "shade_zorder": 2,
    },
    "cnnmca-table-allgeom": {
        "color": "#009E73",
        "linewidth": 2.1,
        "alpha": 0.17,
        "line_zorder": 13,
        "shade_zorder": 3,
    },
    "cnnmca-table": {
        "color": "#D55E00",
        "linewidth": 2.1,
        "alpha": 0.17,
        "line_zorder": 11,
        "shade_zorder": 1,
    },
}


def read_metric_jsonl(
    run_dir: Path | str,
    *,
    model: str,
    seed_label: str,
    metric: str,
) -> pd.DataFrame:
    run_path = Path(run_dir).expanduser().resolve()
    metrics_path = run_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.jsonl for {model}/{seed_label}: {metrics_path}")

    rows: list[dict[str, float | int | str]] = []
    seen_metric = False
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
            if step is None or not isinstance(stats, dict) or metric not in stats:
                continue
            value = pd.to_numeric(pd.Series([stats[metric]]), errors="coerce").iloc[0]
            rows.append(
                {
                    "model": str(model),
                    "model_display": MODEL_DISPLAY_NAMES.get(str(model), str(model)),
                    "seed_label": str(seed_label),
                    "run_dir": str(run_path),
                    "step": int(step),
                    "raw_value": float(value) if pd.notna(value) else np.nan,
                }
            )
            seen_metric = True

    if not seen_metric:
        warnings.warn(
            f"Metric {metric!r} was not found in {metrics_path}; {model}/{seed_label} will be absent.",
            stacklevel=2,
        )
    if not rows:
        raise ValueError(f"No requested metric rows were found in {metrics_path}.")
    return pd.DataFrame(rows)


def load_backbone_metrics(*, metric: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model, run_dirs in DEFAULT_MODEL_RUN_DIRS.items():
        if len(run_dirs) != len(DEFAULT_SEED_LABELS):
            raise ValueError(f"{model} has {len(run_dirs)} run dirs, expected {len(DEFAULT_SEED_LABELS)}.")
        for run_dir, seed_label in zip(run_dirs, DEFAULT_SEED_LABELS, strict=True):
            frames.append(read_metric_jsonl(run_dir, model=model, seed_label=seed_label, metric=metric))
    return pd.concat(frames, ignore_index=True)


def smooth_seed_metrics(frame: pd.DataFrame, *, smooth_window: int) -> pd.DataFrame:
    if smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1.")
    required = {"model", "seed_label", "step", "raw_value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"backbone training metrics data is missing columns: {missing}")

    smoothed_parts: list[pd.DataFrame] = []
    sort_cols = ["model", "seed_label", "step"]
    for _, group in frame.sort_values(sort_cols).groupby(["model", "seed_label"], sort=False):
        part = group.copy()
        part["smoothed_value"] = (
            pd.to_numeric(part["raw_value"], errors="coerce")
            .rolling(window=smooth_window, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
        smoothed_parts.append(part)
    if not smoothed_parts:
        raise ValueError("No backbone training metrics are available to smooth.")
    return pd.concat(smoothed_parts, ignore_index=True)


def aggregate_backbone_curves(smoothed_frame: pd.DataFrame, *, align: str) -> pd.DataFrame:
    if align not in {"common", "available"}:
        raise ValueError(f"Unknown alignment mode: {align!r}")
    rows: list[pd.DataFrame] = []

    for model in DEFAULT_MODEL_RUN_DIRS:
        model_frame = smoothed_frame.loc[smoothed_frame["model"] == model].copy()
        if model_frame.empty:
            warnings.warn(f"No rows are available for model {model!r}; skipping.", stacklevel=2)
            continue
        smooth_wide = model_frame.pivot_table(
            index="step",
            columns="seed_label",
            values="smoothed_value",
            aggfunc="mean",
        )
        raw_wide = model_frame.pivot_table(
            index="step",
            columns="seed_label",
            values="raw_value",
            aggfunc="mean",
        )
        for seed_label in DEFAULT_SEED_LABELS:
            if seed_label not in smooth_wide.columns:
                smooth_wide[seed_label] = np.nan
            if seed_label not in raw_wide.columns:
                raw_wide[seed_label] = np.nan
        smooth_wide = smooth_wide[list(DEFAULT_SEED_LABELS)].sort_index()
        raw_wide = raw_wide[list(DEFAULT_SEED_LABELS)].reindex(smooth_wide.index)
        keep_mask = smooth_wide.notna().all(axis=1) if align == "common" else smooth_wide.notna().any(axis=1)
        smooth_wide = smooth_wide.loc[keep_mask]
        raw_wide = raw_wide.loc[keep_mask]
        if smooth_wide.empty:
            warnings.warn(f"No aligned steps remain for model {model!r} with align={align!r}; skipping.", stacklevel=2)
            continue

        summary = pd.DataFrame(
            {
                "model": model,
                "model_display": MODEL_DISPLAY_NAMES.get(model, model),
                "step": smooth_wide.index.astype(int),
                "mean": smooth_wide.mean(axis=1, skipna=True).to_numpy(dtype=float),
                "std": smooth_wide.std(axis=1, skipna=True, ddof=0).fillna(0.0).to_numpy(dtype=float),
                "num_seeds": smooth_wide.notna().sum(axis=1).astype(int).to_numpy(),
            }
        )
        for seed_label in DEFAULT_SEED_LABELS:
            summary[f"raw_{seed_label}"] = raw_wide[seed_label].to_numpy(dtype=float)
        for seed_label in DEFAULT_SEED_LABELS:
            summary[f"smooth_{seed_label}"] = smooth_wide[seed_label].to_numpy(dtype=float)
        rows.append(summary)

    if not rows:
        raise ValueError("No backbone training curve data remains after alignment.")
    combined = pd.concat(rows, ignore_index=True)
    combined["model_order"] = pd.Categorical(combined["model"], categories=list(DEFAULT_MODEL_RUN_DIRS), ordered=True)
    combined = combined.sort_values(["model_order", "step"], kind="stable").drop(columns=["model_order"])
    return combined.reset_index(drop=True)


def prepare_data(*, metric: str, align: str, smooth_window: int) -> pd.DataFrame:
    raw_frame = load_backbone_metrics(metric=metric)
    smoothed_frame = smooth_seed_metrics(raw_frame, smooth_window=smooth_window)
    return aggregate_backbone_curves(smoothed_frame, align=align)


def plot_backbone_curves(plot_frame: pd.DataFrame, *, dpi: int, out_dir: Path | str) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_4_3)
    for model, model_frame in plot_frame.groupby("model", sort=False):
        style = MODEL_STYLES.get(
            str(model),
            {"color": "#4C78A8", "linewidth": 2.0, "alpha": 0.16, "line_zorder": 10, "shade_zorder": 1},
        )
        ordered = model_frame.sort_values("step")
        x = ordered["step"].to_numpy(dtype=float)
        mean = ordered["mean"].to_numpy(dtype=float)
        std = ordered["std"].to_numpy(dtype=float)
        label = str(ordered["model_display"].iloc[0])
        ax.plot(
            x,
            mean,
            label=label,
            color=str(style["color"]),
            linewidth=float(style["linewidth"]),
            zorder=int(style["line_zorder"]),
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=str(style["color"]),
            alpha=float(style["alpha"]),
            linewidth=0,
            zorder=int(style["shade_zorder"]),
        )

    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(Y_AXIS_LABEL)
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
        description="Plot four-backbone 3-seed training curves from metrics.jsonl.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric key to extract from stats in metrics.jsonl.",
    )
    parser.add_argument(
        "--align",
        choices=("common", "available"),
        default="common",
        help="common keeps only steps present for all seeds per model; available uses any seed present.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Rolling window over available points per model/seed. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "plot_scripts/generated/training/backbone_seed789"),
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
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    plot_frame = prepare_data(
        metric=str(args.metric),
        align=str(args.align),
        smooth_window=int(args.smooth_window),
    )
    written = plot_backbone_curves(plot_frame, dpi=int(args.dpi), out_dir=out_dir)
    data_path = write_plot_data(plot_frame, out_dir=out_dir)
    print_written_paths([*written, data_path])
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
