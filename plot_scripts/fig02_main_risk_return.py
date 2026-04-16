from __future__ import annotations

import numpy as np
import pandas as pd

from plot_common import (
    add_zero_reference,
    build_base_parser,
    compute_adjusted_per_object_values,
    load_per_object_table_with_baseline,
    maybe_print_plot_data,
    normalize_cli_args,
    percent_label,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    set_label_ticks,
    summarize_adjusted_experiment,
    FIGURE_SIZE_4_3,
    plt,
)
from plot_config import BASELINE_LABEL, BENEFIT_COLOR, RISK_COLOR

FIGURE_STEM = "fig02_main_risk_return"


def plot_labels_without_baseline(labels: list[str]) -> list[str]:
    filtered = [label for label in labels if label != BASELINE_LABEL]
    if not filtered:
        raise ValueError(f"{FIGURE_STEM} requires at least one non-baseline label to plot.")
    return filtered


def _summarize_adjusted_metric(per_object_frame: pd.DataFrame, labels: list[str], metric_key: str, prefix: str) -> pd.DataFrame:
    adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key=metric_key)
    summary = summarize_adjusted_experiment(adjusted_frame, labels)
    return summary.rename(
        columns={
            "adjusted_mean": f"{prefix}_mean",
            "adjusted_ci95_low": f"{prefix}_ci95_low",
            "adjusted_ci95_high": f"{prefix}_ci95_high",
        }
    )


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    degradation = _summarize_adjusted_metric(per_object_frame, labels, "excess_degradation", "degradation")
    recovery = _summarize_adjusted_metric(per_object_frame, labels, "excess_recovery", "recovery")
    return degradation.merge(
        recovery[["label", "recovery_mean", "recovery_ci95_low", "recovery_ci95_high"]],
        on="label",
        how="inner",
    ).reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot no-action-adjusted degradation and recovery bars with object-bootstrap 95% CI.",
        default_group="main",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    plot_labels = plot_labels_without_baseline(labels)
    plot_frame = prepare_data(per_object_frame, plot_labels)

    positions = np.arange(len(plot_frame), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_4_3)
    ax.bar(positions - width / 2.0, plot_frame["degradation_mean"], width=width, color=RISK_COLOR, label="Excess Degradation")
    ax.bar(positions + width / 2.0, plot_frame["recovery_mean"], width=width, color=BENEFIT_COLOR, label="Excess Recovery")
    ax.errorbar(
        positions - width / 2.0,
        plot_frame["degradation_mean"],
        yerr=np.vstack(
            [
                plot_frame["degradation_mean"] - plot_frame["degradation_ci95_low"],
                plot_frame["degradation_ci95_high"] - plot_frame["degradation_mean"],
            ]
        ),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.1,
        capsize=4,
    )
    ax.errorbar(
        positions + width / 2.0,
        plot_frame["recovery_mean"],
        yerr=np.vstack(
            [
                plot_frame["recovery_mean"] - plot_frame["recovery_ci95_low"],
                plot_frame["recovery_ci95_high"] - plot_frame["recovery_mean"],
            ]
        ),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.1,
        capsize=4,
    )
    set_default_axis_style(ax)
    add_zero_reference(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_ylabel(percent_label("Rate Difference over No-Action"))
    ax.legend(frameon=False)

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
