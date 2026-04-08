from __future__ import annotations

import numpy as np
import pandas as pd

from plot_common import (
    SUMMARY_METRIC_SPECS,
    build_base_parser,
    load_table_for_labels,
    normalize_cli_args,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    set_label_ticks,
    validate_columns,
    plt,
)
from plot_config import BENEFIT_COLOR, RISK_COLOR

FIGURE_STEM = "fig02_main_risk_return"


def prepare_data(summary_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    pos_spec = SUMMARY_METRIC_SPECS["pos_drop"]
    neg_spec = SUMMARY_METRIC_SPECS["neg_hold"]
    validate_columns(
        summary_frame,
        (
            "label",
            "display_name",
            pos_spec["mean"],
            pos_spec["ci_low"],
            pos_spec["ci_high"],
            neg_spec["mean"],
            neg_spec["ci_low"],
            neg_spec["ci_high"],
        ),
        context="summary.csv",
    )
    filtered = summary_frame.loc[summary_frame["label"].isin(labels)].copy()
    return filtered.reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot positive-drop and negative-hold trade-off bars with 95% CI.",
        default_group="all_formal",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    summary_frame = load_table_for_labels(args.root, "summary.csv", labels)
    plot_frame = prepare_data(summary_frame, labels)
    pos_spec = SUMMARY_METRIC_SPECS["pos_drop"]
    neg_spec = SUMMARY_METRIC_SPECS["neg_hold"]

    positions = np.arange(len(plot_frame), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(positions - width / 2.0, plot_frame[pos_spec["mean"]], width=width, color=RISK_COLOR, label="Pos-Drop")
    ax.bar(positions + width / 2.0, plot_frame[neg_spec["mean"]], width=width, color=BENEFIT_COLOR, label="Neg-Hold")
    ax.errorbar(
        positions - width / 2.0,
        plot_frame[pos_spec["mean"]],
        yerr=np.vstack(
            [
                plot_frame[pos_spec["mean"]] - plot_frame[pos_spec["ci_low"]],
                plot_frame[pos_spec["ci_high"]] - plot_frame[pos_spec["mean"]],
            ]
        ),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.1,
        capsize=4,
    )
    ax.errorbar(
        positions + width / 2.0,
        plot_frame[neg_spec["mean"]],
        yerr=np.vstack(
            [
                plot_frame[neg_spec["mean"]] - plot_frame[neg_spec["ci_low"]],
                plot_frame[neg_spec["ci_high"]] - plot_frame[neg_spec["mean"]],
            ]
        ),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.1,
        capsize=4,
    )
    set_default_axis_style(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_ylabel("Rate")
    ax.set_xlabel("Experiment")
    ax.set_title("Risk-Return Trade-Off")
    ax.legend(frameon=False)

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
