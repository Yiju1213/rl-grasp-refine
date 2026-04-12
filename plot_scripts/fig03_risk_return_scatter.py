from __future__ import annotations

import pandas as pd

from plot_common import (
    SUMMARY_METRIC_SPECS,
    build_base_parser,
    color_for,
    load_table_for_labels,
    marker_for,
    normalize_cli_args,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    validate_columns,
    plt,
)

FIGURE_STEM = "fig03_risk_return_scatter"


def prepare_data(summary_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    pos_spec = SUMMARY_METRIC_SPECS["pos_drop"]
    neg_spec = SUMMARY_METRIC_SPECS["neg_hold"]
    validate_columns(
        summary_frame,
        ("label", "display_name", pos_spec["mean"], neg_spec["mean"]),
        context="summary.csv",
    )
    filtered = summary_frame.loc[summary_frame["label"].isin(labels)].copy()
    return filtered.reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot experiment-level positive-drop vs negative-hold trade-off scatter.",
        default_group="main",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    summary_frame = load_table_for_labels(args.root, "summary.csv", labels)
    plot_frame = prepare_data(summary_frame, labels)
    pos_col = SUMMARY_METRIC_SPECS["pos_drop"]["mean"]
    neg_col = SUMMARY_METRIC_SPECS["neg_hold"]["mean"]

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for _, row in plot_frame.iterrows():
        ax.scatter(
            float(row[pos_col]),
            float(row[neg_col]),
            color=color_for(str(row["label"])),
            marker=marker_for(str(row["label"])),
            s=70,
        )
        ax.annotate(
            str(row["display_name"]),
            (float(row[pos_col]), float(row[neg_col])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    set_default_axis_style(ax)
    ax.set_xlabel("Positive-Sample Drop Rate")
    ax.set_ylabel("Negative-Sample Hold Rate")
    ax.set_title("Risk-Return Frontier")

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
