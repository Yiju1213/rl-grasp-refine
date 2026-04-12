from __future__ import annotations

import pandas as pd

from plot_common import (
    SUMMARY_METRIC_SPECS,
    add_zero_reference,
    build_base_parser,
    draw_bar_with_ci,
    draw_point_with_ci,
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

FIGURE_STEM = "fig01_main_overall_performance"


def prepare_data(summary_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    spec = SUMMARY_METRIC_SPECS["macro_success_lift"]
    validate_columns(
        summary_frame,
        ("label", "display_name", spec["mean"], spec["ci_low"], spec["ci_high"]),
        context="summary.csv",
    )
    filtered = summary_frame.loc[summary_frame["label"].isin(labels)].copy()
    return filtered.reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot experiment-level macro success lift with 95% CI.",
        default_group="main",
        style_choices=("point", "bar"),
        default_style="point",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    summary_frame = load_table_for_labels(args.root, "summary.csv", labels)
    plot_frame = prepare_data(summary_frame, labels)
    spec = SUMMARY_METRIC_SPECS["macro_success_lift"]

    fig, ax = plt.subplots(figsize=(10, 5))
    if args.style == "bar":
        draw_bar_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
    else:
        draw_point_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
    set_default_axis_style(ax)
    add_zero_reference(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_ylabel(spec["ylabel"])
    ax.set_xlabel("Experiment")
    ax.set_title("Overall Performance")

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
