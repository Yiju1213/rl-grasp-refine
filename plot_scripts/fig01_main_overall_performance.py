from __future__ import annotations

import pandas as pd

from plot_common import (
    ADJUSTED_METRIC_SPECS,
    add_zero_reference,
    build_base_parser,
    compute_adjusted_per_object_values,
    draw_bar_with_ci,
    draw_point_with_ci,
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

FIGURE_STEM = "fig01_main_overall_performance"


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key="success_gain")
    return summarize_adjusted_experiment(adjusted_frame, labels)


def build_parser():
    return build_base_parser(
        "Plot no-action-adjusted success gain with object-bootstrap 95% CI.",
        default_group="main",
        style_choices=("point", "bar"),
        default_style="point",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    plot_frame = prepare_data(per_object_frame, labels)
    spec = ADJUSTED_METRIC_SPECS["success_gain"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_4_3)
    if args.style == "bar":
        draw_bar_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
    else:
        draw_point_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
    set_default_axis_style(ax)
    add_zero_reference(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_ylabel(percent_label(spec["ylabel"]))

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
