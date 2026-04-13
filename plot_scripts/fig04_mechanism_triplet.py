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
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    set_label_ticks,
    summarize_adjusted_experiment,
    plt,
)

FIGURE_STEM = "fig04_mechanism_triplet"
PANELS = (
    ("excess_t_cover_delta", "Excess T-Cover Delta"),
    ("excess_t_edge_delta", "Excess T-Edge Delta"),
    ("excess_probability_delta", "Excess Probability Delta"),
)


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str]) -> dict[str, pd.DataFrame]:
    plot_frames: dict[str, pd.DataFrame] = {}
    for metric_key, _ in PANELS:
        adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key=metric_key)
        plot_frames[metric_key] = summarize_adjusted_experiment(adjusted_frame, labels)
    return plot_frames


def build_parser():
    return build_base_parser(
        "Plot no-action-adjusted mechanism metrics with object-bootstrap 95% CI.",
        default_group="ablation",
        style_choices=("point", "bar"),
        default_style="point",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    plot_frames = prepare_data(per_object_frame, labels)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)
    for ax, (metric_key, title) in zip(axes, PANELS):
        plot_frame = plot_frames[metric_key]
        spec = ADJUSTED_METRIC_SPECS[metric_key]
        if args.style == "bar":
            draw_bar_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
        else:
            draw_point_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
        set_default_axis_style(ax)
        add_zero_reference(ax)
        set_label_ticks(ax, plot_frame)
        ax.set_title(title)
        ax.set_ylabel(spec["ylabel"])

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, plot_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
