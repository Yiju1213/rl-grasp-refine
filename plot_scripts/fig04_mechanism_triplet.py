from __future__ import annotations

import pandas as pd

from plot_common import (
    SUMMARY_METRIC_SPECS,
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

FIGURE_STEM = "fig04_mechanism_triplet"
PANELS = (
    ("t_cover_delta", "T-Cover Delta"),
    ("t_edge_delta", "T-Edge Delta"),
    ("prob_delta_mean", "Probability Delta Mean"),
)


def prepare_data(summary_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    required = ["label", "display_name"]
    for metric_key, _ in PANELS:
        spec = SUMMARY_METRIC_SPECS[metric_key]
        required.extend([spec["mean"], spec["ci_low"], spec["ci_high"]])
    validate_columns(summary_frame, required, context="summary.csv")
    filtered = summary_frame.loc[summary_frame["label"].isin(labels)].copy()
    return filtered.reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot the three mechanism metrics with 95% CI.",
        default_group="ablation",
        style_choices=("point", "bar"),
        default_style="point",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    summary_frame = load_table_for_labels(args.root, "summary.csv", labels)
    plot_frame = prepare_data(summary_frame, labels)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)
    for ax, (metric_key, title) in zip(axes, PANELS):
        spec = SUMMARY_METRIC_SPECS[metric_key]
        if args.style == "bar":
            draw_bar_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
        else:
            draw_point_with_ci(ax, plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"])
        set_default_axis_style(ax)
        set_label_ticks(ax, plot_frame)
        ax.set_title(title)
        ax.set_xlabel("Experiment")
        ax.set_ylabel(spec["ylabel"])

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
