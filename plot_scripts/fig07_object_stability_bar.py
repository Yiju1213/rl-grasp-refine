from __future__ import annotations

import numpy as np
import pandas as pd

from plot_common import (
    STABILITY_METRIC_SPECS,
    add_zero_reference,
    average_adjusted_object_metric_across_seeds,
    build_base_parser,
    color_for,
    compute_adjusted_per_object_values,
    load_per_object_table_with_baseline,
    maybe_print_plot_data,
    normalize_cli_args,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    set_label_ticks,
    summarize_adjusted_object_stability,
    plt,
)

FIGURE_STEM = "fig07_object_stability_bar"


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str], metric: str) -> pd.DataFrame:
    adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key="success_gain")
    object_frame = average_adjusted_object_metric_across_seeds(adjusted_frame, labels=labels)
    return summarize_adjusted_object_stability(object_frame, labels, metric=metric)


def build_parser():
    return build_base_parser(
        "Plot IQR/std of no-action-adjusted object success gain.",
        default_group="ablation",
        metric_choices=("iqr", "std"),
        default_metric="iqr",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    plot_frame = prepare_data(per_object_frame, labels, args.metric)
    spec = STABILITY_METRIC_SPECS[args.metric]

    positions = np.arange(len(plot_frame), dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(
        positions,
        plot_frame[spec["column"]].to_numpy(dtype=float),
        color=[color_for(str(label)) for label in plot_frame["label"]],
        width=0.72,
    )
    set_default_axis_style(ax)
    add_zero_reference(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_ylabel(spec["ylabel"])

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
