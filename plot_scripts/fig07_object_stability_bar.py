from __future__ import annotations

import numpy as np
import pandas as pd

from plot_common import (
    STABILITY_METRIC_SPECS,
    add_zero_reference,
    build_base_parser,
    color_for,
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

FIGURE_STEM = "fig07_object_stability_bar"


def prepare_data(summary_frame: pd.DataFrame, labels: list[str], metric: str) -> pd.DataFrame:
    spec = STABILITY_METRIC_SPECS[metric]
    validate_columns(summary_frame, ("label", "display_name", spec["column"]), context="summary.csv")
    filtered = summary_frame.loc[summary_frame["label"].isin(labels)].copy()
    return filtered.reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot a compact across-object stability summary bar chart.",
        default_group="all_formal",
        metric_choices=("iqr", "std"),
        default_metric="iqr",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    summary_frame = load_table_for_labels(args.root, "summary.csv", labels)
    plot_frame = prepare_data(summary_frame, labels, args.metric)
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
    ax.set_xlabel("Experiment")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title("Across-Object Stability Summary")

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
