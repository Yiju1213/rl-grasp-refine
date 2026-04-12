from __future__ import annotations

import pandas as pd

from plot_common import (
    add_zero_reference,
    average_adjusted_object_metric_across_seeds,
    build_base_parser,
    color_for,
    compute_adjusted_per_object_values,
    load_per_object_table_with_baseline,
    normalize_cli_args,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    plt,
)

FIGURE_STEM = "fig06_object_stability_boxplot"


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key="success_gain")
    return average_adjusted_object_metric_across_seeds(adjusted_frame, labels=labels)


def build_parser():
    return build_base_parser(
        "Plot no-action-adjusted object success-gain boxplots after averaging each object across seeds.",
        default_group="ablation",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    averaged_frame = prepare_data(per_object_frame, labels)
    present_labels = [label for label in labels if label in set(averaged_frame["label"])]

    grouped_data = [
        averaged_frame.loc[averaged_frame["label"] == label, "seed_avg_value"].to_numpy(dtype=float)
        for label in present_labels
    ]
    display_names = [
        averaged_frame.loc[averaged_frame["label"] == label, "display_name"].iloc[0]
        for label in present_labels
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    boxplot = ax.boxplot(grouped_data, patch_artist=True, widths=0.6)
    for patch, label in zip(boxplot["boxes"], present_labels):
        patch.set_facecolor(color_for(label))
        patch.set_alpha(0.65)
    for median in boxplot["medians"]:
        median.set_color("#202020")
        median.set_linewidth(1.5)
    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xticks(range(1, len(display_names) + 1))
    ax.set_xticklabels(display_names, rotation=25, ha="right")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Object Success Gain over No-Action")
    ax.set_title("No-Action-Adjusted Across-Object Stability")

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
