from __future__ import annotations

import numpy as np
import pandas as pd

from plot_common import (
    SUMMARY_METRIC_SPECS,
    add_zero_reference,
    build_base_parser,
    ci_yerr,
    color_for,
    load_table_for_labels,
    marker_for,
    normalize_cli_args,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    set_label_ticks,
    validate_columns,
    plt,
)

FIGURE_STEM = "fig05_reward_scale_response"


def prepare_data(summary_frame: pd.DataFrame, labels: list[str], metric: str) -> pd.DataFrame:
    spec = SUMMARY_METRIC_SPECS[metric]
    validate_columns(
        summary_frame,
        ("label", "display_name", spec["mean"], spec["ci_low"], spec["ci_high"]),
        context="summary.csv",
    )
    filtered = summary_frame.loc[summary_frame["label"].isin(labels)].copy()
    return filtered.reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot reward-scale response on the reward-scan group.",
        default_group="group_b",
        style_choices=("line", "bar"),
        default_style="line",
        metric_choices=("macro_success_lift", "prob_delta_mean", "neg_hold"),
        default_metric="macro_success_lift",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    summary_frame = load_table_for_labels(args.root, "summary.csv", labels)
    plot_frame = prepare_data(summary_frame, labels, args.metric)
    spec = SUMMARY_METRIC_SPECS[args.metric]
    positions = np.arange(len(plot_frame), dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    if args.style == "bar":
        ax.bar(
            positions,
            plot_frame[spec["mean"]].to_numpy(dtype=float),
            color=[color_for(str(label)) for label in plot_frame["label"]],
            width=0.72,
        )
        ax.errorbar(
            positions,
            plot_frame[spec["mean"]].to_numpy(dtype=float),
            yerr=ci_yerr(plot_frame, mean_col=spec["mean"], low_col=spec["ci_low"], high_col=spec["ci_high"]),
            fmt="none",
            ecolor="#303030",
            elinewidth=1.1,
            capsize=4,
        )
    else:
        ax.plot(
            positions,
            plot_frame[spec["mean"]].to_numpy(dtype=float),
            color="#303030",
            linewidth=1.4,
            alpha=0.8,
        )
        for index, row in plot_frame.reset_index(drop=True).iterrows():
            ax.errorbar(
                positions[index],
                float(row[spec["mean"]]),
                yerr=np.asarray(
                    [[float(row[spec["mean"]]) - float(row[spec["ci_low"]])], [float(row[spec["ci_high"]]) - float(row[spec["mean"]])]],
                    dtype=float,
                ),
                fmt=marker_for(str(row["label"])),
                color=color_for(str(row["label"])),
                markersize=7,
                elinewidth=1.3,
                capsize=4,
            )
    set_default_axis_style(ax)
    add_zero_reference(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_xlabel("Reward Scale Setting")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title("Reward Scale Response")

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
