from __future__ import annotations

import numpy as np
import pandas as pd

from plot_common import (
    ADJUSTED_METRIC_SPECS,
    build_base_parser,
    compute_adjusted_per_object_values,
    color_for,
    ci_yerr,
    load_per_object_table_with_baseline,
    marker_for,
    maybe_print_plot_data,
    normalize_cli_args,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    summarize_adjusted_experiment,
    plt,
)

FIGURE_STEM = "fig04_mechanism_triplet"
PANELS = (
    ("excess_t_cover_delta", "(a) Excess T-Cover Delta", (0.0, 0.10), 0.02),
    ("excess_t_edge_delta", "(b) Excess T-Edge Delta", (0.0, 0.10), 0.02),
    ("excess_probability_delta", "(c) Excess Probability Delta", (0.0, 0.25), 0.05),
)

FIG04_DISPLAY_NAMES = {
    "drop-only-latent-only-128-epi": "Vanilla",
    "vanilla": "Vanilla",
    "wo-onl-cal_latefus_128-epi": "w/o Onl. Cal.",
    "wo-onl-cal": "w/o Onl. Cal.",
    "wo-stb-rwd_latefus_128-epi": "w/o Stb. Rwd.",
    "wo-stb-rwd": "w/o Stb. Rwd.",
    "wo-tac-rwd_latefus_128-epi": "w/o Tac. Rwd.",
    "wo-tac-rwd": "w/o Tac. Rwd.",
    "wo-tac-sem-n-rwd_latefus_128-epi": "w/o Tac. Sem./Rwd.",
    "wo-tac-sem-rwd": "w/o Tac. Sem./Rwd.",
    "full-latefus-128-epi": "Full",
    "full-sga-gsn": "Full",
}


def apply_fig04_display_names(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed["display_name"] = [
        FIG04_DISPLAY_NAMES.get(str(label), str(display_name))
        for label, display_name in zip(renamed["label"], renamed["display_name"], strict=True)
    ]
    return renamed


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str]) -> dict[str, pd.DataFrame]:
    plot_frames: dict[str, pd.DataFrame] = {}
    for metric_key, _, _, _ in PANELS:
        adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key=metric_key)
        plot_frames[metric_key] = apply_fig04_display_names(summarize_adjusted_experiment(adjusted_frame, labels))
    return plot_frames


def build_parser():
    return build_base_parser(
        "Plot horizontal no-action-adjusted mechanism metrics with object-bootstrap 95% CI.",
        default_group="ablation",
    )


def draw_horizontal_point_ci(
    ax,
    plot_frame: pd.DataFrame,
    *,
    mean_col: str,
    low_col: str,
    high_col: str,
    show_y_labels: bool,
) -> None:
    y_positions = list(reversed(range(len(plot_frame))))
    yerr = ci_yerr(plot_frame, mean_col=mean_col, low_col=low_col, high_col=high_col)
    xerr = yerr[[0, 1], :]
    for index, row in plot_frame.reset_index(drop=True).iterrows():
        ax.errorbar(
            float(row[mean_col]),
            y_positions[index],
            xerr=xerr[:, [index]],
            fmt=marker_for(str(row["label"])),
            color=color_for(str(row["label"])),
            markersize=5.5,
            elinewidth=1.3,
            capsize=3,
            linewidth=1.3,
        )
    ax.set_yticks(y_positions)
    if show_y_labels:
        ax.set_yticklabels(plot_frame["display_name"].tolist(), fontsize=13)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    plot_frames = prepare_data(per_object_frame, labels)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), sharex=False, sharey=True)
    for panel_index, (ax, (metric_key, caption, xlim, tick_step)) in enumerate(zip(axes, PANELS)):
        plot_frame = plot_frames[metric_key]
        spec = ADJUSTED_METRIC_SPECS[metric_key]
        draw_horizontal_point_ci(
            ax,
            plot_frame,
            mean_col=spec["mean"],
            low_col=spec["ci_low"],
            high_col=spec["ci_high"],
            show_y_labels=panel_index == 0,
        )
        set_default_axis_style(ax)
        ax.axvline(0.0, color="#6E6E6E", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_xlim(*xlim)
        ax.set_xticks(np.arange(xlim[0], xlim[1] + tick_step / 2.0, tick_step))
        ax.text(
            0.5,
            -0.23,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
        )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.98))
    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, plot_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
