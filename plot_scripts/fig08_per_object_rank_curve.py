from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from plot_common import (
    add_zero_reference,
    average_adjusted_object_metric_across_seeds,
    build_base_parser,
    color_for,
    compute_adjusted_per_object_values,
    load_per_object_table_with_baseline,
    marker_for,
    maybe_print_plot_data,
    normalize_cli_args,
    percent_label,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    FIGURE_SIZE_4_3,
    plt,
)

FIGURE_STEM = "fig08_per_object_rank_curve"


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key="success_gain")
    averaged_frame = average_adjusted_object_metric_across_seeds(adjusted_frame, labels=labels)
    ranked_frames: list[pd.DataFrame] = []
    for label in labels:
        label_frame = averaged_frame.loc[averaged_frame["label"] == label].copy()
        label_frame = label_frame.sort_values("seed_avg_value", ascending=False).reset_index(drop=True)
        label_frame["rank"] = np.arange(1, len(label_frame) + 1, dtype=int)
        ranked_frames.append(label_frame)
    return pd.concat(ranked_frames, ignore_index=True)


def build_parser():
    return build_base_parser(
        "Plot no-action-adjusted per-object success-gain rank curves after averaging each object across seeds.",
        default_group="ablation",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    ranked_frame = prepare_data(per_object_frame, labels)
    present_labels = [label for label in labels if label in set(ranked_frame["label"])]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_4_3)
    for label in present_labels:
        label_frame = ranked_frame.loc[ranked_frame["label"] == label].copy()
        ax.plot(
            label_frame["rank"].to_numpy(dtype=float),
            label_frame["seed_avg_value"].to_numpy(dtype=float),
            marker=marker_for(label),
            markersize=4.5,
            linewidth=1.6,
            color=color_for(label),
            label=str(label_frame["display_name"].iloc[0]),
        )
    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xlabel("Object Rank")
    max_rank = int(ranked_frame["rank"].max())
    ax.set_xticks(np.arange(1, max_rank + 1, dtype=int))
    ax.set_ylabel(percent_label("Object Success Gain over No-Action"))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.legend(frameon=False)

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, ranked_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
