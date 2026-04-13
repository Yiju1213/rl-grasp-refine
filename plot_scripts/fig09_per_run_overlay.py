from __future__ import annotations

import pandas as pd

from plot_common import (
    ADJUSTED_METRIC_SPECS,
    add_zero_reference,
    build_base_parser,
    ci_yerr,
    color_for,
    compute_adjusted_per_object_values,
    jitter_positions,
    load_per_object_table_with_baseline,
    maybe_print_plot_data,
    normalize_cli_args,
    print_written_paths,
    resolve_selected_labels,
    save_figure,
    set_default_axis_style,
    set_label_ticks,
    summarize_adjusted_experiment,
    summarize_adjusted_runs,
    plt,
)

FIGURE_STEM = "fig09_per_run_overlay"


def prepare_data(
    per_object_frame: pd.DataFrame,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key="success_gain")
    return summarize_adjusted_experiment(adjusted_frame, labels), summarize_adjusted_runs(adjusted_frame, labels)


def build_parser():
    return build_base_parser(
        "Overlay adjusted test-seed run means on no-action-adjusted experiment mean with CI.",
        default_group="main",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    plot_frame, run_frame = prepare_data(per_object_frame, labels)
    spec = ADJUSTED_METRIC_SPECS["success_gain"]

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    for index, row in plot_frame.reset_index(drop=True).iterrows():
        ax.errorbar(
            index,
            float(row[spec["mean"]]),
            yerr=ci_yerr(
                plot_frame.iloc[[index]],
                mean_col=spec["mean"],
                low_col=spec["ci_low"],
                high_col=spec["ci_high"],
            ),
            fmt="o",
            color=color_for(str(row["label"])),
            markersize=7,
            elinewidth=1.4,
            capsize=4,
        )
        label_runs = run_frame.loc[run_frame["label"] == row["label"]].copy().sort_values("test_seed")
        ax.scatter(
            jitter_positions(float(index), len(label_runs)),
            label_runs["adjusted_run_mean"].to_numpy(dtype=float),
            color=color_for(str(row["label"])),
            alpha=0.75,
            s=34,
        )
    set_default_axis_style(ax)
    add_zero_reference(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_ylabel(spec["ylabel"])

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, (("experiment_summary", plot_frame), ("run_points", run_frame)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
