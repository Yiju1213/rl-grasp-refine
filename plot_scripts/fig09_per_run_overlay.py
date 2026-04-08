from __future__ import annotations

import pandas as pd

from plot_common import (
    SUMMARY_METRIC_SPECS,
    add_zero_reference,
    build_base_parser,
    ci_yerr,
    color_for,
    jitter_positions,
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

FIGURE_STEM = "fig09_per_run_overlay"


def prepare_data(
    summary_frame: pd.DataFrame,
    per_run_frame: pd.DataFrame,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_spec = SUMMARY_METRIC_SPECS["macro_success_lift"]
    validate_columns(
        summary_frame,
        ("label", "display_name", summary_spec["mean"], summary_spec["ci_low"], summary_spec["ci_high"]),
        context="summary.csv",
    )
    validate_columns(
        per_run_frame,
        ("label", "display_name", "test_seed", "macro_success_lift"),
        context="per_run_summary.csv",
    )
    filtered_summary = summary_frame.loc[summary_frame["label"].isin(labels)].copy().reset_index(drop=True)
    filtered_runs = per_run_frame.loc[per_run_frame["label"].isin(labels)].copy().reset_index(drop=True)
    return filtered_summary, filtered_runs


def build_parser():
    return build_base_parser(
        "Overlay 3 run-level repeated-test dots on experiment-level macro success lift with CI.",
        default_group="all_formal",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    summary_frame = load_table_for_labels(args.root, "summary.csv", labels)
    per_run_frame = load_table_for_labels(args.root, "per_run_summary.csv", labels)
    plot_frame, run_frame = prepare_data(summary_frame, per_run_frame, labels)
    spec = SUMMARY_METRIC_SPECS["macro_success_lift"]

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
            label_runs["macro_success_lift"].to_numpy(dtype=float),
            color=color_for(str(row["label"])),
            alpha=0.75,
            s=34,
        )
    set_default_axis_style(ax)
    add_zero_reference(ax)
    set_label_ticks(ax, plot_frame)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title("Experiment Mean With Per-Run Dots")

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
