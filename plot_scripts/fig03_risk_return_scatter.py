from __future__ import annotations

import pandas as pd

from plot_common import (
    ADJUSTED_METRIC_SPECS,
    FONT_SIZES,
    build_base_parser,
    color_for,
    compute_adjusted_per_object_values,
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

FIGURE_STEM = "fig03_risk_return_scatter"


def _summarize_adjusted_metric(per_object_frame: pd.DataFrame, labels: list[str], metric_key: str, prefix: str) -> pd.DataFrame:
    adjusted_frame = compute_adjusted_per_object_values(per_object_frame, labels, metric_key=metric_key)
    summary = summarize_adjusted_experiment(adjusted_frame, labels)
    return summary.rename(columns={"adjusted_mean": f"{prefix}_mean"})


def prepare_data(per_object_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    degradation = _summarize_adjusted_metric(per_object_frame, labels, "excess_degradation", "degradation")
    recovery = _summarize_adjusted_metric(per_object_frame, labels, "excess_recovery", "recovery")
    return degradation.merge(recovery[["label", "recovery_mean"]], on="label", how="inner").reset_index(drop=True)


def build_parser():
    return build_base_parser(
        "Plot no-action-adjusted degradation vs recovery scatter.",
        default_group="main",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_cli_args(parser.parse_args(argv))
    labels = resolve_selected_labels(args.root, group=args.group, labels=args.labels)
    per_object_frame = load_per_object_table_with_baseline(args.root, labels)
    plot_frame = prepare_data(per_object_frame, labels)
    degradation_spec = ADJUSTED_METRIC_SPECS["excess_degradation"]
    recovery_spec = ADJUSTED_METRIC_SPECS["excess_recovery"]

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for _, row in plot_frame.iterrows():
        ax.scatter(
            float(row["degradation_mean"]),
            float(row["recovery_mean"]),
            color=color_for(str(row["label"])),
            marker=marker_for(str(row["label"])),
            s=70,
        )
        ax.annotate(
            str(row["display_name"]),
            (float(row["degradation_mean"]), float(row["recovery_mean"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=FONT_SIZES["annotation"],
        )
    set_default_axis_style(ax)
    ax.axhline(0.0, color="#6E6E6E", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.axvline(0.0, color="#6E6E6E", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel(degradation_spec["ylabel"])
    ax.set_ylabel(recovery_spec["ylabel"])

    written = save_figure(fig, out_dir=args.out_dir, stem=FIGURE_STEM, formats=args.formats, dpi=args.dpi)
    plt.close(fig)
    print_written_paths(written)
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
