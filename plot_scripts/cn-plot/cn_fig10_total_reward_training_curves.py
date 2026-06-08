from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator

CURRENT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPTS_DIR = CURRENT_DIR.parent
if str(PLOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_SCRIPTS_DIR))

from fig10_full_training_curves import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_RUN_DIRS,
    METRIC_STYLES,
    normalize_multi_value,
    prepare_data,
)
from plot_common import (  # noqa: E402
    add_zero_reference,
    maybe_print_plot_data,
    print_written_paths,
    save_figure,
    set_default_axis_style,
    plt,
)

FIGURE_STEM = "cn_fig10_total_reward_training_curves"
CM_TO_INCH = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.0
FIGURE_HEIGHT_CM = FIGURE_WIDTH_CM * 9.0 / 16.0

SIMSUN_PATH = CURRENT_DIR / "simsun.ttc"
TIMES_PATH = CURRENT_DIR / "Times_New_Roman.ttf"

CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
LEGEND_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=9.5)
TICK_FONT = fm.FontProperties(fname=str(TIMES_PATH), size=8.5)

DEFAULT_METRICS = (
    "validation/reward/total_mean",
    "reward/total_mean",
)
METRIC_DISPLAY_NAMES_CN = {
    "validation/reward/total_mean": "验证集",
    "reward/total_mean": "训练集",
}
METRIC_STYLES_CN = {
    "validation/reward/total_mean": {
        **METRIC_STYLES["outcome/success_lift_vs_dataset"],
        "linestyle": "-",
    },
    "reward/total_mean": {
        **METRIC_STYLES["validation/outcome/success_lift_vs_dataset"],
        "linestyle": "--",
        "dashes": (2.5, 1.0),
    },
}


def _apply_tick_fonts(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontproperties(TICK_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(TICK_FONT)


def plot_training_curves_cn(plot_frame, *, dpi: int, out_dir: Path | str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_CM * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH))
    for metric, metric_frame in plot_frame.groupby("metric", sort=False):
        style = METRIC_STYLES_CN.get(
            str(metric),
            {"color": "#4C78A8", "linestyle": "-", "linewidth": 2.0, "alpha": 0.16, "zorder": 2},
        )
        ordered = metric_frame.sort_values("step")
        x = ordered["step"].to_numpy(dtype=float)
        mean = ordered["mean"].to_numpy(dtype=float)
        std = ordered["std"].to_numpy(dtype=float)
        label = METRIC_DISPLAY_NAMES_CN.get(str(metric), str(metric))
        (line,) = ax.plot(
            x,
            mean,
            label=label,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=float(style["linewidth"]),
            zorder=int(style["zorder"]),
        )
        dashes = style.get("dashes")
        if dashes is not None:
            line.set_dashes(dashes)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=str(style["color"]),
            alpha=float(style["alpha"]),
            linewidth=0,
            zorder=int(style["zorder"]) - 1,
        )

    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xlabel("训练轮次", fontproperties=CN_FONT)
    ax.set_ylabel("总奖励", fontproperties=CN_FONT)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    legend = ax.legend(frameon=False, loc="best", prop=LEGEND_FONT)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(LEGEND_FONT)
    _apply_tick_fonts(ax)
    plt.tight_layout(pad=0.15)
    written = save_figure(fig, out_dir=out_dir, stem=FIGURE_STEM, formats=("png",), dpi=dpi)
    plt.close(fig)
    return written


def write_plot_data(plot_frame, *, out_dir: Path | str) -> Path:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{FIGURE_STEM}_data.csv"
    plot_frame.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文绘制 Full policy 三个 seed 的 total reward 训练曲线。")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[str(path) for path in DEFAULT_RUN_DIRS],
        help="训练目录列表，每个目录需要包含 metrics.jsonl。",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="seed 标签，用于导出原始列名。",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="从 metrics.jsonl 的 stats 中读取的指标名。",
    )
    parser.add_argument(
        "--align",
        choices=("common", "available"),
        default="common",
        help="common 只保留所有 seed 共有 step；available 保留任一 seed 有值的 step。",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="每个 seed/metric 的 rolling 平滑窗口；1 表示不平滑。",
    )
    parser.add_argument(
        "--out-dir",
        default=str(CURRENT_DIR / "generated/training"),
        help="输出目录。",
    )
    parser.add_argument("--dpi", type=int, default=330, help="PNG 导出 DPI。")
    print_group = parser.add_mutually_exclusive_group()
    print_group.add_argument(
        "--print-data",
        dest="print_data",
        action="store_true",
        default=True,
        help="打印最终作图数据，默认开启。",
    )
    print_group.add_argument(
        "--no-print-data",
        dest="print_data",
        action="store_false",
        help="不打印最终作图数据。",
    )
    parser.add_argument(
        "--print-data-format",
        choices=("table", "csv"),
        default="table",
        help="作图数据打印格式。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    labels = normalize_multi_value(args.labels)
    metrics = normalize_multi_value(args.metrics)
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    out_dir = Path(args.out_dir).expanduser().resolve()

    plot_frame = prepare_data(
        run_dirs,
        labels,
        metrics,
        align=args.align,
        smooth_window=args.smooth_window,
    )
    written = plot_training_curves_cn(plot_frame, dpi=args.dpi, out_dir=out_dir)
    data_path = write_plot_data(plot_frame, out_dir=out_dir)
    print_written_paths([*written, data_path])
    maybe_print_plot_data(args, plot_frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
