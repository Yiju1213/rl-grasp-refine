from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPTS_DIR = CURRENT_DIR.parent
if str(PLOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_SCRIPTS_DIR))

from plot_common import add_zero_reference, maybe_print_plot_data, print_written_paths, save_figure, set_default_axis_style, plt

FIGURE_STEM = "cn_tmp_fig01_main_overall_performance_from_table"
CM_TO_INCH = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.0
FIGURE_HEIGHT_CM = FIGURE_WIDTH_CM * 9.0 / 16.0

SIMSUN_PATH = CURRENT_DIR / "simsun.ttc"
TIMES_PATH = CURRENT_DIR / "Times_New_Roman.ttf"

CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
TICK_CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=8.5)
TICK_NUM_FONT = fm.FontProperties(fname=str(TIMES_PATH), size=8.5)

PLOT_FRAME = pd.DataFrame(
    [
        {
            "label": "no_action",
            "display_name": "无动作基线",
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci_minus": 0.0,
            "ci_plus": 0.0,
            "color": "#7F7F7F",
            "marker": "o",
        },
        {
            "label": "random",
            "display_name": "随机动作",
            "mean": -11.2,
            "ci_low": -14.4,
            "ci_high": -7.6,
            "ci_minus": 3.2,
            "ci_plus": 3.6,
            "color": "#D08C60",
            "marker": "o",
        },
        {
            "label": "vanilla",
            "display_name": "基础策略",
            "mean": 10.1,
            "ci_low": 6.6,
            "ci_high": 13.7,
            "ci_minus": 3.5,
            "ci_plus": 3.6,
            "color": "#4C78A8",
            "marker": "o",
        },
        {
            "label": "full",
            "display_name": "完整策略",
            "mean": 11.5,
            "ci_low": 7.6,
            "ci_high": 15.6,
            "ci_minus": 3.9,
            "ci_plus": 4.1,
            "color": "#1F4E79",
            "marker": "o",
        },
    ]
)


def _apply_tick_fonts(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontproperties(TICK_CN_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(TICK_NUM_FONT)


def plot_main_overall_performance(*, dpi: int, out_dir: Path | str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_CM * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH))
    positions = range(len(PLOT_FRAME))

    for idx, row in PLOT_FRAME.iterrows():
        ax.errorbar(
            idx,
            float(row["mean"]),
            yerr=[[float(row["ci_minus"])], [float(row["ci_plus"])]],
            fmt=str(row["marker"]),
            color=str(row["color"]),
            markersize=4.8,
            elinewidth=0.9,
            capsize=2.5,
            linewidth=1.0,
            zorder=3,
        )

    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(PLOT_FRAME["display_name"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("净收益（%）", fontproperties=CN_FONT)
    _apply_tick_fonts(ax)
    plt.tight_layout(pad=0.15)

    written = save_figure(fig, out_dir=out_dir, stem=FIGURE_STEM, formats=("png",), dpi=dpi)
    plt.close(fig)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文绘制主结果净收益图，数据直接写死为表格数值。")
    parser.add_argument(
        "--out-dir",
        default=str(CURRENT_DIR / "generated/main"),
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
    out_dir = Path(args.out_dir).expanduser().resolve()

    written = plot_main_overall_performance(dpi=args.dpi, out_dir=out_dir)
    print_written_paths(written)
    maybe_print_plot_data(args, PLOT_FRAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
