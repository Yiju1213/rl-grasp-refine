from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

CURRENT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPTS_DIR = CURRENT_DIR.parent
if str(PLOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_SCRIPTS_DIR))

from plot_common import add_zero_reference, maybe_print_plot_data, print_written_paths, save_figure, set_default_axis_style, plt

FIGURE_STEM = "cn_tmp_fig08_per_object_rank_curve_from_table"
CM_TO_INCH = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.0
FIGURE_HEIGHT_CM = FIGURE_WIDTH_CM * 9.0 / 16.0

SIMSUN_PATH = CURRENT_DIR / "simsun.ttc"
TIMES_PATH = CURRENT_DIR / "Times_New_Roman.ttf"

CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
LEGEND_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=9.5)
TICK_CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=8.5)
TICK_NUM_FONT = fm.FontProperties(fname=str(TIMES_PATH), size=8.5)

SERIES_STYLES = {
    "no-action": {"display_name_cn": "无动作基线", "color": "#7F7F7F", "marker": "o"},
    "rand-action": {"display_name_cn": "随机动作", "color": "#D08C60", "marker": "s"},
    "drop-only-latent-only-128-epi": {"display_name_cn": "基础策略", "color": "#5DA5DA", "marker": "^"},
    "full-latefus-128-epi": {"display_name_cn": "完整策略", "color": "#2F6DA3", "marker": "D"},
}

PLOT_FRAME = pd.DataFrame(
    [
        {"label": "no-action", "display_name": "No Action", "object_id": 75, "seed_avg_value": 0.000000, "rank": 1},
        {"label": "no-action", "display_name": "No Action", "object_id": 76, "seed_avg_value": 0.000000, "rank": 2},
        {"label": "no-action", "display_name": "No Action", "object_id": 77, "seed_avg_value": 0.000000, "rank": 3},
        {"label": "no-action", "display_name": "No Action", "object_id": 78, "seed_avg_value": 0.000000, "rank": 4},
        {"label": "no-action", "display_name": "No Action", "object_id": 79, "seed_avg_value": 0.000000, "rank": 5},
        {"label": "no-action", "display_name": "No Action", "object_id": 80, "seed_avg_value": 0.000000, "rank": 6},
        {"label": "no-action", "display_name": "No Action", "object_id": 81, "seed_avg_value": 0.000000, "rank": 7},
        {"label": "no-action", "display_name": "No Action", "object_id": 82, "seed_avg_value": 0.000000, "rank": 8},
        {"label": "no-action", "display_name": "No Action", "object_id": 83, "seed_avg_value": 0.000000, "rank": 9},
        {"label": "no-action", "display_name": "No Action", "object_id": 84, "seed_avg_value": 0.000000, "rank": 10},
        {"label": "no-action", "display_name": "No Action", "object_id": 85, "seed_avg_value": 0.000000, "rank": 11},
        {"label": "no-action", "display_name": "No Action", "object_id": 86, "seed_avg_value": 0.000000, "rank": 12},
        {"label": "no-action", "display_name": "No Action", "object_id": 87, "seed_avg_value": 0.000000, "rank": 13},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 76, "seed_avg_value": 0.036667, "rank": 1},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 87, "seed_avg_value": -0.023333, "rank": 2},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 84, "seed_avg_value": -0.076667, "rank": 3},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 86, "seed_avg_value": -0.103333, "rank": 4},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 75, "seed_avg_value": -0.106667, "rank": 5},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 80, "seed_avg_value": -0.110000, "rank": 6},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 83, "seed_avg_value": -0.110000, "rank": 7},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 81, "seed_avg_value": -0.116667, "rank": 8},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 78, "seed_avg_value": -0.123333, "rank": 9},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 77, "seed_avg_value": -0.166667, "rank": 10},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 79, "seed_avg_value": -0.170000, "rank": 11},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 85, "seed_avg_value": -0.190000, "rank": 12},
        {"label": "rand-action", "display_name": "Rand. Action", "object_id": 82, "seed_avg_value": -0.193333, "rank": 13},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 75, "seed_avg_value": 0.223333, "rank": 1},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 76, "seed_avg_value": 0.200000, "rank": 2},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 84, "seed_avg_value": 0.166667, "rank": 3},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 87, "seed_avg_value": 0.150000, "rank": 4},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 86, "seed_avg_value": 0.140000, "rank": 5},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 83, "seed_avg_value": 0.083333, "rank": 6},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 81, "seed_avg_value": 0.080000, "rank": 7},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 77, "seed_avg_value": 0.073333, "rank": 8},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 78, "seed_avg_value": 0.056667, "rank": 9},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 79, "seed_avg_value": 0.056667, "rank": 10},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 82, "seed_avg_value": 0.056667, "rank": 11},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 80, "seed_avg_value": 0.036667, "rank": 12},
        {"label": "drop-only-latent-only-128-epi", "display_name": "Vanilla", "object_id": 85, "seed_avg_value": -0.006667, "rank": 13},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 76, "seed_avg_value": 0.236667, "rank": 1},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 75, "seed_avg_value": 0.223333, "rank": 2},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 84, "seed_avg_value": 0.200000, "rank": 3},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 86, "seed_avg_value": 0.193333, "rank": 4},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 87, "seed_avg_value": 0.176667, "rank": 5},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 81, "seed_avg_value": 0.086667, "rank": 6},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 80, "seed_avg_value": 0.070000, "rank": 7},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 77, "seed_avg_value": 0.070000, "rank": 8},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 78, "seed_avg_value": 0.066667, "rank": 9},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 82, "seed_avg_value": 0.053333, "rank": 10},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 79, "seed_avg_value": 0.043333, "rank": 11},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 83, "seed_avg_value": 0.043333, "rank": 12},
        {"label": "full-latefus-128-epi", "display_name": "Full", "object_id": 85, "seed_avg_value": 0.030000, "rank": 13},
    ]
)


def _apply_tick_fonts(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontproperties(TICK_NUM_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(TICK_NUM_FONT)


def plot_rank_curve(*, dpi: int, out_dir: Path | str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_CM * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH))
    label_order = ["no-action", "rand-action", "drop-only-latent-only-128-epi", "full-latefus-128-epi"]
    for label in label_order:
        label_frame = PLOT_FRAME.loc[PLOT_FRAME["label"] == label].sort_values("rank").copy()
        style = SERIES_STYLES[label]
        ax.plot(
            label_frame["rank"].to_numpy(dtype=float),
            label_frame["seed_avg_value"].to_numpy(dtype=float) * 100.0,
            marker=style["marker"],
            markersize=4.0,
            linewidth=1.4,
            color=style["color"],
            label=style["display_name_cn"],
        )

    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xlabel("物体排序", fontproperties=CN_FONT)
    ax.set_ylabel("无动作基线下的净收益（%）", fontproperties=CN_FONT)
    ax.set_xticks(np.arange(1, 14, dtype=int))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    legend = ax.legend(frameon=False, loc="best", prop=LEGEND_FONT)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(LEGEND_FONT)
    _apply_tick_fonts(ax)
    plt.tight_layout(pad=0.15)

    written = save_figure(fig, out_dir=out_dir, stem=FIGURE_STEM, formats=("png",), dpi=dpi)
    plt.close(fig)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文绘制物体排序曲线，数据直接提取自 plot_data_main.txt 的 fig08 段。")
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

    written = plot_rank_curve(dpi=args.dpi, out_dir=out_dir)
    print_written_paths(written)
    maybe_print_plot_data(args, PLOT_FRAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
