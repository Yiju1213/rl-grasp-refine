from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPTS_DIR = CURRENT_DIR.parent
if str(PLOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_SCRIPTS_DIR))

from plot_common import add_zero_reference, maybe_print_plot_data, print_written_paths, save_figure, set_default_axis_style, plt

FIGURE_STEM = "cn_tmp_fig02_main_risk_return_from_table"
CM_TO_INCH = 1.0 / 2.54
FIGURE_WIDTH_CM = 12.0
FIGURE_HEIGHT_CM = FIGURE_WIDTH_CM * 9.0 / 16.0

SIMSUN_PATH = CURRENT_DIR / "simsun.ttc"
TIMES_PATH = CURRENT_DIR / "Times_New_Roman.ttf"

CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
LEGEND_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=9.5)
TICK_CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
TICK_NUM_FONT = fm.FontProperties(fname=str(TIMES_PATH), size=8.5)

RISK_COLOR = "#D08C60"
RECOVERY_COLOR = "#4C78A8"

PLOT_FRAME = pd.DataFrame(
    [
        {
            "label": "no_action",
            "display_name": "无动作基线",
            "degradation_mean": 0.0,
            "degradation_ci_minus": 0.0,
            "degradation_ci_plus": 0.0,
            "recovery_mean": 0.0,
            "recovery_ci_minus": 0.0,
            "recovery_ci_plus": 0.0,
        },
        {
            "label": "random",
            "display_name": "随机动作",
            "degradation_mean": 28.1,
            "degradation_ci_minus": 3.1,
            "degradation_ci_plus": 3.5,
            "recovery_mean": 13.4,
            "recovery_ci_minus": 2.7,
            "recovery_ci_plus": 2.5,
        },
        {
            "label": "vanilla",
            "display_name": "基础策略",
            "degradation_mean": 8.5,
            "degradation_ci_minus": 3.2,
            "degradation_ci_plus": 3.9,
            "recovery_mean": 43.0,
            "recovery_ci_minus": 6.4,
            "recovery_ci_plus": 6.4,
        },
        {
            "label": "full",
            "display_name": "完整策略",
            "degradation_mean": 8.5,
            "degradation_ci_minus": 3.5,
            "degradation_ci_plus": 3.9,
            "recovery_mean": 44.9,
            "recovery_ci_minus": 4.9,
            "recovery_ci_plus": 5.1,
        },
    ]
)


def _plot_frame_without_baseline() -> pd.DataFrame:
    filtered = PLOT_FRAME.loc[PLOT_FRAME["label"] != "no_action"].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No non-baseline rows remain to plot.")
    return filtered


def _apply_tick_fonts(ax) -> None:
    for label in ax.get_xticklabels():
        label.set_fontproperties(TICK_CN_FONT)
    for label in ax.get_yticklabels():
        label.set_fontproperties(TICK_NUM_FONT)


def plot_main_risk_return(*, dpi: int, out_dir: Path | str) -> list[Path]:
    plot_frame = _plot_frame_without_baseline()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_CM * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH))
    positions = np.arange(len(plot_frame), dtype=float)
    width = 0.34

    ax.bar(
        positions - width / 2.0,
        plot_frame["degradation_mean"].to_numpy(dtype=float),
        width=width,
        color=RISK_COLOR,
        label="退化风险",
        zorder=2,
    )
    ax.bar(
        positions + width / 2.0,
        plot_frame["recovery_mean"].to_numpy(dtype=float),
        width=width,
        color=RECOVERY_COLOR,
        label="失败恢复",
        zorder=2,
    )

    ax.errorbar(
        positions - width / 2.0,
        plot_frame["degradation_mean"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                plot_frame["degradation_ci_minus"].to_numpy(dtype=float),
                plot_frame["degradation_ci_plus"].to_numpy(dtype=float),
            ]
        ),
        fmt="none",
        ecolor="#303030",
        elinewidth=0.9,
        capsize=2.5,
        zorder=3,
    )
    ax.errorbar(
        positions + width / 2.0,
        plot_frame["recovery_mean"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                plot_frame["recovery_ci_minus"].to_numpy(dtype=float),
                plot_frame["recovery_ci_plus"].to_numpy(dtype=float),
            ]
        ),
        fmt="none",
        ecolor="#303030",
        elinewidth=0.9,
        capsize=2.5,
        zorder=3,
    )

    set_default_axis_style(ax)
    add_zero_reference(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_frame["display_name"].tolist(), rotation=0, ha="center")
    ax.set_ylabel("无动作基线下的指标（%）", fontproperties=CN_FONT)
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
    parser = argparse.ArgumentParser(description="中文绘制主结果风险-恢复图，数据直接写死为表格数值。")
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

    written = plot_main_risk_return(dpi=args.dpi, out_dir=out_dir)
    print_written_paths(written)
    maybe_print_plot_data(args, PLOT_FRAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
