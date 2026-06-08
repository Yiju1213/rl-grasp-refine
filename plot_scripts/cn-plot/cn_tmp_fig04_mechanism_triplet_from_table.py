from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, FuncFormatter

CURRENT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPTS_DIR = CURRENT_DIR.parent
if str(PLOT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_SCRIPTS_DIR))

from plot_common import print_written_paths, save_figure, set_default_axis_style, plt

FIGURE_STEM = "cn_tmp_fig04_mechanism_triplet_from_table"
CM_TO_INCH = 1.0 / 2.54
FIGURE_WIDTH_CM = 16.0
FIGURE_HEIGHT_CM = 6

SIMSUN_PATH = CURRENT_DIR / "simsun.ttc"
TIMES_PATH = CURRENT_DIR / "Times_New_Roman.ttf"

TITLE_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
YLABEL_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
LEGEND_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=9.5)
TICK_CN_FONT = fm.FontProperties(fname=str(SIMSUN_PATH), size=10.5)
TICK_NUM_FONT = fm.FontProperties(fname=str(TIMES_PATH), size=8.5)

LABEL_ORDER = [
    "drop-only-latent-only-128-epi",
    "wo-onl-cal_latefus_128-epi",
    "wo-stb-rwd_latefus_128-epi",
    "wo-tac-rwd_latefus_128-epi",
    "wo-tac-sem-n-rwd_latefus_128-epi",
    "full-latefus-128-epi",
]

LABEL_NAMES_CN = {
    "drop-only-latent-only-128-epi": "基础策略",
    "wo-onl-cal_latefus_128-epi": "无在线校准",
    "wo-stb-rwd_latefus_128-epi": "无稳定性奖励",
    "wo-tac-rwd_latefus_128-epi": "无触觉奖励",
    "wo-tac-sem-n-rwd_latefus_128-epi": "无触觉语义及奖励",
    "full-latefus-128-epi": "完整策略",
}

SERIES_STYLES = {
    "drop-only-latent-only-128-epi": {"color": "#5DA5DA", "marker": "o"},
    "wo-onl-cal_latefus_128-epi": {"color": "#F0AD00", "marker": "o"},
    "wo-stb-rwd_latefus_128-epi": {"color": "#D55E00", "marker": "o"},
    "wo-tac-rwd_latefus_128-epi": {"color": "#00A087", "marker": "o"},
    "wo-tac-sem-n-rwd_latefus_128-epi": {"color": "#56B4E9", "marker": "o"},
    "full-latefus-128-epi": {"color": "#2F6DA3", "marker": "o"},
}

PANELS = (
    (
        "excess_t_cover_delta",
        "接触覆盖度变化（%）",
        (0.0, 0.10),
        0.02,
        pd.DataFrame(
            [
                {"label": "drop-only-latent-only-128-epi", "mean": 0.017746, "ci_low": 0.010111, "ci_high": 0.029014},
                {"label": "wo-onl-cal_latefus_128-epi", "mean": 0.030069, "ci_low": 0.017679, "ci_high": 0.047185},
                {"label": "wo-stb-rwd_latefus_128-epi", "mean": 0.016237, "ci_low": 0.010565, "ci_high": 0.022967},
                {"label": "wo-tac-rwd_latefus_128-epi", "mean": 0.024671, "ci_low": 0.016423, "ci_high": 0.036603},
                {"label": "wo-tac-sem-n-rwd_latefus_128-epi", "mean": 0.020897, "ci_low": 0.011799, "ci_high": 0.034302},
                {"label": "full-latefus-128-epi", "mean": 0.031161, "ci_low": 0.016553, "ci_high": 0.054190},
            ]
        ),
    ),
    (
        "excess_t_edge_delta",
        "边界接近度变化（%）",
        (0.0, 0.10),
        0.02,
        pd.DataFrame(
            [
                {"label": "drop-only-latent-only-128-epi", "mean": 0.034981, "ci_low": 0.019407, "ci_high": 0.052922},
                {"label": "wo-onl-cal_latefus_128-epi", "mean": 0.040970, "ci_low": 0.024040, "ci_high": 0.062032},
                {"label": "wo-stb-rwd_latefus_128-epi", "mean": 0.039614, "ci_low": 0.024728, "ci_high": 0.056254},
                {"label": "wo-tac-rwd_latefus_128-epi", "mean": 0.049014, "ci_low": 0.033184, "ci_high": 0.066882},
                {"label": "wo-tac-sem-n-rwd_latefus_128-epi", "mean": 0.035813, "ci_low": 0.019129, "ci_high": 0.055189},
                {"label": "full-latefus-128-epi", "mean": 0.045972, "ci_low": 0.022692, "ci_high": 0.074499},
            ]
        ),
    ),
    (
        "excess_probability_delta",
        "校准稳定性概率变化（%）",
        (0.0, 0.25),
        0.05,
        pd.DataFrame(
            [
                {"label": "drop-only-latent-only-128-epi", "mean": 0.166099, "ci_low": 0.143525, "ci_high": 0.190171},
                {"label": "wo-onl-cal_latefus_128-epi", "mean": 0.195264, "ci_low": 0.164570, "ci_high": 0.229759},
                {"label": "wo-stb-rwd_latefus_128-epi", "mean": 0.087246, "ci_low": 0.076061, "ci_high": 0.099606},
                {"label": "wo-tac-rwd_latefus_128-epi", "mean": 0.094210, "ci_low": 0.081281, "ci_high": 0.108882},
                {"label": "wo-tac-sem-n-rwd_latefus_128-epi", "mean": 0.105278, "ci_low": 0.091198, "ci_high": 0.122083},
                {"label": "full-latefus-128-epi", "mean": 0.102339, "ci_low": 0.085537, "ci_high": 0.122308},
            ]
        ),
    ),
)


def _prepare_panel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["label"] = pd.Categorical(ordered["label"], categories=LABEL_ORDER, ordered=True)
    ordered = ordered.sort_values("label", kind="stable").reset_index(drop=True)
    ordered["label"] = ordered["label"].astype(str)
    ordered["display_name_cn"] = ordered["label"].map(LABEL_NAMES_CN)
    ordered["mean_pct"] = ordered["mean"] * 100.0
    ordered["ci_low_pct"] = ordered["ci_low"] * 100.0
    ordered["ci_high_pct"] = ordered["ci_high"] * 100.0
    return ordered


def _apply_tick_fonts(ax, *, show_y_labels: bool) -> None:
    if show_y_labels:
        for label in ax.get_yticklabels():
            label.set_fontproperties(TICK_CN_FONT)
    for label in ax.get_xticklabels():
        label.set_fontproperties(TICK_NUM_FONT)


def _percent_formatter(x, _pos) -> str:
    return f"{x:.0f}"


def _draw_horizontal_point_ci(ax, frame: pd.DataFrame, *, show_y_labels: bool) -> None:
    y_positions = list(reversed(range(len(frame))))
    for index, row in frame.iterrows():
        style = SERIES_STYLES[str(row["label"])]
        xerr = np.asarray(
            [
                [float(row["mean_pct"] - row["ci_low_pct"])],
                [float(row["ci_high_pct"] - row["mean_pct"])],
            ],
            dtype=float,
        )
        ax.errorbar(
            float(row["mean_pct"]),
            y_positions[index],
            xerr=xerr,
            fmt=style["marker"],
            color=style["color"],
            markersize=4.6,
            elinewidth=1.0,
            capsize=2.6,
            linewidth=1.0,
            zorder=3,
        )
    ax.set_yticks(y_positions)
    if show_y_labels:
        ax.set_yticklabels(frame["display_name_cn"].tolist())
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)


def plot_mechanism_triplet(*, dpi: int, out_dir: Path | str) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH_CM * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH),
        sharex=False,
        sharey=True,
    )
    plot_frames: dict[str, pd.DataFrame] = {}
    for panel_index, (ax, (metric_key, title, xlim, tick_step, frame)) in enumerate(zip(axes, PANELS)):
        plot_frame = _prepare_panel_frame(frame)
        plot_frames[metric_key] = plot_frame
        _draw_horizontal_point_ci(ax, plot_frame, show_y_labels=panel_index == 0)
        set_default_axis_style(ax)
        ax.axvline(0.0, color="#6E6E6E", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_xlim(xlim[0] * 100.0, xlim[1] * 100.0)
        ax.xaxis.set_major_locator(MultipleLocator(tick_step * 100.0))
        ax.xaxis.set_major_formatter(FuncFormatter(_percent_formatter))
        ax.set_title(title, fontproperties=TITLE_FONT)
        _apply_tick_fonts(ax, show_y_labels=panel_index == 0)

    plt.tight_layout(pad=0.15)
    written = save_figure(fig, out_dir=out_dir, stem=FIGURE_STEM, formats=("png",), dpi=dpi)
    plt.close(fig)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中文绘制机制三联图，数据直接提取自 plot_data_ablation.txt 的 fig04 段。")
    parser.add_argument(
        "--out-dir",
        default=str(CURRENT_DIR / "generated/ablation"),
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

    written = plot_mechanism_triplet(dpi=args.dpi, out_dir=out_dir)
    print_written_paths(written)
    if args.print_data:
        maybe_payload = {metric_key: _prepare_panel_frame(frame) for metric_key, _, _, _, frame in PANELS}
        from plot_common import maybe_print_plot_data  # local import to keep top-level imports minimal
        maybe_print_plot_data(args, maybe_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
