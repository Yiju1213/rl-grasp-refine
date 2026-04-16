from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path("/rl-grasp-refine/outputs/unseen_test_formal")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "generated"
DEFAULT_FORMATS = ("png",)
DEFAULT_DPI = 330
FIGURE_SIZE_4_3 = (6.4, 4.8)
BASELINE_LABEL = "no-action"
DEFAULT_BOOTSTRAP_ITERATIONS = 10000
DEFAULT_BOOTSTRAP_SEED = 17
CONFIDENCE_LEVEL = 0.95
FONT_FAMILY = "DejaVu Sans"
FONT_SIZES = {
    "title": 15,
    "axis_label": 13,
    "tick_label": 11,
    "legend": 11,
    "annotation": 9,
}

MAIN_LABELS = (
    "no-action",
    "rand-action",
    "drop-only-latent-only-128-epi",
    "full-latefus-128-epi",
)

ABLATION_LABELS = (
    "drop-only-latent-only-128-epi",
    "wo-onl-cal_latefus_128-epi",
    "wo-stb-rwd_latefus_128-epi",
    "wo-tac-rwd_latefus_128-epi",
    "wo-tac-sem-n-rwd_latefus_128-epi",
    "full-latefus-128-epi",
)

ORDERED_LABELS = (
    *MAIN_LABELS,
    "wo-onl-cal_latefus_128-epi",
    "wo-stb-rwd_latefus_128-epi",
    "wo-tac-rwd_latefus_128-epi",
    "wo-tac-sem-n-rwd_latefus_128-epi",
)

GROUPS = {
    "main": MAIN_LABELS,
    "ablation": ABLATION_LABELS,
}

DISPLAY_NAMES = {
    "no-action": "No Action",
    "rand-action": "Rand. Action",
    "drop-only-latent-only-128-epi": "Vanilla",
    "wo-onl-cal_latefus_128-epi": "w/o Onl. Cal.",
    "wo-stb-rwd_latefus_128-epi": "w/o Stb. Reward",
    "wo-tac-rwd_latefus_128-epi": "w/o Tac Reward",
    "wo-tac-sem-n-rwd_latefus_128-epi": "w/o Tac Sem.+Reward",
    "full-latefus-128-epi": "Full",
}

ADJUSTED_METRIC_DISPLAY_NAMES = {
    "success_gain": "Success Gain over No-Action",
    "excess_degradation": "Excess Degradation over No-Action",
    "excess_recovery": "Excess Recovery over No-Action",
    "excess_t_cover_delta": "Excess T-Cover Delta over No-Action",
    "excess_t_edge_delta": "Excess T-Edge Delta over No-Action",
    "excess_probability_delta": "Excess Probability Delta over No-Action",
}

# Okabe-Ito color-blind-safe palette plus neutral gray for no-action.
COLORS = {
    "no-action": "#7A7A7A",
    "rand-action": "#CC79A7",
    "drop-only-latent-only-128-epi": "#0072B2",
    "wo-onl-cal_latefus_128-epi": "#E69F00",
    "wo-stb-rwd_latefus_128-epi": "#D55E00",
    "wo-tac-rwd_latefus_128-epi": "#009E73",
    "wo-tac-sem-n-rwd_latefus_128-epi": "#56B4E9",
    "full-latefus-128-epi": "#310D0D",
}

MARKERS = {
    "no-action": "x",
    "rand-action": "*",
    "drop-only-latent-only-128-epi": "o",
    "wo-onl-cal_latefus_128-epi": "s",
    "wo-stb-rwd_latefus_128-epi": "D",
    "wo-tac-rwd_latefus_128-epi": "^",
    "wo-tac-sem-n-rwd_latefus_128-epi": "v",
    "full-latefus-128-epi": "P",
}

RISK_COLOR = "#D55E00"
BENEFIT_COLOR = "#009E73"
