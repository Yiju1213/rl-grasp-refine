from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path("/rl-grasp-refine/outputs/unseen_test_formal")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "generated"
DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 330

ORDERED_LABELS = (
    "stb-rwd-5x-full-latefus-128-epi",
    "drop-only-latent-only-128-epi",
    "drop-only",
    "stb_rwd-1x-full",
    "stb-rwd-5x-full",
    "stb-rwd-10x-full",
    "stb-rwd-15x-full",
    "wo-tac-rwd",
    "wo-tac-sem-n-rwd",
)

GROUPS = {
    "all_formal": ORDERED_LABELS,
    "group_a": ORDERED_LABELS,
    "group_b": (
        "drop-only",
        "stb-rwd-5x-full",
        "stb-rwd-10x-full",
        "stb-rwd-15x-full",
    ),
    "self": (
        "drop-only",
        "stb-rwd-5x-full-latefus-128-epi",
        "drop-only-latent-only-128-epi",
        "stb-rwd-5x-full",
        "stb_rwd-1x-full",
        
    ),
}

DISPLAY_NAMES = {
    "drop-only": "Drop-Only",
    "stb_rwd-1x-full": "Stb x1",
    "stb-rwd-5x-full": "Stb x5",
    "stb-rwd-10x-full": "Stb x10",
    "stb-rwd-15x-full": "Stb x15",
    "stb-rwd-5x-full-latefus-128-epi": "Stb x5 LateFus 128 Epi",
    "drop-only-latent-only-128-epi": "Drop-Only Latent Only 128 Epi",
    "wo-tac-rwd": "w/o Tac Reward",
    "wo-tac-sem-n-rwd": "w/o Tac Sem+Reward",
}

COLORS = {
    "drop-only": "#4C78A8",
    "stb-rwd-5x-full": "#F58518",
    "stb-rwd-10x-full": "#54A24B",
    "stb-rwd-15x-full": "#E45756",
    "stb-rwd-5x-full-latefus-128-epi": "#72B7B2",
    "drop-only-latent-only-128-epi": "#874ADB",
    "wo-tac-rwd": "#B279A2",
    "wo-tac-sem-n-rwd": "#FF9DA6",
    "stb_rwd-1x-full": "#203A43",
}

MARKERS = {
    "drop-only": "o",
    "stb-rwd-5x-full": "s",
    "stb-rwd-10x-full": "^",
    "stb-rwd-15x-full": "D",
    "stb-rwd-5x-full-latefus-128-epi": "P",
    "drop-only-latent-only-128-epi": '<',
    "wo-tac-rwd": "X",
    "wo-tac-sem-n-rwd": "v",
}

RISK_COLOR = "#E45756"
BENEFIT_COLOR = "#54A24B"

