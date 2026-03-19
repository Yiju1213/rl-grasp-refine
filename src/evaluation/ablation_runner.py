from __future__ import annotations


class AblationRunner:
    """Reserved ablation entrypoint for future experiments."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run_all(self):
        raise NotImplementedError("AblationRunner is intentionally left as a v1 stub.")
