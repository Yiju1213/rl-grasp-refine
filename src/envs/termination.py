from __future__ import annotations


class SingleStepTermination:
    """Termination policy for one-action episodes."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def is_done(self) -> bool:
        return True
