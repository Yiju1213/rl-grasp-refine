from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.best_checkpoint_pipeline import run_best_checkpoint_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple best checkpoints under one unseen-test protocol and "
            "write per-object, per-run, and experiment summary tables."
        )
    )
    parser.add_argument(
        "manifest",
        help="Path to the evaluation manifest yaml.",
    )
    args = parser.parse_args(argv)

    output_paths = run_best_checkpoint_evaluation(args.manifest)
    failed_experiments = output_paths.get("failed_experiments", {})
    if failed_experiments:
        print(
            json.dumps(
                {"failed_experiments": failed_experiments},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    print(output_paths["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
