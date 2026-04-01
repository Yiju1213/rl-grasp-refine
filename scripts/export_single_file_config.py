from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.single_file_config import build_single_file_config, dump_single_file_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a split experiment config bundle into one resolved yaml. "
            "The input may be an experiment yaml, a config snapshot directory, or any file inside that snapshot."
        )
    )
    parser.add_argument("input", help="Experiment yaml path, snapshot configs directory, or any file inside it.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output yaml path. If omitted, the merged yaml is printed to stdout.",
    )
    args = parser.parse_args()

    if args.output:
        output_path = dump_single_file_config(args.input, args.output)
        print(output_path)
        return 0

    payload = build_single_file_config(args.input)
    yaml.safe_dump(payload, sys.stdout, sort_keys=False, allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
