#!/usr/bin/env python3
"""Run any tokenizer trainer with portable process-tree resource telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import run_monitored_command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--input-bytes", type=int, required=True)
    parser.add_argument("--projection-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--max-rss-gb", type=float)
    parser.add_argument("--min-available-gb", type=float, default=1.0)
    parser.add_argument(
        "--metadata",
        type=json.loads,
        default={},
        help="JSON object included in the result",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after --, e.g. -- python -m train_tokenizer ...",
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("A command is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_monitored_command(
        name=args.name,
        command=args.command,
        result_path=args.result,
        log_path=args.log,
        input_bytes=args.input_bytes,
        projection_bytes=args.projection_bytes,
        cwd=args.cwd,
        output_path=args.output_path,
        poll_seconds=args.poll_seconds,
        max_rss_gb=args.max_rss_gb,
        min_available_gb=args.min_available_gb,
        metadata=args.metadata,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

