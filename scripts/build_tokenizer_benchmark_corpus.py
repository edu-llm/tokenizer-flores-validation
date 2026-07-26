#!/usr/bin/env python3
"""Build a deterministic byte-bounded corpus and manifest for local/AWS benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json, build_round_robin_corpus, discover_input_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input text files or directories (recursively scanned)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output UTF-8 text corpus")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Output portable JSON manifest",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        required=True,
        help="Maximum UTF-8 corpus bytes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = discover_input_files(args.input, args.output)
    manifest = build_round_robin_corpus(files, args.output, args.target_bytes)
    atomic_write_json(args.manifest, manifest)
    print(
        f"Built {manifest['actual_bytes']:,} bytes from {len(files)} files "
        f"-> {args.output.resolve()}"
    )
    print(f"Manifest: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

