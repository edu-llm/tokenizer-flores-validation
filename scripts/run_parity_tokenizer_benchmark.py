#!/usr/bin/env python3
"""Train parity-aware BPE with official STAGE1 pretok under resource telemetry.

AWS Batch / local entrypoint. Emits the same ``tokenizer_resource_benchmark``
schema as the official BPE/SuperBPE runner and writes HF-compatible artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import run_monitored_command
from src.parity_official import load_lang_text_dir, train_parity_official


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--dev-dir", type=Path, required=True, help="CR-dev (calibration) texts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--num-bytes", type=int, required=True)
    parser.add_argument("--projection-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--max-rss-gb", type=float)
    parser.add_argument("--min-available-gb", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists (use --force): {path}")
        import shutil

        shutil.rmtree(path)
    path.mkdir(parents=True)


def _run_worker(args: argparse.Namespace) -> int:
    train_by_lang = load_lang_text_dir(args.train_dir)
    dev_by_lang = load_lang_text_dir(args.dev_dir)
    train_files = sorted(str(path) for path in args.train_dir.iterdir() if path.is_file())
    meta = train_parity_official(
        train_by_lang=train_by_lang,
        dev_by_lang=dev_by_lang,
        target_vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        train_bytes=args.num_bytes,
        train_files=train_files,
    )
    print(json.dumps(meta, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.vocab_size <= 256 or args.num_bytes <= 0:
        raise ValueError("vocab-size must exceed 256 and num-bytes must be positive")
    if args.worker:
        return _run_worker(args)

    _prepare_output(args.output_dir, args.force)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--train-dir",
        str(args.train_dir.resolve()),
        "--dev-dir",
        str(args.dev_dir.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--result",
        str(args.result.resolve()),
        "--log",
        str(args.log.resolve()),
        "--vocab-size",
        str(args.vocab_size),
        "--num-bytes",
        str(args.num_bytes),
    ]
    result = run_monitored_command(
        name=f"parity-{args.vocab_size}",
        command=command,
        result_path=args.result,
        log_path=args.log,
        input_bytes=args.num_bytes,
        projection_bytes=args.projection_bytes,
        cwd=ROOT,
        output_path=args.output_dir,
        max_rss_gb=args.max_rss_gb,
        min_available_gb=args.min_available_gb,
        metadata={
            "arm": "parity",
            "vocab_size": args.vocab_size,
            "pretok": "official_stage1",
            "merge_selection": "parity_fair_max_worst_cr_dev",
        },
    )
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
