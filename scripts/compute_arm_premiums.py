#!/usr/bin/env python3
"""Compute per-language token premiums for BPE / SuperBPE / Parity artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json
from src.official_bpe_encode import load_official_bpe_tokenizer, token_premiums_vs_english
from src.parity_official import load_lang_text_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpe-dir", type=Path, required=True)
    parser.add_argument("--superbpe-dir", type=Path, required=True)
    parser.add_argument("--parity-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--reference-lang", default="eng_Latn")
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    by_lang = load_lang_text_dir(args.calibration_dir)
    premiums = {
        "bpe": token_premiums_vs_english(
            load_official_bpe_tokenizer(args.bpe_dir),
            by_lang,
            reference_lang=args.reference_lang,
        ),
        "superbpe": token_premiums_vs_english(
            load_official_bpe_tokenizer(args.superbpe_dir),
            by_lang,
            reference_lang=args.reference_lang,
        ),
        "parity": token_premiums_vs_english(
            load_official_bpe_tokenizer(args.parity_dir),
            by_lang,
            reference_lang=args.reference_lang,
        ),
    }
    atomic_write_json(args.result, premiums)
    print(json.dumps(premiums, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
