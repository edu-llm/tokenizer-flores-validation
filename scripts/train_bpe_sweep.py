#!/usr/bin/env python3
"""Train byte vs grapheme BPE tokenizers at 8k/16k/32k on FLORES dev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bpe_train import save_artifact, train_bpe
from src.load_flores import LANGUAGES, load_flores_sentences

ALL_UNITS = ("byte", "grapheme", "grapheme_constrained")
SIZES = (8000, 16000, 32000)

# Subdirectory naming per unit (grapheme_constrained -> gconstr)
UNIT_DIR = {
    "byte": "byte",
    "grapheme": "grapheme",
    "grapheme_constrained": "gconstr",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/bpe"),
        help="Root directory for trained tokenizer artifacts",
    )
    p.add_argument(
        "--units",
        nargs="+",
        choices=ALL_UNITS,
        default=list(ALL_UNITS),
        help="Which tokenizer units to train (default: all)",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional cap per language for smoke tests",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Retrain arms even if a saved artifact already exists",
    )
    return p.parse_args(argv)


def size_label(size: int) -> str:
    return f"{size // 1000}k"


def artifact_label(unit: str, size: int) -> str:
    return f"{UNIT_DIR[unit]}_{size_label(size)}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading FLORES dev (all languages)...")
    by_lang = load_flores_sentences(split="dev")
    if args.max_sentences:
        by_lang = {k: v[: args.max_sentences] for k, v in by_lang.items()}

    texts: list[str] = []
    for code in LANGUAGES:
        if code in by_lang:
            texts.extend(by_lang[code])
    print(f"Training corpus: {len(texts)} sentences from {len(by_lang)} languages")

    for unit in args.units:
        for size in SIZES:
            label = artifact_label(unit, size)
            dest = out_root / label
            if not args.force and (dest / "tokenizer.json").exists():
                print(f"Skipping {label} (artifact exists; use --force to retrain)")
                continue
            print(f"Training {label} -> {dest} ...", flush=True)
            vocab, merges = train_bpe(texts, unit=unit, target_vocab_size=size)
            save_artifact(
                dest,
                vocab=vocab,
                merges=merges,
                unit=unit,
                target_vocab_size=size,
            )
            print(f"  vocab={len(vocab)} merges={len(merges)}")

    print(f"Done. Artifacts under {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
