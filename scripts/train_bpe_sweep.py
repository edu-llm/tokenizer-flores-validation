#!/usr/bin/env python3
"""Train byte / grapheme / parity BPE tokenizers at 8k/16k/32k on FLORES dev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bpe_train import (
    build_lang_word_freqs,
    build_weighted_lang_freqs,
    build_weighted_word_freqs,
    o200k_pattern,
    save_artifact,
    train_bpe,
    train_bpe_from_freqs,
    train_parity_bpe_from_lang_freqs,
)
from src.load_flores import LANGUAGES, load_flores_sentences

ALL_UNITS = ("byte", "grapheme", "grapheme_constrained", "parity")
SIZES = (8000, 16000, 32000)

ENGLISH_AGGRESSIVE_WEIGHTS: dict[str, float] = {
    "eng_Latn": 0.85,
    "zho_Hans": 0.05,
    "ukr_Cyrl": 0.025,
    "hun_Latn": 0.02,
    "arz_Arab": 0.015,
    "swh_Latn": 0.01,
    "ary_Arab": 0.01,
    "hau_Latn": 0.005,
    "amh_Ethi": 0.005,
    "ory_Orya": 0.005,
    "quy_Latn": 0.0025,
    "grn_Latn": 0.0025,
}

# Subdirectory naming per unit (grapheme_constrained -> gconstr)
UNIT_DIR = {
    "byte": "byte",
    "grapheme": "grapheme",
    "grapheme_constrained": "gconstr",
    "parity": "parity",
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
    p.add_argument(
        "--skew",
        choices=("english_aggressive",),
        default=None,
        help="Skew training corpus byte-mass toward English (web-realistic)",
    )
    return p.parse_args(argv)


def size_label(size: int) -> str:
    return f"{size // 1000}k"


def artifact_label(unit: str, size: int) -> str:
    return f"{UNIT_DIR[unit]}_{size_label(size)}"


def _print_skew_mass_shares(by_lang: dict[str, list[str]], weights: dict[str, float]) -> None:
    """Report achieved byte-mass shares after reweighting (sanity check)."""
    from src.bpe_train import _lang_raw_byte_mass, build_word_freqs

    pat = o200k_pattern()
    raw: dict[str, float] = {}
    for lang in weights:
        if lang not in by_lang:
            continue
        raw[lang] = _lang_raw_byte_mass(build_word_freqs(by_lang[lang], "byte", pat))
    total_raw = sum(raw.values())
    weighted: dict[str, float] = {}
    for lang, target in weights.items():
        if lang not in raw or total_raw <= 0:
            continue
        raw_share = raw[lang] / total_raw
        weighted[lang] = target
        print(f"  {lang}: raw_share={raw_share:.4f} -> target={target:.4f}")
    print(f"  (target shares sum={sum(weighted.values()):.4f})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading FLORES dev (all languages)...")
    by_lang = load_flores_sentences(split="dev")
    if args.max_sentences:
        by_lang = {k: v[: args.max_sentences] for k, v in by_lang.items()}

    use_skew = args.skew == "english_aggressive"
    if use_skew:
        weights = ENGLISH_AGGRESSIVE_WEIGHTS
        print("English-aggressive skew (byte-mass targets):")
        _print_skew_mass_shares(by_lang, weights)
        n_sents = sum(len(v) for v in by_lang.values())
        print(f"Training corpus: {n_sents} sentences from {len(by_lang)} languages (weighted)")
    else:
        texts: list[str] = []
        for code in LANGUAGES:
            if code in by_lang:
                texts.extend(by_lang[code])
        print(f"Training corpus: {len(texts)} sentences from {len(by_lang)} languages")

    # Parallel CR-dev for parity (unweighted FLORES — line-normalized fair-max).
    need_parity = "parity" in args.units
    if need_parity:
        print("Building unweighted FLORES-dev CR counters for parity-aware selection...")
        dev_by_lang = build_lang_word_freqs(by_lang, unit="byte")

    for unit in args.units:
        skewed_combined = None
        skewed_by_lang = None
        if use_skew:
            if unit == "parity":
                skewed_by_lang = build_weighted_lang_freqs(by_lang, weights, "byte")
            else:
                skewed_combined = build_weighted_word_freqs(by_lang, weights, unit)

        for size in SIZES:
            label = artifact_label(unit, size)
            dest = out_root / label
            if not args.force and (dest / "tokenizer.json").exists():
                print(f"Skipping {label} (artifact exists; use --force to retrain)")
                continue
            print(f"Training {label} -> {dest} ...", flush=True)

            if unit == "parity":
                if not use_skew:
                    train_by_lang = build_lang_word_freqs(by_lang, unit="byte")
                else:
                    assert skewed_by_lang is not None
                    train_by_lang = skewed_by_lang
                vocab, merges = train_parity_bpe_from_lang_freqs(
                    train_by_lang,
                    dev_by_lang,
                    target_vocab_size=size,
                )
            elif use_skew:
                assert skewed_combined is not None
                vocab, merges = train_bpe_from_freqs(
                    skewed_combined, unit=unit, target_vocab_size=size
                )
            else:
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
