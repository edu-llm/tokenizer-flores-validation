"""Stage 1: profile how a tokenizer's vocabulary is allocated across languages.

Writes two complementary views to ``--out-dir``:

*Static* -- every mergeable rank classified by Unicode script, plus attribution
of the partial-UTF-8 byte fragments. A property of the tokenizer alone.

*Empirical* -- for every FLORES-200 language, which token ids the language
actually reaches, how much of that is shared with the control language, and what
share of its distinct word types survive as single tokens.

The empirical pass covers all 204 FLORES languages by default even though the
Zipf study uses 18: it is cheap, and it documents the allocation axis the 18 were
selected from rather than asserting the selection.

Usage::

    python scripts/run_vocab_profile.py --out-dir results/zipf
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.load_flores import FLORES_ROOT  # noqa: E402
from src.tokenizers_registry import load_tokenizers  # noqa: E402
from src.vocab_profile import (  # noqa: E402
    BYTE_FRAGMENT,
    active_vocabulary,
    encode_per_sentence,
    exclusivity_stats,
    fragment_token_ids,
    fragment_usage,
    script_allocation,
    whole_word_coverage,
)
from src.zipf import matched_token_draw  # noqa: E402
from src.zipf_langs import (  # noqa: E402
    BY_CODE,
    CONTROL_LANG,
    MATCHED_TOKEN_BUDGET,
    SPLITS,
    STUDY_CODES,
)

# Draws averaged for the size-controlled type count. A single draw is noisy;
# five is enough to stabilise the count without slowing the 204-language pass.
MATCHED_DRAWS = 5


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/zipf"))
    p.add_argument(
        "--tokenizer",
        default="o200k",
        help="Tokenizer id for the empirical pass (default: o200k)",
    )
    p.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Explicit FLORES codes (default: every language present on disk)",
    )
    p.add_argument(
        "--all-langs-word-coverage",
        action="store_true",
        help=(
            "Compute type-level whole-word coverage for every language, not just "
            "the 18 study languages. Adds one encode per distinct word type."
        ),
    )
    p.add_argument(
        "--rare-max-langs",
        type=int,
        default=5,
        help="A token is 'rare' if at most this many languages reach it",
    )
    p.add_argument("--seed", type=int, default=20260731)
    return p.parse_args(argv)


def discover_languages() -> List[str]:
    """Every FLORES code with files present in all required splits."""
    per_split = []
    for split in SPLITS:
        split_dir = FLORES_ROOT / split
        if not split_dir.is_dir():
            raise RuntimeError(
                f"FLORES split directory not found: {split_dir.resolve()}. "
                "See README.md for the download step, or set FLORES200_DIR."
            )
        per_split.append({p.name[: -len(f".{split}")] for p in split_dir.glob(f"*.{split}")})
    common = set.intersection(*per_split)
    return sorted(common)


def load_concatenated(codes: List[str]) -> Dict[str, List[str]]:
    """Concatenate the study splits per language.

    Read directly rather than via ``load_flores_sentences`` because that helper
    enforces equal line counts across everything passed in a single call, and
    dev (997) and devtest (1,012) legitimately differ.
    """
    out: Dict[str, List[str]] = {}
    for code in codes:
        lines: List[str] = []
        for split in SPLITS:
            path = FLORES_ROOT / split / f"{code}.{split}"
            if not path.is_file():
                lines = []
                break
            lines.extend(path.read_text(encoding="utf-8").splitlines())
        if lines:
            out[code] = lines
    return out


def write_static_allocation(out_dir: Path, encoding_name: str) -> dict:
    print(f"Classifying {encoding_name} vocabulary by Unicode script...")
    allocation = script_allocation(encoding_name)

    rows = pd.DataFrame(
        [
            {"script": r.script, "n_tokens": r.n_tokens, "share": r.share}
            for r in allocation.rows
        ]
    )
    rows.to_csv(out_dir / "vocab_allocation_by_script.csv", index=False)

    fragments = pd.DataFrame(
        [
            {
                "block": block,
                "n_tokens": count,
                "n_certain": allocation.fragment_blocks_certain.get(block, 0),
            }
            for block, count in allocation.fragment_blocks.items()
        ]
    )
    fragments.to_csv(out_dir / "vocab_allocation_fragments.csv", index=False)

    mixed = pd.DataFrame(
        [{"scripts": combo, "n_tokens": count} for combo, count in allocation.mixed_combinations.items()]
    )
    mixed.to_csv(out_dir / "vocab_allocation_mixed.csv", index=False)

    n_fragment = next(
        (r.n_tokens for r in allocation.rows if r.script == BYTE_FRAGMENT), 0
    )
    print(f"  {allocation.n_vocab:,} mergeable ranks; {n_fragment:,} partial-UTF-8 fragments")
    for row in allocation.rows[:8]:
        print(f"    {row.script:24s} {row.n_tokens:7,d}  {100 * row.share:6.3f}%")

    return {
        "encoding": allocation.encoding,
        "n_vocab": allocation.n_vocab,
        "by_script": [asdict(r) for r in allocation.rows],
        "fragment_blocks": allocation.fragment_blocks,
        "fragment_blocks_certain": allocation.fragment_blocks_certain,
        "mixed_combinations": allocation.mixed_combinations,
    }


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_tokenizers([args.tokenizer])[args.tokenizer]
    static = write_static_allocation(out_dir, spec.source)
    n_vocab = static["n_vocab"]

    codes = args.languages or discover_languages()
    print(f"\nEncoding {len(codes)} languages x {'+'.join(SPLITS)} with {args.tokenizer}...")
    corpora = load_concatenated(codes)
    if CONTROL_LANG not in corpora:
        raise RuntimeError(
            f"control language {CONTROL_LANG} not found on disk; "
            "exclusivity metrics are undefined without it"
        )

    frag_ids = fragment_token_ids(spec.source)
    active_counts: Dict[str, Counter] = {}
    base: Dict[str, dict] = {}

    for i, (code, sentences) in enumerate(sorted(corpora.items()), start=1):
        per_sentence = encode_per_sentence(spec, sentences)
        counts = active_vocabulary(per_sentence)
        n_tokens = int(sum(counts.values()))
        n_types = len(counts)

        matched_types: List[int] = []
        budget_met = True
        for draw in range(MATCHED_DRAWS):
            rng = np.random.default_rng(args.seed + draw)
            drawn_counts, _, met = matched_token_draw(
                per_sentence, MATCHED_TOKEN_BUDGET, rng
            )
            matched_types.append(len(drawn_counts))
            budget_met = budget_met and met

        frag = fragment_usage(counts, frag_ids)
        active_counts[code] = counts
        base[code] = {
            "code": code,
            "n_sentences": len(sentences),
            "n_tokens": n_tokens,
            "n_types": n_types,
            "share_of_vocab": n_types / n_vocab,
            "tokens_per_type": (n_tokens / n_types) if n_types else float("nan"),
            "n_types_matched_mean": float(np.mean(matched_types)),
            "share_of_vocab_matched": float(np.mean(matched_types)) / n_vocab,
            "matched_budget_met": budget_met,
            "n_fragment_types": frag.n_fragment_types,
            "share_fragment_types": frag.share_types,
            "share_fragment_mass": frag.share_mass,
        }
        if i % 25 == 0 or i == len(corpora):
            print(f"  {i}/{len(corpora)} languages encoded")

    print(f"\nComputing exclusivity against {CONTROL_LANG} over {len(active_counts)} languages...")
    exclusivity = exclusivity_stats(
        active_counts, control=CONTROL_LANG, rare_max_langs=args.rare_max_langs
    )

    coverage_codes = list(corpora) if args.all_langs_word_coverage else [
        c for c in STUDY_CODES if c in corpora
    ]
    print(f"Computing type-level whole-word coverage for {len(coverage_codes)} languages...")
    coverage: Dict[str, dict] = {}
    for code in coverage_codes:
        result = whole_word_coverage(spec, corpora[code])
        coverage[code] = {
            "n_word_types": result.n_word_types,
            "n_single_token_word_types": result.n_single_token,
            "whole_word_coverage": result.coverage,
        }

    records = []
    for code, row in base.items():
        study = BY_CODE.get(code)
        excl = exclusivity[code]
        records.append(
            {
                **row,
                "language_name": study.name if study else "",
                "script": study.script if study else "",
                "tier": study.tier if study else "",
                "in_study": study is not None,
                "in_locked_12": bool(study and study.in_locked_12),
                "has_word_boundary": (study.has_word_boundary if study else None),
                "n_not_in_control": excl.n_not_in_control,
                "share_not_in_control": excl.share_not_in_control,
                "share_mass_not_in_control": excl.share_mass_not_in_control,
                "n_rare": excl.n_rare,
                "share_rare": excl.share_rare,
                "share_mass_rare": excl.share_mass_rare,
                **coverage.get(
                    code,
                    {
                        "n_word_types": None,
                        "n_single_token_word_types": None,
                        "whole_word_coverage": None,
                    },
                ),
            }
        )

    df = pd.DataFrame(records).sort_values("share_of_vocab", ascending=False)
    df.to_csv(out_dir / "lang_vocab_profile.csv", index=False)

    payload = {
        "tokenizer": args.tokenizer,
        "splits": list(SPLITS),
        "matched_token_budget": MATCHED_TOKEN_BUDGET,
        "matched_draws": MATCHED_DRAWS,
        "rare_max_langs": args.rare_max_langs,
        "control_language": CONTROL_LANG,
        "static_allocation": static,
        "languages": records,
    }
    (out_dir / "vocab_profile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    study = df[df["in_study"]]
    print("\nStudy languages by active-vocabulary share:")
    print(
        study[
            [
                "code",
                "language_name",
                "script",
                "n_tokens",
                "n_types",
                "share_of_vocab",
                "share_mass_not_in_control",
                "share_fragment_mass",
                "whole_word_coverage",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(f"\nWrote Stage 1 outputs to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
