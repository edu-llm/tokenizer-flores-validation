#!/usr/bin/env python3
"""Plan A scale FLORES suite: fertility, token premium, Zipf deviation (no LM BPB).

BPB is deferred until models are trained. This script reports tokenizer-only metrics
on the Plan A 6-language set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load_flores import load_flores_sentences
from src.metrics import attach_token_premiums, metrics_for_language, rows_to_dicts
from src.plan_a_langs import LANG_NAMES, PLAN_A_CODES, REGION
from src.tokenizers_registry import TokenizerSpec
from src.zipf import (
    counts_from_interned,
    draw_counts_matched,
    draw_counts_resample,
    summarize_bootstrap,
)

# Reuse Zipf measurement helpers without pulling the full study language set.
sys.path.insert(0, str(ROOT / "scripts"))
from run_vocab_profile import load_concatenated  # noqa: E402
from run_zipf_eval import (  # noqa: E402
    BUDGET_FRACTION,
    METRICS,
    build_unit_cache,
    measure,
    total_units,
)

PLAN_A_FLORES_LANGS = list(PLAN_A_CODES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokenizers-root",
        type=Path,
        required=True,
        help="Directory with bpe/ and superbpe/ each containing tokenizer.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "plan_a" / "scale" / "flores_suite",
    )
    p.add_argument("--split", default="devtest")
    p.add_argument("--max-sentences", type=int, default=None)
    p.add_argument(
        "--zipf-bootstrap",
        type=int,
        default=50,
        help="Bootstrap draws per Zipf cell (full study uses 200; 50 is enough for suite)",
    )
    p.add_argument("--seed", type=int, default=20260803)
    p.add_argument(
        "--skip-zipf",
        action="store_true",
        help="Only fertility + premium (smoke)",
    )
    return p.parse_args()


def _surface_fn(tok: Tokenizer):
    def surface(token_id: int) -> str:
        return tok.decode([token_id])

    return surface


def load_arm_spec(arm: str, directory: Path) -> TokenizerSpec:
    tok_json = directory / "tokenizer.json"
    if not tok_json.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json under {directory}")
    tok = Tokenizer.from_file(str(tok_json))

    def encode(text: str) -> list[int]:
        return tok.encode(text).ids

    return TokenizerSpec(
        id=arm,
        name=f"Plan A scale {arm}",
        source=str(directory),
        is_frontier=False,
        encode=encode,
        surface=_surface_fn(tok),
        leading_space_for_words=True,
    )


def run_compression(
    specs: dict[str, TokenizerSpec],
    by_lang: dict[str, list[str]],
) -> list[dict]:
    rows = []
    for arm, spec in specs.items():
        print(f"Compression metrics: {arm}…", flush=True)
        for lang, sentences in by_lang.items():
            rows.append(metrics_for_language(spec, lang, sentences))
    rows = attach_token_premiums(rows)
    records = rows_to_dicts(rows)
    for r in records:
        r["language_name"] = LANG_NAMES.get(r["language"], r["language"])
        r["region"] = REGION.get(r["language"]) or "Reference"
        # Strip metrics not requested for this suite from the *reported* record.
        for drop in ("stfr", "strr", "chars_per_token", "fertility_per_char", "is_cjk"):
            r.pop(drop, None)
    return records


def evaluate_zipf_cell(
    cache,
    view: str,
    budget: int,
    n_boot: int,
    seed: int,
    full_vocab_size: int,
) -> dict:
    reference = measure(counts_from_interned(cache), full_vocab_size, start_hint=None)
    hint = None
    if math.isfinite(reference["alpha"]):
        hint = [(reference["alpha"], reference["b"])]

    draws = []
    used = []
    budget_met = True
    for i in range(n_boot):
        rng = np.random.default_rng(seed + i)
        if view == "matched_token":
            counts, drawn, met = draw_counts_matched(cache, budget, rng)
            budget_met = budget_met and met
            used.append(drawn)
        else:
            counts = draw_counts_resample(cache, rng)
            used.append(int(counts.sum()))
        draws.append(measure(counts, full_vocab_size, start_hint=hint))

    row: dict = {
        "view": view,
        "budget": budget if view == "matched_token" else total_units(cache),
        "budget_met": budget_met,
        "n_units_full": total_units(cache),
        "n_bootstrap": n_boot,
        "full_corpus_ks_zipf": reference["ks_zipf"],
        "full_corpus_alpha": reference["alpha"],
    }
    for metric in METRICS:
        summary = summarize_bootstrap([d[metric] for d in draws])
        row[metric] = reference[metric] if view == "matched_sentence" else summary["mean"]
        row[f"{metric}_lo"] = summary["lo"]
        row[f"{metric}_hi"] = summary["hi"]
    return row


def run_zipf(
    specs: dict[str, TokenizerSpec],
    *,
    bootstrap: int,
    seed: int,
    max_sentences: int | None,
) -> pd.DataFrame:
    sentences = load_concatenated(PLAN_A_FLORES_LANGS)
    if max_sentences:
        sentences = {k: v[:max_sentences] for k, v in sentences.items()}

    # Matched budget from smallest token corpus across arms × langs.
    per_lang_token_totals: list[int] = []
    caches: dict[tuple[str, str], list] = {}
    for arm, spec in specs.items():
        for lang, sents in sentences.items():
            cache = build_unit_cache(sents, "token", spec)
            caches[(arm, lang)] = cache
            per_lang_token_totals.append(total_units(cache))
    budget = max(1, int(min(per_lang_token_totals) * BUDGET_FRACTION))

    rows = []
    for arm, spec in specs.items():
        # vocab size from max id seen is awkward; use 100000 for scale gigatoken.
        full_vocab = 100_000
        for lang in PLAN_A_FLORES_LANGS:
            cache = caches[(arm, lang)]
            print(f"Zipf token-unit: {arm} / {lang}…", flush=True)
            for view in ("matched_token", "matched_sentence"):
                cell = evaluate_zipf_cell(
                    cache, view, budget, bootstrap, seed, full_vocab
                )
                cell.update(
                    {
                        "tokenizer_id": arm,
                        "language": lang,
                        "language_name": LANG_NAMES.get(lang, lang),
                        "unit": "token",
                    }
                )
                rows.append(cell)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    arms = {
        "bpe": args.tokenizers_root / "bpe",
        "superbpe": args.tokenizers_root / "superbpe",
    }
    for arm, path in arms.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing tokenizer arm {arm}: {path}")

    specs = {arm: load_arm_spec(arm, path) for arm, path in arms.items()}

    by_lang = load_flores_sentences(PLAN_A_FLORES_LANGS, split=args.split)
    if args.max_sentences:
        by_lang = {k: v[: args.max_sentences] for k, v in by_lang.items()}

    records = run_compression(specs, by_lang)
    # eng premium must be exactly 1.0
    for r in records:
        if r["language"] == "eng_Latn" and r["token_premium"] != 1.0:
            raise SystemExit(
                f"eng_Latn premium must be 1.0, got {r['token_premium']} for {r['tokenizer_id']}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    keep_cols = [
        "tokenizer_id",
        "language",
        "language_name",
        "region",
        "n_sentences",
        "ctc",
        "n_words",
        "n_chars",
        "fertility",
        "token_premium",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df.to_csv(args.out_dir / "metrics.csv", index=False)
    payload = {
        "schema_version": 1,
        "kind": "plan_a_scale_flores_suite",
        "split": args.split,
        "tokenizers_root": str(args.tokenizers_root),
        "languages": PLAN_A_FLORES_LANGS,
        "metrics_reported": ["fertility", "token_premium", "ks_zipf"],
        "bpb_deferred": (
            "LM bits-per-byte requires trained models; deferred until Plan B / OLMo eval"
        ),
        "metrics": records,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    for metric in ("token_premium", "fertility"):
        wide = df.pivot(index="language", columns="tokenizer_id", values=metric)
        wide = wide.reindex(PLAN_A_FLORES_LANGS)
        wide.to_csv(args.out_dir / f"{metric}_wide.csv")

    print("\n=== Token premium vs English ===")
    print(
        df.pivot(index="language", columns="tokenizer_id", values="token_premium")
        .reindex(PLAN_A_FLORES_LANGS)
        .round(4)
        .to_string()
    )
    print("\n=== Fertility (tokens/word) ===")
    print(
        df.pivot(index="language", columns="tokenizer_id", values="fertility")
        .reindex(PLAN_A_FLORES_LANGS)
        .round(4)
        .to_string()
    )

    if not args.skip_zipf:
        zipf_dir = args.out_dir / "zipf"
        zipf_dir.mkdir(parents=True, exist_ok=True)
        zipf_df = run_zipf(
            specs,
            bootstrap=args.zipf_bootstrap,
            seed=args.seed,
            max_sentences=args.max_sentences,
        )
        zipf_df.to_csv(zipf_dir / "zipf_fits.csv", index=False)
        # Headline: matched_sentence token-unit ks_zipf
        headline = zipf_df[zipf_df["view"] == "matched_sentence"][
            ["language", "tokenizer_id", "ks_zipf", "alpha", "n_types"]
        ]
        headline.to_csv(zipf_dir / "ks_zipf_wide_source.csv", index=False)
        ks_wide = headline.pivot(index="language", columns="tokenizer_id", values="ks_zipf")
        ks_wide = ks_wide.reindex(PLAN_A_FLORES_LANGS)
        ks_wide.to_csv(zipf_dir / "ks_zipf_wide.csv")
        print("\n=== Zipf deviation (ks_zipf, matched_sentence, token unit) ===")
        print(ks_wide.round(4).to_string())
        (zipf_dir / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "plan_a_scale_zipf_suite",
                    "unit": "token",
                    "languages": PLAN_A_FLORES_LANGS,
                    "bootstrap": args.zipf_bootstrap,
                    "ks_zipf_matched_sentence": ks_wide.to_dict(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"\nWrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
