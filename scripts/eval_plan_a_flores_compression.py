#!/usr/bin/env python3
"""Evaluate Plan A BPE / SuperBPE compression metrics on FLORES-200.

Reports fertility, chars/token, token premium vs English, STRR, STFR, and Gini
of tokens-per-line on FLORES ``devtest`` (held-out eval).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load_flores import CONTINENT, load_flores_sentences
from src.metrics import (
    attach_token_premiums,
    gini_for_tokenizer,
    metrics_for_language,
    rows_to_dicts,
)
from src.official_bpe_encode import load_official_bpe_tokenizer
from src.tokenizers_registry import TokenizerSpec

PLAN_A_FLORES_LANGS = [
    "eng_Latn",
    "amh_Ethi",
    "hau_Latn",
    "swh_Latn",
    "ukr_Cyrl",
    "pol_Latn",
    "hun_Latn",
    "tel_Telu",
    "ory_Orya",
    "zho_Hans",
    "tur_Latn",
    "ayr_Latn",
    "quy_Latn",
    "grn_Latn",
]

LANG_NAMES = {
    "eng_Latn": "English",
    "amh_Ethi": "Amharic",
    "hau_Latn": "Hausa",
    "swh_Latn": "Swahili",
    "ukr_Cyrl": "Ukrainian",
    "pol_Latn": "Polish",
    "hun_Latn": "Hungarian",
    "tel_Telu": "Telugu",
    "ory_Orya": "Odia",
    "zho_Hans": "Mandarin",
    "tur_Latn": "Turkish",
    "ayr_Latn": "Aymara",
    "quy_Latn": "Quechua",
    "grn_Latn": "Guarani",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokenizers-root",
        type=Path,
        default=Path("artifacts/plan_a/research_cpu/tokenizers/pilot"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/plan_a/research_cpu/results/pilot/flores_compression"),
    )
    p.add_argument("--max-sentences", type=int, default=None)
    p.add_argument("--split", default="devtest")
    return p.parse_args()


def _surface_fn(tok: Tokenizer):
    def surface(token_id: int) -> str:
        return tok.decode([token_id])

    return surface


def load_arm_spec(arm: str, directory: Path) -> TokenizerSpec:
    tok_json = directory / "tokenizer.json"
    if tok_json.is_file():
        tok = Tokenizer.from_file(str(tok_json))
    else:
        tok = load_official_bpe_tokenizer(directory)

    def encode(text: str) -> list[int]:
        return tok.encode(text).ids

    return TokenizerSpec(
        id=arm,
        name=f"Plan A {arm}",
        source=str(directory),
        is_frontier=False,
        encode=encode,
        surface=_surface_fn(tok),
        leading_space_for_words=True,
    )


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

    rows = []
    for arm, spec in specs.items():
        print(f"Evaluating {arm} on {len(by_lang)} languages...", flush=True)
        for lang, sentences in by_lang.items():
            rows.append(metrics_for_language(spec, lang, sentences))
    rows = attach_token_premiums(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = rows_to_dicts(rows)
    for r in records:
        r["language_name"] = LANG_NAMES.get(r["language"], r["language"])
        r["continent"] = CONTINENT.get(r["language"], "Other")

    df = pd.DataFrame(records)
    csv_path = args.out_dir / "metrics.csv"
    json_path = args.out_dir / "metrics.json"
    df.to_csv(csv_path, index=False)
    payload = {
        "schema_version": 1,
        "kind": "plan_a_flores_compression",
        "split": args.split,
        "tokenizers_root": str(args.tokenizers_root),
        "languages": sorted(by_lang),
        "n_sentences": {k: len(v) for k, v in by_lang.items()},
        "gini_tokens_per_line": {
            arm: gini_for_tokenizer(rows, arm) for arm in specs
        },
        "metrics": records,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Wide premium / fertility / STFR tables for quick reading
    for metric in ("token_premium", "fertility", "chars_per_token", "stfr", "strr"):
        wide = df.pivot(index="language", columns="tokenizer_id", values=metric)
        wide = wide.reindex(PLAN_A_FLORES_LANGS)
        wide.to_csv(args.out_dir / f"{metric}_wide.csv")

    print("\n=== Token premium vs English (FLORES devtest) ===")
    prem = df.pivot(index="language", columns="tokenizer_id", values="token_premium")
    prem = prem.reindex([c for c in PLAN_A_FLORES_LANGS if c in prem.index])
    print(prem.round(3).to_string())
    print("\n=== Fertility (tokens/word; omit Mandarin interpretation) ===")
    fert = df.pivot(index="language", columns="tokenizer_id", values="fertility")
    fert = fert.reindex([c for c in PLAN_A_FLORES_LANGS if c in fert.index])
    print(fert.round(3).to_string())
    print("\n=== STFR (share of length-1 tokens) ===")
    stfr = df.pivot(index="language", columns="tokenizer_id", values="stfr")
    stfr = stfr.reindex([c for c in PLAN_A_FLORES_LANGS if c in stfr.index])
    print(stfr.round(3).to_string())
    print("\n=== Gini of tokens-per-line ===")
    for arm, g in payload["gini_tokens_per_line"].items():
        print(f"  {arm}: {g:.4f}")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
