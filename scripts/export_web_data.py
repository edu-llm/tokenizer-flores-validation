"""Export results/metrics.json into web/data.js for the static viewer."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.json"
LANGS_PATH = ROOT / "artifacts" / "languages.json"
OUT_PATH = ROOT / "web" / "data.js"

TOKENIZER_ORDER = [
    "o200k",
    "glm",
    "llama",
    "qwen",
    "multi",
    "unigram",
    "wordpiece",
    "superbpe",
]
TOKENIZER_LABELS = {
    "o200k": "o200k",
    "glm": "GLM-5.2",
    "llama": "Llama 3.1",
    "qwen": "Qwen2.5",
    "multi": "NLLB-200",
    "unigram": "mT5 (Unigram)",
    "wordpiece": "mBERT (WordPiece)",
    "superbpe": "SuperBPE t180k",
}

METRIC_META = [
    {
        "id": "fertility",
        "label": "Token fertility",
        "higher_is_worse": True,
        "omit_languages": ["zho_Hans"],
        "explanation": (
            "Tokens per whitespace-delimited word after Unicode NFKC normalization "
            "(Petrov et al.). Higher fertility means the same word is broken into more "
            "pieces. Mandarin is omitted from the table and heatmap: it does not use "
            "whitespace between words, so whitespace-delimited fertility is not comparable "
            "to the other languages. Use characters per token and STFR for Mandarin instead."
        ),
    },
    {
        "id": "token_premium",
        "label": "Token premium",
        "higher_is_worse": True,
        "explanation": (
            "Corpus token count for a language divided by the English corpus token count "
            "on the same tokenizer (Arnett et al.). English is 1.0 by construction. "
            "A premium of 2.0 means that language needs twice as many tokens as English "
            "for the same parallel FLORES content—higher cost and less usable context."
        ),
    },
    {
        "id": "chars_per_token",
        "label": "Characters per token",
        "higher_is_worse": False,
        "explanation": (
            "Non-whitespace Unicode characters divided by corpus token count "
            "(Somide, The African Language Tax). Higher means each token carries more "
            "orthographic content. Very low values (below ~1) indicate heavy "
            "character-level shredding."
        ),
    },
    {
        "id": "strr",
        "label": "STRR (Single Token Retention Rate)",
        "higher_is_worse": False,
        "omit_languages": ["zho_Hans"],
        "explanation": (
            "Share of whitespace-delimited words that encode to exactly one token. "
            "Higher STRR means more words stay intact as a single subword. "
            "Mandarin is omitted from the table and heatmap: it does not use whitespace "
            "between words, so whitespace-delimited word counts (and thus STRR) are not "
            "comparable to the other languages. Use characters per token and STFR for Mandarin instead."
        ),
    },
    {
        "id": "stfr",
        "label": "STFR (Single Token Fragmentation Rate)",
        "higher_is_worse": True,
        "explanation": (
            "Share of emitted tokens whose decoded surface form has length 1 "
            "(one character). Higher STFR means more tiny, often semantically empty "
            "fragments. Named STFR to avoid confusion with type–token ratio (TTR)."
        ),
    },
]


def main() -> None:
    rows = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    langs_doc = json.loads(LANGS_PATH.read_text(encoding="utf-8"))
    lang_meta = {L["code"]: L for L in langs_doc["languages"]}

    # Stable language order from artifacts
    language_order = [L["code"] for L in langs_doc["languages"]]
    languages = [
        {
            "code": code,
            "name": lang_meta[code]["name"],
            "continent": lang_meta[code]["continent"],
        }
        for code in language_order
        if any(r["language"] == code for r in rows)
    ]

    tokenizers = [
        {"id": tid, "label": TOKENIZER_LABELS.get(tid, tid)}
        for tid in TOKENIZER_ORDER
        if any(r["tokenizer_id"] == tid for r in rows)
    ]

    payload = {
        "source": "FLORES-200 devtest · NFKC · specials excluded",
        "n_sentences": rows[0]["n_sentences"] if rows else 0,
        "metrics": METRIC_META,
        "languages": languages,
        "tokenizers": tokenizers,
        "rows": rows,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        "window.METRICS_DATA = "
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH} ({len(rows)} rows, {len(languages)} langs)")


if __name__ == "__main__":
    main()
