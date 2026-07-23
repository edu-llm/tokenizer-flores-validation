"""Export results/metrics.json into web/data.js for the static viewer."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.json"
LANGS_PATH = ROOT / "artifacts" / "languages.json"
OUT_PATH = ROOT / "web" / "data.js"
WEB_DIR = ROOT / "web"

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

MORPH_EXPLANATION = (
    "Section 1 (Morph) is a from-scratch A/B testing whether making grapheme "
    "clusters atomic improves multilingual tokenization at matched vocab size. "
    "It did not. Seeding the vocabulary with every grapheme cluster consumes the "
    "vocab budget: at 8k, the morph arm learns only 4,165 merges versus 7,744 for "
    "the byte baseline. With fewer learned multi-character merges, single-character "
    "fragmentation rises (STFR up) and single-token word retention falls (STRR "
    "down). Fertility and token premium improve slightly, and all gaps shrink as "
    "vocab grows to 32k, where the seed overhead amortizes. Conclusion: "
    "grapheme-as-atom trades vocab budget and is not a net win at matched size."
)

MORPH_CONSTRAINED_EXPLANATION = (
    "Section 2 (Morph constrained) fixes the vocab-budget problem: it keeps the "
    "byte baseline's 256-byte seed (so the morph-constrained arm learns the same "
    "7,744 merges at 8k as byte) but forbids any merge from splitting a grapheme "
    "cluster. This isolates the grapheme-integrity constraint from the seed tax. "
    "It still does not beat byte on the fragmentation metrics: STFR is slightly "
    "worse at every vocab size, and STRR is marginally better only at 8k (worse at "
    "16k/32k). Constraining merge choices preserves the budget but nudges more "
    "single-character tokens. Conclusion: grapheme integrity, by either mechanism, "
    "is not a net win for these metrics at this scale."
)

EXPERIMENT_SECTIONS = [
    {
        "id": "morph",
        "title": "Section 1: Morph",
        "source": "Morph (grapheme-as-atom) A/B \u00b7 FLORES-200 devtest \u00b7 12 languages \u00b7 matched vocab",
        "metrics_path": ROOT / "artifacts" / "bpe" / "eval_metrics.json",
        "summary_path": ROOT / "artifacts" / "bpe" / "ab_summary_macro.csv",
        "plot_src": ROOT / "artifacts" / "bpe" / "gap_vs_vocab_size.png",
        "plot_web": "gap_vs_vocab_size.png",
        "compare_col": "grapheme",
        "compare_plot_label": "morph BPE",
        "explanation": MORPH_EXPLANATION,
        "arms": [
            ("bpe_byte_8k", "byte 8k", "byte", 8000),
            ("bpe_grapheme_8k", "morph 8k", "grapheme", 8000),
            ("bpe_byte_16k", "byte 16k", "byte", 16000),
            ("bpe_grapheme_16k", "morph 16k", "grapheme", 16000),
            ("bpe_byte_32k", "byte 32k", "byte", 32000),
            ("bpe_grapheme_32k", "morph 32k", "grapheme", 32000),
            ("o200k", "o200k (ref)", "reference", 200000),
        ],
    },
    {
        "id": "morph_constrained",
        "title": "Section 2: Morph constrained",
        "source": "Morph-constrained (byte seed, no grapheme splits) A/B \u00b7 FLORES-200 devtest \u00b7 12 languages",
        "metrics_path": ROOT / "artifacts" / "bpe_constrained" / "eval_metrics.json",
        "summary_path": ROOT / "artifacts" / "bpe_constrained" / "ab_summary_macro.csv",
        "plot_src": ROOT / "artifacts" / "bpe_constrained" / "gap_vs_vocab_size.png",
        "plot_web": "gap_vs_vocab_size_constrained.png",
        "compare_col": "gconstr",
        "compare_plot_label": "morph-constrained BPE",
        "explanation": MORPH_CONSTRAINED_EXPLANATION,
        "arms": [
            ("bpe_byte_8k", "byte 8k", "byte", 8000),
            ("bpe_gconstr_8k", "morph-c 8k", "grapheme_constrained", 8000),
            ("bpe_byte_16k", "byte 16k", "byte", 16000),
            ("bpe_gconstr_16k", "morph-c 16k", "grapheme_constrained", 16000),
            ("bpe_byte_32k", "byte 32k", "byte", 32000),
            ("bpe_gconstr_32k", "morph-c 32k", "grapheme_constrained", 32000),
            ("o200k", "o200k (ref)", "reference", 200000),
        ],
    },
]

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


def load_macro_deltas(path: Path, compare_col: str) -> list[dict]:
    delta_col = f"delta_{compare_col}_minus_byte"
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "vocab_size": int(row["vocab_size"]),
                    "metric": row["metric"],
                    "byte": float(row["byte"]),
                    "compare": float(row[compare_col]),
                    "delta_compare_minus_byte": float(row[delta_col]),
                    "pct_change": float(row["pct_change"]),
                }
            )
    return rows


def plot_gap_vs_vocab(
    macro_deltas: list[dict], out_path: Path, compare_label: str
) -> None:
    import matplotlib.pyplot as plt

    by_metric: dict[str, list[dict]] = {}
    for row in macro_deltas:
        by_metric.setdefault(row["metric"], []).append(row)
    for metric_rows in by_metric.values():
        metric_rows.sort(key=lambda r: r["vocab_size"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, ax, title in (
        ("token_premium", axes[0], "Token premium (macro mean)"),
        ("stfr", axes[1], "STFR (macro mean)"),
    ):
        sub = by_metric[metric]
        xs = [r["vocab_size"] // 1000 for r in sub]
        ax.plot(xs, [r["byte"] for r in sub], marker="o", label="byte BPE")
        ax.plot(xs, [r["compare"] for r in sub], marker="s", label=compare_label)
        ax.set_xlabel("Vocab size (thousands)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"byte vs {compare_label} gap vs vocab size (FLORES devtest)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def resolve_experiment_plot(
    section: dict, macro_deltas: list[dict]
) -> None:
    dst = WEB_DIR / section["plot_web"]
    src = section["plot_src"]
    if src.is_file():
        shutil.copy(src, dst)
        return
    plot_gap_vs_vocab(macro_deltas, dst, section["compare_plot_label"])


def build_experiment(
    section: dict, lang_meta: dict[str, dict], language_order: list[str]
) -> dict | None:
    metrics_path: Path = section["metrics_path"]
    summary_path: Path = section["summary_path"]
    if not metrics_path.is_file() or not summary_path.is_file():
        return None

    exp_rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    arm_ids = {arm_id for arm_id, *_ in section["arms"]}
    exp_rows = [r for r in exp_rows if r["tokenizer_id"] in arm_ids]
    if not exp_rows:
        return None

    macro_deltas = load_macro_deltas(summary_path, section["compare_col"])
    resolve_experiment_plot(section, macro_deltas)

    exp_lang_codes = sorted(
        {r["language"] for r in exp_rows},
        key=lambda code: language_order.index(code) if code in language_order else 999,
    )
    languages = [
        {
            "code": code,
            "name": lang_meta[code]["name"],
            "continent": lang_meta[code]["continent"],
        }
        for code in exp_lang_codes
        if code in lang_meta
    ]

    arms = [
        {
            "id": arm_id,
            "label": label,
            "unit": unit,
            "vocab_size": vocab_size,
        }
        for arm_id, label, unit, vocab_size in section["arms"]
        if any(r["tokenizer_id"] == arm_id for r in exp_rows)
    ]

    return {
        "id": section["id"],
        "title": section["title"],
        "source": section["source"],
        "n_sentences": exp_rows[0]["n_sentences"] if exp_rows else 0,
        "explanation": section["explanation"],
        "plot": section["plot_web"],
        "arms": arms,
        "languages": languages,
        "rows": exp_rows,
        "macro_deltas": macro_deltas,
    }


def main() -> None:
    rows = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    langs_doc = json.loads(LANGS_PATH.read_text(encoding="utf-8"))
    lang_meta = {L["code"]: L for L in langs_doc["languages"]}

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

    payload: dict = {
        "source": "FLORES-200 devtest · NFKC · specials excluded",
        "n_sentences": rows[0]["n_sentences"] if rows else 0,
        "metrics": METRIC_META,
        "languages": languages,
        "tokenizers": tokenizers,
        "rows": rows,
    }

    experiments = []
    for section in EXPERIMENT_SECTIONS:
        exp = build_experiment(section, lang_meta, language_order)
        if exp:
            experiments.append(exp)
    if experiments:
        payload["experiments"] = experiments

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        "window.METRICS_DATA = "
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    exp_note = (
        ", experiments=" + ", ".join(f"{e['id']}({len(e['rows'])})" for e in experiments)
        if experiments
        else ""
    )
    print(f"Wrote {OUT_PATH} ({len(rows)} rows, {len(languages)} langs{exp_note})")


if __name__ == "__main__":
    main()
