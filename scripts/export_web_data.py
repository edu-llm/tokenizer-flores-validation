"""Export results/metrics.json into web/data.js for the static viewer."""

from __future__ import annotations

import csv
import json
import math
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
    "multi": "NLLB-200 (Unigram)",
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

MORPH_SKEW_EXPLANATION = (
    "Section 3 (Realistic English-skewed) is the strongest test of the "
    "grapheme-integrity hypothesis. Sections 1-2 trained on a balanced 12-language "
    "corpus, which is unrealistic: real tokenizers are trained on English-dominated "
    "data, so tail languages are starved of merge budget and fragment badly. Here "
    "we reweight the training corpus to a realistic skew (English \u224885% of "
    "byte-mass, Mandarin 5%, every other language \u22642.5% down to 0.25%), keep the "
    "byte-seed + no-grapheme-split constraint from Section 2, and evaluate on the "
    "balanced FLORES devtest so tail languages still count equally. The starved "
    "regime is exactly where grapheme integrity should help most. It barely does. "
    "At 16k the constraint gives a small macro win on fertility (-0.50%) and token "
    "premium (-0.46%), driven by tail scripts (Oriya, Amharic, Egyptian/Moroccan "
    "Arabic all improve fertility and premium), but STFR is still slightly worse "
    "(+0.77%). At 8k and 32k the deltas are mixed and tiny, and STFR is worse at "
    "every vocab size. Conclusion: even under realistic English skew, grapheme "
    "integrity is at best a marginal, metric-dependent wash for tail languages, not "
    "the fragmentation fix the hypothesis predicted."
)

PARITY_EXPLANATION = (
    "Section 4 (Parity-aware BPE) is the OpenAI training-time upgrade candidate "
    "from Foroutan, Meister et al. (arXiv:2508.04796). Unlike grapheme integrity "
    "(Sections 1\u20133), which changes seed atoms or merge legality and failed to "
    "move the metrics, Parity-aware BPE keeps the classical byte seed and o200k "
    "pretok recipe but changes *which* merge is chosen: at each step it picks the "
    "currently worst-compressed language on parallel FLORES-dev (fair-max) and "
    "takes that language's most frequent pair, then applies the merge to all "
    "languages. Training uses the same English-skewed mix as Section 3 "
    "(\u224885% English); evaluation is on balanced FLORES-200 devtest. "
    "It works. Gini of tokens-per-line falls 95% at 8k (0.139\u21920.006), 92% at "
    "16k, and 85% at 32k. Macro token premium collapses toward 1.0 "
    "(\u221246% at 8k, \u221239% at 16k, \u221226% at 32k). Fertility, chars/token, STFR, "
    "and STRR all improve at every vocab size, with the largest gains on starved "
    "tail scripts (e.g. Odia fertility 5.73\u21922.65 and Amharic 5.05\u21922.93 at 8k). "
    "Inference is identical to classical BPE \u2014 only the learned merge list "
    "differs \u2014 so this is directly actionable as a next o200k-class training "
    "objective, not a frozen-tokenizer wrap."
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
    {
        "id": "morph_skew",
        "title": "Section 3: Realistic (English-skewed)",
        "source": "English-skewed training (\u224885% English byte-mass) \u00b7 byte vs morph-constrained \u00b7 eval on balanced FLORES-200 devtest \u00b7 12 languages",
        "metrics_path": ROOT / "artifacts" / "bpe_skew" / "eval_metrics.json",
        "summary_path": ROOT / "artifacts" / "bpe_skew" / "ab_summary_macro.csv",
        "plot_src": ROOT / "artifacts" / "bpe_skew" / "gap_vs_vocab_size.png",
        "plot_web": "gap_vs_vocab_size_skew.png",
        "compare_col": "gconstr",
        "compare_plot_label": "morph-constrained BPE (skewed train)",
        "explanation": MORPH_SKEW_EXPLANATION,
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
    {
        "id": "parity",
        "title": "Section 4: Parity-aware BPE",
        "source": "Parity-aware BPE (fair-max) vs classical byte \u00b7 English-skewed train \u00b7 FLORES-dev CR selection \u00b7 eval on balanced FLORES-200 devtest",
        "metrics_path": ROOT / "artifacts" / "bpe_parity" / "eval_metrics.json",
        "summary_path": ROOT / "artifacts" / "bpe_parity" / "ab_summary_macro.csv",
        "plot_src": ROOT / "artifacts" / "bpe_parity" / "gap_vs_vocab_size.png",
        "plot_web": "gap_vs_vocab_size_parity.png",
        "compare_col": "parity",
        "compare_plot_label": "parity-aware BPE",
        "explanation": PARITY_EXPLANATION,
        "arms": [
            ("bpe_byte_8k", "byte 8k", "byte", 8000),
            ("bpe_parity_8k", "parity 8k", "parity", 8000),
            ("bpe_byte_16k", "byte 16k", "byte", 16000),
            ("bpe_parity_16k", "parity 16k", "parity", 16000),
            ("bpe_byte_32k", "byte 32k", "byte", 32000),
            ("bpe_parity_32k", "parity 32k", "parity", 32000),
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


ZIPF_DIR = ROOT / "results" / "zipf"

ZIPF_EXPLANATION = (
    "This section asks a different question from the efficiency metrics: not how "
    "expensive a language is, but whether the token distribution a tokenizer "
    "produces is structurally sound for it. Word frequencies are Zipfian in every "
    "language, so a tokenizer that serves a language well should preserve that "
    "shape. Each language is evaluated against its own active subset of the "
    "vocabulary, not the full 200k. Zipf-Mandelbrot p(r) proportional to (r+b)^-a "
    "is fit by maximum likelihood over the observed support. Because the exponent "
    "and the KS statistics both degrade on a few-hundred-type support - exactly "
    "the regime Amharic, Tigrinya and Santali occupy under o200k - the entropy "
    "family is reported alongside, and effective vocabulary size exp(H) is the "
    "most robust single number.\n\n"
    "The sharpest way to read the result: compare each language's token "
    "distribution against the raw Unicode grapheme clusters of the same text. "
    "For English, tokenizing multiplies effective vocabulary by 56x (23 -> 1,317) "
    "and for Spanish by 51x. For Odia, Tigrinya, Amharic and Santali the ratio "
    "falls BELOW 1 (0.23 to 0.36): o200k's tokens carry fewer effective units "
    "than simply reading the text character by character. For those four "
    "languages the tokenizer is doing less than nothing relative to the trivial "
    "character baseline."
)

ZIPF_VIEW_META = [
    {
        "id": "matched_token",
        "label": "Matched token budget",
        "explanation": (
            "Every language subsampled to an identical unit budget, so "
            "sample-size-sensitive statistics are comparable. Matched statistical "
            "power, but unmatched content: a language with a 3x token premium "
            "covers roughly a third as many sentences."
        ),
    },
    {
        "id": "matched_sentence",
        "label": "Matched sentences (full corpus)",
        "explanation": (
            "All 2,009 parallel FLORES sentences. Matched semantic content, but "
            "unmatched sample size - token counts range from about 52k for "
            "English to 719k for Santali."
        ),
    },
]

ZIPF_METRIC_META = [
    {
        "id": "log_effective_vocab",
        "formula": "ln exp(H) = H, the Shannon entropy in nats over the active support",
        "label": "Log effective vocabulary ln exp(H)",
        "higher_is_worse": False,
        "explanation": (
            "Natural log of exp(H), the number of equally likely units the token "
            "distribution behaves like. The most robust measure here: it stays "
            "meaningful when a power-law exponent does not, and at a matched "
            "budget it is directly comparable across languages. Lower means the "
            "tokenizer has collapsed the language onto fewer effective units."
        ),
    },
    {
        "id": "effective_vocab",
        "formula": "exp(H) where H = -sum_i p_i ln p_i over observed types",
        "label": "Effective vocabulary size exp(H)",
        "higher_is_worse": False,
        "explanation": (
            "exp of the unigram Shannon entropy. Bounded above by the active "
            "support size, with equality only for a uniform distribution."
        ),
    },
    {
        "id": "n_types",
        "formula": "V = count of distinct units observed at this budget",
        "label": "Active support size (distinct units)",
        "higher_is_worse": False,
        "explanation": (
            "Distinct units observed at this budget. Reported beside every fit "
            "because truncation is the dominant effect and must not be "
            "normalized out of view."
        ),
    },
    {
        "id": "alpha_abs_dev_from_1",
        "formula": "|a - 1| where a is the MLE Zipf-Mandelbrot exponent",
        "label": "|alpha - 1| (deviation from Zipf's exponent)",
        "higher_is_worse": True,
        "explanation": (
            "Absolute distance of the fitted Zipf-Mandelbrot exponent from 1, the "
            "value Zipf's law specifies. Derived from alpha so that larger always "
            "means further from the law, in either direction."
        ),
    },
    {
        "id": "ks_zipf",
        "formula": "max_r |F_emp(r) - F_zipf(r)| with F_zipf from a = 1, b = 0",
        "label": "KS distance from Zipf's law",
        "higher_is_worse": True,
        "explanation": (
            "Kolmogorov-Smirnov distance between the empirical rank CDF and pure "
            "Zipf (alpha=1, b=0) over the same support. Note this is measured "
            "over each language's OWN support, so it partly normalizes away the "
            "truncation that is the main effect - it answers 'conditional on its "
            "support, is this Zipf-shaped', not 'how well is this language "
            "served'. Use effective vocabulary for the latter."
        ),
    },
    {
        "id": "ks",
        "formula": "max_r |F_emp(r) - F_fit(r)| with F_fit from the MLE (a, b)",
        "label": "KS distance from best-fit Zipf-Mandelbrot",
        "higher_is_worse": True,
        "explanation": (
            "Goodness of fit to the best-fitting Zipf-Mandelbrot, which is a "
            "different question from distance to Zipf's law. A uniform "
            "distribution is Zipf-Mandelbrot with alpha=0, so its KS here is near "
            "zero while its distance from Zipf's law is large."
        ),
    },
    {
        "id": "entropy_norm",
        "formula": "H / ln V",
        "label": "Normalized entropy H / ln V",
        "higher_is_worse": False,
        "explanation": (
            "Shannon entropy divided by the log of the active support size. 1.0 "
            "means every unit is equally likely; low values mean a few units "
            "dominate."
        ),
    },
    {
        "id": "renyi_efficiency_active",
        "formula": "H_2.5 / ln V, H_a = ln(sum_i p_i^a) / (1 - a)",
        "label": "Renyi efficiency (order 2.5)",
        "higher_is_worse": False,
        "explanation": (
            "Renyi entropy of order 2.5 over ln of the active support size, "
            "following Zouhar et al., Tokenization and the Noiseless Channel, "
            "which reports this order as the best-correlating tokenizer quality "
            "proxy."
        ),
    },
    {
        "id": "alpha",
        "formula": "argmin over (a, b) of a*sum_r n_r ln(r+b) + N ln H",
        "label": "Zipf-Mandelbrot exponent alpha (raw)",
        "higher_is_worse": True,
        "explanation": (
            "The raw fitted exponent, shown for reference. Zipf's law specifies "
            "alpha = 1, so neither direction is uniformly 'worse' - the colour "
            "scale here is not a badness ranking. Use |alpha - 1| for that."
        ),
    },
]

# Step-by-step method shown on the page, so every number on it can be traced
# back to how it was computed. Each entry becomes one collapsible block.
ZIPF_METHOD = [
    {
        "title": "1. Corpus and the unit streams being compared",
        "body": (
            "FLORES-200 dev + devtest concatenated: 2,009 sentences per language, "
            "parallel by sentence index, so every language expresses the same "
            "content. All text is NFKC-normalized first.\n\n"
            "Each sentence is converted into three independent unit streams:\n\n"
            "token — the tokenizer's own output, spec.encode(sentence).\n"
            "word — whitespace-delimited words. Verified usable for 16 of the 18 "
            "languages; zho_Hans averages 2.0 whitespace words per sentence and "
            "tha_Thai 4.2, so neither gets a word baseline.\n"
            "grapheme — UAX #29 extended grapheme clusters via the regex \\X "
            "operator, excluding whitespace. Defined for every script, so this is "
            "the universal reference.\n\n"
            "The word and grapheme streams do not depend on the tokenizer, so they "
            "are built once and reused across all four tokenizers. Units are then "
            "interned to integer ids: every statistic below depends only on the "
            "multiset of counts, never on surface forms."
        ),
    },
    {
        "title": "2. Vocabulary allocation (Stage 1)",
        "body": (
            "Static allocation walks all 199,998 mergeable ranks of o200k_base "
            "(special tokens excluded, they carry no language content) and decodes "
            "each token's bytes as UTF-8.\n\n"
            "If it decodes, the token is bucketed by the Unicode Script property of "
            "its characters: one bucket per script, plus mixed-script and "
            "no-letter. Script=Common, Script=Inherited and Script=Unknown confer "
            "no script identity, so digits, punctuation, modifier letters such as "
            "U+02BB, and unassigned codepoints cannot make a token look "
            "mixed-script.\n\n"
            "If it does not decode, the token is a fragment of a multi-byte "
            "character (1,562 of them). Its leading byte fixes the sequence length "
            "and the high bits of the codepoint, and each present continuation byte "
            "adds six more bits, leaving a window of 64^missing candidate "
            "codepoints. That window is matched against Unicode blocks: certain "
            "when it lies wholly inside one block, best-guess otherwise. This is "
            "the only way Amharic, Tigrinya and Santali allocation is visible at "
            "all, because o200k contains no token holding a complete Ethiopic or "
            "Ol Chiki character.\n\n"
            "Empirical allocation encodes each of the 204 FLORES languages and "
            "records which ids it reaches. share_of_vocab = distinct ids reached / "
            "199,998."
        ),
    },
    {
        "title": "3. Whole-word coverage and byte-fragment mass",
        "body": (
            "whole_word_coverage — collect the set of DISTINCT word types in the "
            "language's corpus, encode each one in isolation (with a leading space "
            "where the tokenizer is space-sensitive), and take the share that come "
            "back as exactly one token. Type-level on purpose: the "
            "occurrence-level sibling, STRR in the production tab, is dominated by "
            "frequent function words and hides how a language's long tail fares.\n\n"
            "share_fragment_mass — of all tokens the language emits, the share "
            "whose ids are partial-UTF-8 fragments. Mass-weighted, not "
            "type-weighted, and the difference matters enormously: Amharic shares "
            "538 of its 676 types with English, which makes type-level exclusivity "
            "read 0.20, but those shared types are punctuation and Latin names "
            "appearing a few times each while its 76 exclusive Ethiopic fragments "
            "carry 88.8% of every token. Exclusivity is therefore reported "
            "mass-weighted too."
        ),
    },
    {
        "title": "4. Fitting Zipf-Mandelbrot",
        "body": (
            "Counts are sorted descending into n_1 >= n_2 >= ... >= n_V with total "
            "N. The model is\n\n"
            "    p(r) = (r + b)^-a / H,   H = sum over r of (r + b)^-a\n\n"
            "fit by maximizing the log-likelihood, i.e. minimizing\n\n"
            "    NLL(a, b) = a * sum_r n_r * ln(r + b) + N * ln H\n\n"
            "over a and b with L-BFGS-B from eight starting points, taking the best "
            "objective. H is computed via logsumexp so extreme exponents cannot "
            "underflow. Pure Zipf is the same fit with b pinned to 0.\n\n"
            "The support size V is taken from the data, never estimated. "
            "Truncation is the tokenizer's allocation for that language showing "
            "up in the data; it is reported, not fitted away.\n\n"
            "OLS on the log-log rank-frequency curve is exposed as "
            "loglog_ols_slope for comparison with older literature only. It is "
            "biased (Clauset, Shalizi & Newman 2009): measured on synthetic draws "
            "with a known exponent, MLE error stays under 0.01 while OLS error runs "
            "0.03 to 0.07."
        ),
    },
    {
        "title": "5. Two different goodness-of-fit questions",
        "body": (
            "ks — Kolmogorov-Smirnov distance between the empirical rank CDF and "
            "the BEST-FIT Zipf-Mandelbrot: max over r of |F_emp(r) - F_fit(r)|. "
            "Answers whether the distribution is power-law shaped at all.\n\n"
            "ks_zipf — the same distance but against PURE ZIPF, a = 1 and b = 0, "
            "over the same support. Answers how far the distribution sits from "
            "Zipf's law.\n\n"
            "These are genuinely different questions, and a uniform distribution "
            "shows why: uniform IS Zipf-Mandelbrot with a = 0, so its ks is near "
            "zero while its ks_zipf is large.\n\n"
            "Both are measured over each language's OWN support, which means they "
            "partly normalize away the truncation that is the main effect here. "
            "Use effective vocabulary and the token-minus-baseline deltas to ask "
            "how well a language is served."
        ),
    },
    {
        "title": "6. Entropy family, which survives a tiny support",
        "body": (
            "a and the KS statistics both degrade when a language has only a few "
            "hundred types, which is exactly the regime Amharic, Tigrinya, Odia and "
            "Santali occupy. These do not:\n\n"
            "H = -sum p ln p over the observed types (nats).\n"
            "entropy_norm = H / ln V, so 1.0 means every unit equally likely.\n"
            "effective_vocab = exp(H) — the number of equally likely units the "
            "distribution behaves like. Bounded above by V, with equality only for "
            "a uniform distribution. This is the most robust single number here, "
            "and at a matched budget it is directly comparable across languages.\n"
            "renyi_efficiency_active = H_a / ln V with Renyi order a = 2.5, "
            "following Zouhar et al., Tokenization and the Noiseless Channel, where "
            "H_a = ln(sum p^a) / (1 - a).\n\n"
            "n_types (V) is reported beside every fit, because truncation is the "
            "dominant effect and must never be normalized out of view."
        ),
    },
    {
        "title": "7. Matched budgets and bootstrap intervals",
        "body": (
            "Raw token counts span 52,409 for English to 719,023 for Santali, so "
            "naive fits across that range are not comparable.\n\n"
            "matched_token view — for each unit type, the budget is 95% of the "
            "smallest corpus any language supplies for that unit (49,788 tokens; "
            "28,684 words; 79,190 graphemes). Slightly under 100% so that even the "
            "smallest language is subsampled and therefore carries sampling "
            "variability like every other. Each draw shuffles sentences and "
            "accumulates them without replacement until the budget is met, "
            "truncating the final sentence so the total is exact.\n\n"
            "matched_sentence view — all 2,009 parallel sentences. The point "
            "estimate is the full-corpus fit; intervals come from resampling "
            "sentences with replacement.\n\n"
            "Both views are reported because they answer different questions and "
            "quietly picking one would mislead: matching tokens unmatches content, "
            "and matching content unmatches sample size.\n\n"
            "200 draws per cell. Each sentence is encoded once into a cached id "
            "array and draws resample from that cache, so the intervals are cheap. "
            "Reported bounds are the 2.5th and 97.5th percentiles across draws; "
            "non-finite draws are excluded and the surviving count reported."
        ),
    },
    {
        "title": "8. Attributing the deviation to the tokenizer",
        "body": (
            "Word frequencies are Zipfian in every language, so a raw token-level "
            "number cannot separate 'this tokenizer serves the language badly' from "
            "'this language has unusual morphology'. The deltas can, because both "
            "units come from the same text at the same sample size, so "
            "sample-size bias largely cancels:\n\n"
            "    delta_metric = metric(token) - metric(baseline)\n\n"
            "against the word baseline where one exists, and against the grapheme "
            "baseline always. Word supports of 7,000 to 16,000 types are broadly "
            "comparable to token supports, so the exponent delta is meaningful "
            "there. Grapheme supports are 100 to 2,500 types, where a and b are "
            "jointly near-degenerate and a is not comparable to a token-scale fit, "
            "so the grapheme comparison uses effective vocabulary instead."
        ),
    },
    {
        "title": "9. Known limitations",
        "body": (
            "2,009 parallel sentences is thin for power-law estimation. Bootstrap "
            "intervals are shown so it is visible when a cross-language gap sits "
            "inside the noise; small gaps should not be over-read.\n\n"
            "Word-level a lands at 0.64 to 0.94, not the textbook 1. At about 29k "
            "words most word types are hapaxes, the observed curve is truncated at "
            "frequency 1, and MLE over that support returns a flatter exponent. "
            "This is a small-corpus property, not an estimator fault: "
            "Spearman(types-per-token, a) = -0.95, a rises toward 1 for all 16 "
            "languages when the full corpus replaces the matched budget, and a "
            "synthetic sweep at true a = 1 reproduces it exactly (0.90 at 30k "
            "draws, 0.95 at 100k, 0.99 at 500k, 1.00 at 2M). Read a against the "
            "same text's own baseline, not against 1.\n\n"
            "Every fit carries an at_bound flag, aggregated per cell as "
            "at_bound_share. A fit sitting on a bound is a truncation of the "
            "optimum rather than an estimate of it. a = 0 is deliberately not "
            "flagged, being the uniform limit and a natural edge of the parameter "
            "space.\n\n"
            "For a language whose active vocabulary is a few hundred byte "
            "fragments this is not a like-for-like exponent comparison with "
            "English. The claim is that the tokenizer collapses the language onto a "
            "support too small to carry a Zipfian distribution — a statement about "
            "representational capacity.\n\n"
            "Script allocation cannot separate languages sharing a script, which is "
            "why Latin-script conclusions cite mass-weighted exclusivity rather "
            "than raw share."
        ),
    },
]

ZIPF_FIGURES = [
    (
        "zipf_word_coverage.png",
        "What share of each language o200k_base actually has vocabulary for. "
        "Left: the percentage of distinct word types that survive as a single "
        "token. Right: the percentage of emitted tokens that are raw "
        "partial-UTF-8 byte fragments.",
    ),
    (
        "zipf_allocation_vs_deviation.png",
        "Vocabulary allocation vs distributional collapse. Point size and colour "
        "show the fraction of the language encoded as raw byte fragments.",
    ),
    (
        "zipf_rank_frequency.png",
        "Token rank-frequency per language on log-log axes, with the fitted "
        "Zipf-Mandelbrot (dashed) and pure Zipf (dotted) drawn over the "
        "empirical curve. Full corpus, so truncation is visible.",
    ),
    (
        "zipf_script_allocation.png",
        "How o200k_base's 199,998 mergeable ranks divide across Unicode scripts.",
    ),
    (
        "zipf_token_vs_baseline.png",
        "Exponent shift between token units and the same text's own word or "
        "grapheme distribution. This is the attribution argument: word "
        "frequencies are Zipfian in every language, so a gap is the tokenizer's "
        "contribution rather than the language's.",
    ),
    (
        "zipf_deviation_heatmap.png",
        "Log effective vocabulary across languages and tokenizers.",
    ),
]


def _clean(value):
    """JSON-safe scalar: NaN and numpy types are not valid JSON."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def build_zipf(lang_meta: dict[str, dict]) -> dict | None:
    """Assemble the Zipf deviation section.

    Deliberately not routed through EXPERIMENT_SECTIONS: that schema is bound to
    the vocab-size A/B shape (arms with vocab sizes, a compare column,
    gap-vs-vocab plots) and a per-language distribution study does not fit it.
    """
    fits_path = ZIPF_DIR / "zipf_fits.csv"
    if not fits_path.is_file():
        return None

    with fits_path.open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        return None

    numeric_fields = {
        m["id"] for m in ZIPF_METRIC_META if m["id"] != "alpha_abs_dev_from_1"
    }
    numeric_fields |= {f"{name}_lo" for name in numeric_fields}
    numeric_fields |= {f"{name}_hi" for name in numeric_fields}

    rows: list[dict] = []
    lang_info: dict[str, dict] = {}
    tokenizer_ids: list[str] = []
    for r in raw:
        if r.get("unit") != "token":
            continue
        entry = {
            "tokenizer_id": r["tokenizer_id"],
            "language": r["language"],
            "view": r["view"],
        }
        for field in numeric_fields:
            if field in r and r[field] != "":
                try:
                    entry[field] = _clean(float(r[field]))
                except ValueError:
                    entry[field] = None
        alpha = entry.get("alpha")
        entry["alpha_abs_dev_from_1"] = None if alpha is None else abs(alpha - 1.0)
        rows.append(entry)

        lang_info.setdefault(
            r["language"],
            {
                "code": r["language"],
                "name": r.get("language_name") or lang_meta.get(r["language"], {}).get("name", r["language"]),
                "script": r.get("script", ""),
                "tier": r.get("tier", ""),
            },
        )
        if r["tokenizer_id"] not in tokenizer_ids:
            tokenizer_ids.append(r["tokenizer_id"])

    if not rows:
        return None

    # Stage 1 per-language allocation, used for the context table and to order
    # languages by how much of the vocabulary they actually reach.
    profile: list[dict] = []
    profile_path = ZIPF_DIR / "lang_vocab_profile.csv"
    if profile_path.is_file():
        with profile_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("in_study", "").lower() != "true":
                    continue
                profile.append(
                    {
                        "code": r["code"],
                        "name": r.get("language_name") or r["code"],
                        "script": r.get("script", ""),
                        "n_tokens": int(float(r["n_tokens"])),
                        "n_types": int(float(r["n_types"])),
                        "share_of_vocab": _clean(float(r["share_of_vocab"])),
                        "share_fragment_mass": _clean(float(r["share_fragment_mass"])),
                        "share_mass_not_in_control": _clean(
                            float(r["share_mass_not_in_control"])
                        ),
                        "whole_word_coverage": (
                            _clean(float(r["whole_word_coverage"]))
                            if r.get("whole_word_coverage")
                            else None
                        ),
                    }
                )
        profile.sort(key=lambda p: -(p["share_of_vocab"] or 0))

    allocation: list[dict] = []
    alloc_path = ZIPF_DIR / "vocab_allocation_by_script.csv"
    if alloc_path.is_file():
        with alloc_path.open(newline="", encoding="utf-8") as f:
            for r in list(csv.DictReader(f))[:16]:
                allocation.append(
                    {
                        "script": r["script"],
                        "n_tokens": int(float(r["n_tokens"])),
                        "share": _clean(float(r["share"])),
                    }
                )

    order = [p["code"] for p in profile] or sorted(lang_info)
    languages = [lang_info[c] for c in order if c in lang_info]
    languages += [lang_info[c] for c in sorted(lang_info) if c not in set(order)]

    figures = []
    for name, caption in ZIPF_FIGURES:
        src = ZIPF_DIR / name
        if src.is_file():
            shutil.copy(src, WEB_DIR / name)
            figures.append({"file": name, "caption": caption})

    meta: dict = {}
    meta_path = ZIPF_DIR / "zipf_fits.json"
    if meta_path.is_file():
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = {
                "budgets": doc.get("budgets", {}),
                "n_bootstrap": doc.get("n_bootstrap"),
                "hypotheses": doc.get("hypotheses", {}),
                "units": doc.get("units", []),
            }
        except json.JSONDecodeError:
            meta = {}

    return {
        "explanation": ZIPF_EXPLANATION,
        "source": "FLORES-200 dev+devtest · 2,009 parallel sentences · NFKC",
        "method": ZIPF_METHOD,
        "metrics": ZIPF_METRIC_META,
        "views": [v for v in ZIPF_VIEW_META if any(r["view"] == v["id"] for r in rows)],
        "languages": languages,
        "tokenizers": [
            {"id": tid, "label": TOKENIZER_LABELS.get(tid, tid)}
            for tid in TOKENIZER_ORDER
            if tid in tokenizer_ids
        ]
        or [{"id": tid, "label": tid} for tid in tokenizer_ids],
        "rows": rows,
        "profile": profile,
        "allocation": allocation,
        "figures": figures,
        **meta,
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

    zipf = build_zipf(lang_meta)
    if zipf:
        payload["zipf"] = zipf

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
    zipf_note = (
        f", zipf={len(zipf['rows'])} rows/{len(zipf['languages'])} langs"
        f"/{len(zipf['figures'])} figures"
        if zipf
        else ""
    )
    print(f"Wrote {OUT_PATH} ({len(rows)} rows, {len(languages)} langs{exp_note}{zipf_note})")


if __name__ == "__main__":
    main()
