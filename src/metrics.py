"""Corpus metrics: CTC, token premium, fertility, chars/token, STRR, STFR."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from .load_flores import CJK_CODES, REFERENCE_LANG
from .tokenizers_registry import EncodeFn, TokenizerSpec


@dataclass
class LangMetrics:
    tokenizer_id: str
    language: str
    n_sentences: int
    ctc: int
    n_words: int
    n_chars: int
    fertility: float
    chars_per_token: float
    fertility_per_char: float | None  # for CJK / flagged langs
    stfr: float
    strr: float | None  # None for Mandarin / CJK
    token_premium: float | None = None  # filled after English CTC known
    is_cjk: bool = False


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def count_words(text: str) -> int:
    """Whitespace-split word count after NFKC."""
    t = normalize(text).strip()
    if not t:
        return 0
    return len(t.split())


def count_chars(text: str) -> int:
    """Unicode code points after NFKC, excluding whitespace characters."""
    t = normalize(text)
    return sum(1 for ch in t if not ch.isspace())


def _surface_len(surface: str) -> int:
    """Length for STFR: strip SentencePiece-style underline markers if alone with content."""
    # Count Unicode code points in the decoded surface (after NFKC of surface).
    s = unicodedata.normalize("NFKC", surface)
    # Ignore pure whitespace surfaces for "length 1" (space-only tokens are not char shreds).
    if not s or s.isspace():
        return 0
    return len(s)


def compute_stfr(spec: TokenizerSpec, sentences: Sequence[str]) -> tuple[int, int]:
    """Return (length1_token_count, ctc)."""
    length1 = 0
    ctc = 0
    for s in sentences:
        ids = spec.encode(s)
        ctc += len(ids)
        for tid in ids:
            if _surface_len(spec.surface(tid)) == 1:
                length1 += 1
    return length1, ctc


def compute_strr(spec: TokenizerSpec, sentences: Sequence[str]) -> tuple[int, int]:
    """Return (single_token_words, total_words) using isolated word encodes."""
    single = 0
    total = 0
    for s in sentences:
        words = normalize(s).strip().split()
        for i, word in enumerate(words):
            if not word:
                continue
            total += 1
            if spec.leading_space_for_words:
                piece = " " + word
            else:
                piece = word
            ids = spec.encode(piece)
            if len(ids) == 1:
                single += 1
    return single, total


def encode_corpus(encode: EncodeFn, sentences: Sequence[str]) -> tuple[int, int, int]:
    """Return (CTC, total_words, total_non_whitespace_chars)."""
    ctc = 0
    words = 0
    chars = 0
    for s in sentences:
        ids = encode(s)
        ctc += len(ids)
        words += count_words(s)
        chars += count_chars(s)
    return ctc, words, chars


def metrics_for_language(
    spec: TokenizerSpec,
    language: str,
    sentences: Sequence[str],
) -> LangMetrics:
    is_cjk = language in CJK_CODES

    length1, ctc = compute_stfr(spec, sentences)
    words = sum(count_words(s) for s in sentences)
    chars = sum(count_chars(s) for s in sentences)

    fertility = (ctc / words) if words > 0 else float("nan")
    cpt = (chars / ctc) if ctc > 0 else float("nan")
    fert_char = (ctc / chars) if chars > 0 else None
    stfr = (length1 / ctc) if ctc > 0 else float("nan")

    if is_cjk:
        strr: float | None = None
    else:
        single, wtot = compute_strr(spec, sentences)
        strr = (single / wtot) if wtot > 0 else float("nan")

    return LangMetrics(
        tokenizer_id=spec.id,
        language=language,
        n_sentences=len(sentences),
        ctc=ctc,
        n_words=words,
        n_chars=chars,
        fertility=fertility,
        chars_per_token=cpt,
        fertility_per_char=fert_char if is_cjk else None,
        stfr=stfr,
        strr=strr,
        is_cjk=is_cjk,
    )


def attach_token_premiums(rows: Iterable[LangMetrics]) -> List[LangMetrics]:
    """Set token_premium = CTC(lang) / CTC(eng) per tokenizer (Petrov-style)."""
    by_tok: Dict[str, List[LangMetrics]] = {}
    for r in rows:
        by_tok.setdefault(r.tokenizer_id, []).append(r)

    out: List[LangMetrics] = []
    for tid, group in by_tok.items():
        eng = next((g for g in group if g.language == REFERENCE_LANG), None)
        if eng is None or eng.ctc == 0:
            raise ValueError(f"Missing English CTC for tokenizer {tid}")
        for g in group:
            g.token_premium = g.ctc / eng.ctc
            out.append(g)
    return out


def evaluate_decision_rule(rows: Sequence[LangMetrics]) -> dict:
    """Go/no-go rule from the plan.

    Worth pursuing if:
      (A) some non-English lang has token_premium >= 2.0 on >=2 frontier tokenizers, OR
      (B) mean fertility for a non-Latin/complex-morphology language is >=2x English
          on the same tokenizer.
    """
    frontier = {"o200k", "glm", "llama", "qwen"}
    complex_langs = {
        "amh_Ethi",
        "ory_Orya",
        "zho_Hans",
        "arz_Arab",
        "ary_Arab",
        "ukr_Cyrl",
        "hun_Latn",
        "grn_Latn",
        "quy_Latn",
        "hau_Latn",
        "swh_Latn",
    }

    premium_hits: Dict[str, List[str]] = {}
    for r in rows:
        if r.language == REFERENCE_LANG:
            continue
        if r.tokenizer_id not in frontier:
            continue
        if r.token_premium is not None and r.token_premium >= 2.0:
            premium_hits.setdefault(r.language, []).append(r.tokenizer_id)

    clause_a_langs = {lang: toks for lang, toks in premium_hits.items() if len(toks) >= 2}
    clause_a = len(clause_a_langs) > 0

    clause_b_hits: List[dict] = []
    by_tok: Dict[str, Dict[str, LangMetrics]] = {}
    for r in rows:
        by_tok.setdefault(r.tokenizer_id, {})[r.language] = r

    for tid, langs in by_tok.items():
        eng = langs.get(REFERENCE_LANG)
        if eng is None:
            continue
        for lang, m in langs.items():
            if lang == REFERENCE_LANG or lang not in complex_langs:
                continue
            if m.is_cjk:
                eng_rate = eng.ctc / eng.n_chars if eng.n_chars else float("nan")
                lang_rate = m.ctc / m.n_chars if m.n_chars else float("nan")
                ratio = lang_rate / eng_rate if eng_rate else float("nan")
                metric_name = "tokens_per_char_ratio_vs_en"
            else:
                ratio = m.fertility / eng.fertility if eng.fertility else float("nan")
                metric_name = "fertility_ratio_vs_en"
            if ratio >= 2.0:
                clause_b_hits.append(
                    {
                        "tokenizer": tid,
                        "language": lang,
                        "metric": metric_name,
                        "ratio": ratio,
                    }
                )

    clause_b = len(clause_b_hits) > 0
    worth = clause_a or clause_b

    return {
        "worth_pursuing": worth,
        "clause_a_premium_ge_2_on_ge_2_frontier": clause_a,
        "clause_a_languages": clause_a_langs,
        "clause_b_fertility_ge_2x_en": clause_b,
        "clause_b_hits": clause_b_hits,
    }


def gini_token_inequality(costs: Sequence[float]) -> float:
    """Gini coefficient of per-language token costs (Meister / Foroutan et al.).

    costs should be comparable units (e.g. tokens-per-line on a parallel corpus).
    Values near 0 = equal cost across languages; near 1 = max inequality.
    """
    c = sorted(float(x) for x in costs)
    n = len(c)
    if n == 0:
        return float("nan")
    total = sum(c)
    if total <= 0:
        return float("nan")
    # Eq. 13: (1/n) * (n + 1 - 2 * sum_i (n+1-i) c_i / sum c)
    weighted = sum((n + 1 - i) * c[i - 1] for i in range(1, n + 1))
    return (1.0 / n) * (n + 1.0 - 2.0 * weighted / total)


def tokens_per_line(row: LangMetrics) -> float:
    """Parallel-corpus cost proxy: CTC / n_sentences."""
    if row.n_sentences <= 0:
        return float("nan")
    return row.ctc / row.n_sentences


def gini_for_tokenizer(rows: Sequence[LangMetrics], tokenizer_id: str) -> float:
    costs = [
        tokens_per_line(r)
        for r in rows
        if r.tokenizer_id == tokenizer_id and r.n_sentences > 0
    ]
    return gini_token_inequality(costs)


def rows_to_dicts(rows: Sequence[LangMetrics]) -> List[dict]:
    return [asdict(r) for r in rows]
