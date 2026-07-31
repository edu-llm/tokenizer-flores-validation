"""Language set for the language-specific Zipf deviation study.

Deliberately separate from :data:`src.load_flores.LANGUAGES`. That set is the
locked 12-language scope of the efficiency validation and is consumed by
``src/run_eval.py`` and ``scripts/export_web_data.py``; extending it in place
would shift already-published results. This module defines its own 18-language
set and passes codes explicitly to ``load_flores_sentences(lang_codes=...)``.

The set is the locked 12 plus 6 additions chosen to span the measured
active-vocabulary axis of ``o200k_base`` (0.11% - 5.26% of vocab) across 9
scripts, with two high-resource anchors rather than one.

``has_word_boundary`` records whether whitespace splitting yields usable word
units. Verified empirically on FLORES dev+devtest: every language here averages
15-25 whitespace words per sentence except ``zho_Hans`` (2.0 words/sentence,
20.3 chars/word) and ``tha_Thai`` (4.2, 28.7), which are written without word
spacing. Those two get no word-unit baseline; the grapheme-cluster unit covers
them instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .load_flores import REFERENCE_LANG

# Matched token budget for the size-controlled view. Chosen just under the
# smallest full-corpus token count across the study set (eng_Latn: 52,409 with
# o200k on dev+devtest) so every language can supply it without replacement.
MATCHED_TOKEN_BUDGET = 50_000

# FLORES splits concatenated for the study. dev+devtest = 2,009 parallel
# sentences per language; devtest alone (1,012) is too thin for power-law fits.
SPLITS = ("dev", "devtest")


@dataclass(frozen=True)
class StudyLang:
    code: str
    name: str
    script: str
    # Coarse written-resource tier. Used only for grouping in reports, never as
    # a fitted variable -- the quantitative predictor is measured
    # active-vocabulary share, not this label.
    tier: str
    has_word_boundary: bool
    # True when this language was already in the locked 12-language scope.
    in_locked_12: bool


STUDY_LANGS: List[StudyLang] = [
    # --- high-resource anchors -------------------------------------------------
    StudyLang("eng_Latn", "English", "Latin", "high", True, True),
    StudyLang("spa_Latn", "Spanish", "Latin", "high", True, False),
    StudyLang("zho_Hans", "Mandarin (Simplified)", "Han", "high", False, True),
    StudyLang("kor_Hang", "Korean", "Hangul", "high", True, False),
    # --- mid-resource ---------------------------------------------------------
    StudyLang("hin_Deva", "Hindi", "Devanagari", "mid", True, False),
    StudyLang("ukr_Cyrl", "Ukrainian", "Cyrillic", "mid", True, True),
    StudyLang("hun_Latn", "Hungarian", "Latin", "mid", True, True),
    StudyLang("tha_Thai", "Thai", "Thai", "mid", False, False),
    StudyLang("arz_Arab", "Egyptian Arabic", "Arabic", "mid", True, True),
    StudyLang("swh_Latn", "Swahili", "Latin", "mid", True, True),
    # --- low-resource ---------------------------------------------------------
    StudyLang("ary_Arab", "Moroccan Arabic", "Arabic", "low", True, True),
    StudyLang("hau_Latn", "Hausa", "Latin", "low", True, True),
    StudyLang("quy_Latn", "Quechua (Ayacucho)", "Latin", "low", True, True),
    StudyLang("grn_Latn", "Guarani", "Latin", "low", True, True),
    StudyLang("ory_Orya", "Odia", "Oriya", "low", True, True),
    StudyLang("amh_Ethi", "Amharic", "Ethiopic", "low", True, True),
    StudyLang("tir_Ethi", "Tigrinya", "Ethiopic", "low", True, False),
    StudyLang("sat_Olck", "Santali", "Ol Chiki", "low", True, False),
]

STUDY_CODES: List[str] = [lang.code for lang in STUDY_LANGS]
BY_CODE: Dict[str, StudyLang] = {lang.code: lang for lang in STUDY_LANGS}

# English is the control anchor, inherited from the efficiency scope so the two
# experiments report premiums and deltas against the same reference.
CONTROL_LANG = REFERENCE_LANG

# Languages with no whitespace word segmentation -- word-unit fits are recorded
# as None for these, mirroring how src/metrics.py sets strr=None for CJK.
NO_WORD_BOUNDARY = frozenset(
    lang.code for lang in STUDY_LANGS if not lang.has_word_boundary
)

# Tokenizers for the study: o200k is primary, llama/qwen are frontier
# comparators establishing whether an effect is o200k-specific, and multi
# (NLLB-200) is the multilingual-by-design control. Ids match
# src/tokenizers_registry.py:load_tokenizers.
STUDY_TOKENIZERS: List[str] = ["o200k", "llama", "qwen", "multi"]
PRIMARY_TOKENIZER = "o200k"

# Unit types the Zipf fit is computed over.
UNITS: List[str] = ["token", "word", "grapheme"]


def assert_control_present() -> None:
    """Guard: several metrics are defined relative to the control language."""
    if CONTROL_LANG not in BY_CODE:
        raise ValueError(
            f"Control language {CONTROL_LANG!r} missing from STUDY_LANGS; "
            "token premium and exclusivity metrics are undefined without it."
        )


assert_control_present()
