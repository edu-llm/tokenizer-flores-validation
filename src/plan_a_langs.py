"""Language set for Plan A tokenizer training and Plan B model pretraining.

Deliberately separate from :data:`src.load_flores.LANGUAGES` (the locked
12-language efficiency scope) and :data:`src.zipf_langs.STUDY_LANGS` (the
18-language Zipf scope). Those two are published; extending either in place
would shift already-reported numbers. This module defines its own 6-language
set and its own region map rather than extending
``src.load_flores.CONTINENT``.

The set is locked by [plans/02-tokenizer-training.md §3](../plans/02-tokenizer-training.md).
Kept from the superseded 16: ``eng``, ``hun``, ``zho``, ``swh``. Added:
``hin``, ``hat``. All six exist in the local FLORES-200 extraction for both
``dev`` and ``devtest``, so CR-dev is uniformly parallel.

``hat_Latn`` is the binding constraint on corpus size: FineWeb-2 carries only
~0.30 GB of it, 1.87x the scale tier's per-language budget and the thinnest
margin in the set. The byte budgets themselves live in
``configs/benchmarks/tokenizer_local.json``, which is the authority --
duplicating them here would create a second source of truth.

Note the FineWeb-2 config names do not all match the FLORES codes: Mandarin is
``cmn_Hani``, not ``zho_Hans``. English is absent from FineWeb-2 entirely and
comes from ``HuggingFaceFW/fineweb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .load_flores import REFERENCE_LANG

FINEWEB2 = "HuggingFaceFW/fineweb-2"
FINEWEB_EN = "HuggingFaceFW/fineweb"


@dataclass(frozen=True)
class PlanALang:
    code: str
    name: str
    # Coarse region, for grouping in reports only. ``None`` for the premium
    # reference, which is not one of the five regional slots.
    region: Optional[str]
    script: str
    # Upstream dataset id and config. The config is the *upstream* name, which
    # differs from ``code`` for Mandarin.
    dataset_id: str
    dataset_config: str


PLAN_A_LANGS: List[PlanALang] = [
    PlanALang("eng_Latn", "English", None, "Latin", FINEWEB_EN, "sample-10BT"),
    PlanALang("hun_Latn", "Hungarian", "Europe", "Latin", FINEWEB2, "hun_Latn"),
    PlanALang("zho_Hans", "Mandarin", "Asia", "Han", FINEWEB2, "cmn_Hani"),
    PlanALang("hin_Deva", "Hindi", "Asia", "Devanagari", FINEWEB2, "hin_Deva"),
    PlanALang("swh_Latn", "Swahili", "Africa", "Latin", FINEWEB2, "swh_Latn"),
    PlanALang("hat_Latn", "Haitian Creole", "Americas", "Latin", FINEWEB2, "hat_Latn"),
]

PLAN_A_CODES: List[str] = [lang.code for lang in PLAN_A_LANGS]
BY_CODE: Dict[str, PlanALang] = {lang.code: lang for lang in PLAN_A_LANGS}
LANG_NAMES: Dict[str, str] = {lang.code: lang.name for lang in PLAN_A_LANGS}
REGION: Dict[str, Optional[str]] = {lang.code: lang.region for lang in PLAN_A_LANGS}

# Upstream (dataset_id, config) per project code, for the pull scripts.
SOURCES: Dict[str, tuple[str, str]] = {
    lang.code: (lang.dataset_id, lang.dataset_config) for lang in PLAN_A_LANGS
}

# Scripts written without whitespace word segmentation. Mirrors how
# src/metrics.py reports strr=None for CJK; premium is byte-based and
# unaffected.
NO_WORD_BOUNDARY = frozenset({"zho_Hans"})


def assert_reference_present() -> None:
    """Guard: every premium is defined relative to the reference language.

    ``eng_Latn``'s premium must come out to exactly 1.0, so its absence would
    make the whole premium table meaningless rather than merely incomplete.
    """
    if REFERENCE_LANG not in BY_CODE:
        raise ValueError(
            f"Reference language {REFERENCE_LANG!r} missing from PLAN_A_LANGS; "
            "token premiums are undefined without it."
        )


def assert_codes_unique() -> None:
    """Guard: the corpus builder keys per-language budgets by code."""
    if len(PLAN_A_CODES) != len(set(PLAN_A_CODES)):
        raise ValueError(f"Duplicate codes in PLAN_A_LANGS: {PLAN_A_CODES}")


assert_reference_present()
assert_codes_unique()
