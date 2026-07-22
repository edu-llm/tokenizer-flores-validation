"""Load FLORES-200 sentences for the locked language set.

Reads the official ungated FLORES-200 release (plain parallel text, one
sentence per line) from a local extraction of
``https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz``. The HF
`datasets` mirrors (`facebook/flores`, `openlanguagedata/flores_plus`) are
gated and/or script-based, so we avoid them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# Continent → FLORES-200 language codes (script-aware).
LANGUAGES: Dict[str, str] = {
    "swh_Latn": "Swahili",
    "hau_Latn": "Hausa",
    "amh_Ethi": "Amharic",
    "ory_Orya": "Odia",
    "zho_Hans": "Mandarin",
    "arz_Arab": "Egyptian Arabic",
    "ary_Arab": "Moroccan Arabic",
    "hun_Latn": "Hungarian",
    "ukr_Cyrl": "Ukrainian",
    "eng_Latn": "English",
    "quy_Latn": "Quechua",
    "grn_Latn": "Guarani",
}

CONTINENT: Dict[str, str] = {
    "swh_Latn": "Africa",
    "hau_Latn": "Africa",
    "amh_Ethi": "Africa",
    "ory_Orya": "Asia",
    "zho_Hans": "Asia",
    "arz_Arab": "Asia",
    "ary_Arab": "Asia",
    "hun_Latn": "Europe",
    "ukr_Cyrl": "Europe",
    "eng_Latn": "Europe",
    "quy_Latn": "Americas",
    "grn_Latn": "Americas",
}

CJK_CODES = frozenset({"zho_Hans"})
REFERENCE_LANG = "eng_Latn"
SPLIT = "devtest"

# Local extraction of the official FLORES-200 tarball. Override with the
# FLORES200_DIR environment variable if extracted elsewhere.
FLORES_ROOT = Path(os.environ.get("FLORES200_DIR", "data/flores200_dataset"))


def load_flores_sentences(
    lang_codes: List[str] | None = None,
    split: str = SPLIT,
) -> Dict[str, List[str]]:
    """Return parallel sentence lists keyed by FLORES language code.

    Reads ``{FLORES_ROOT}/{split}/{code}.{split}`` (one sentence per line),
    aligned by line index across languages. Languages whose files are absent
    from the base FLORES-200 release are skipped with a warning.
    """
    codes = list(lang_codes or LANGUAGES.keys())
    split_dir = FLORES_ROOT / split
    if not split_dir.is_dir():
        raise RuntimeError(
            f"FLORES split directory not found: {split_dir.resolve()}. "
            "Download and extract "
            "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz "
            "into ./data (or set FLORES200_DIR)."
        )

    out: Dict[str, List[str]] = {}
    n_ref: int | None = None
    missing: List[str] = []

    for code in codes:
        path = split_dir / f"{code}.{split}"
        if not path.is_file():
            missing.append(code)
            continue
        texts = path.read_text(encoding="utf-8").splitlines()
        if n_ref is None:
            n_ref = len(texts)
        elif len(texts) != n_ref:
            raise ValueError(
                f"Parallel length mismatch for {code}: got {len(texts)}, expected {n_ref}"
            )
        out[code] = texts

    if missing:
        print(
            f"[warn] FLORES files not found for {missing} "
            "(not in base FLORES-200 release; needs gated FLORES+). Skipping."
        )
    if not out:
        raise RuntimeError(f"No FLORES languages loaded from {split_dir.resolve()}")
    return out
