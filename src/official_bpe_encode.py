"""Load official vocab.json/merges.txt artifacts for CTC / premium measurement."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE


def load_official_bpe_tokenizer(artifact_dir: Path) -> Tokenizer:
    """Rebuild an arm's tokenizer from its ``vocab.json`` / ``merges.txt``.

    The model is rebuilt from the raw merge list -- the same bytes
    ``scripts/verify_official_tokenizer_pair.py`` checks -- but the
    **pre-tokenizer is taken from the artifact's own ``tokenizer.json``**,
    because encode-time pretokenization must match what the arm was trained
    with and that differs per arm *and* per trainer:

    ==================  ==========================================
    arm / trainer       pre_tokenizer
    ==================  ==========================================
    official BPE        ``Split(STAGE1_REGEX)`` + ``ByteLevel(use_regex=False)``
    official SuperBPE   ``Split(STAGE2_REGEX)`` + ``ByteLevel(use_regex=False)``
    gigatoken BPE       ``Split(STAGE1_REGEX)`` + ``ByteLevel(use_regex=False)``
    gigatoken SuperBPE  ``ByteLevel(use_regex=False)`` (no Split)
    ==================  ==========================================

    Reading it back rather than reconstructing it keeps this self-describing
    and impossible to drift out of sync.

    This previously hardcoded ``ByteLevel(add_prefix_space=False)``, which
    leaves ``use_regex=True`` -- i.e. GPT-2 whitespace splitting -- for
    *both* arms. Superword merges bridge whitespace and BPE cannot merge
    across pretoken boundaries, so under that path no superword could ever
    fire and the SuperBPE arm's premiums collapsed toward the BPE arm's.
    """
    vocab = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
    merges: list[tuple[str, str]] = []
    for line in (artifact_dir / "merges.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        left, right = line.split(" ", 1)
        merges.append((left, right))
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=False))

    saved_path = artifact_dir / "tokenizer.json"
    if not saved_path.is_file():
        raise FileNotFoundError(
            f"{saved_path} is required: it records the pre-tokenizer this arm "
            "was trained with. Encoding with any other pre-tokenizer silently "
            "changes the measured premiums."
        )
    saved = Tokenizer.from_file(str(saved_path))
    if saved.pre_tokenizer is None:
        raise ValueError(
            f"{saved_path} carries no pre_tokenizer; cannot determine the "
            "encode path for this arm."
        )
    tokenizer.pre_tokenizer = saved.pre_tokenizer
    tokenizer.decoder = saved.decoder or ByteLevelDecoder()
    return tokenizer


def load_lang_text_dir(directory: Path) -> dict[str, list[str]]:
    """Load ``{lang}.txt`` or ``{lang}.dev`` files as one language per file."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Language directory not found: {directory}")
    by_lang: dict[str, list[str]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".dev", ".jsonl"}:
            continue
        lang = path.stem
        if path.suffix.lower() == ".jsonl":
            lines: list[str] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                payload = json.loads(raw)
                text = payload.get("text")
                if text:
                    lines.append(str(text))
            by_lang[lang] = lines
        else:
            by_lang[lang] = [
                line.rstrip("\r\n")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    if not by_lang:
        raise ValueError(f"No language text files found in {directory}")
    return by_lang


def corpus_token_count(tokenizer: Tokenizer, sentences: list[str]) -> int:
    total = 0
    for sentence in sentences:
        total += len(tokenizer.encode(sentence).ids)
    return total


def token_premiums_vs_english(
    tokenizer: Tokenizer,
    by_lang: dict[str, list[str]],
    *,
    reference_lang: str = "eng_Latn",
) -> dict[str, float]:
    if reference_lang not in by_lang:
        raise KeyError(f"reference language missing: {reference_lang}")
    eng = corpus_token_count(tokenizer, by_lang[reference_lang])
    if eng <= 0:
        raise ValueError("English CTC must be positive")
    return {
        lang: corpus_token_count(tokenizer, sentences) / eng
        for lang, sentences in by_lang.items()
    }
