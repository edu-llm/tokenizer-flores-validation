"""Load official vocab.json/merges.txt artifacts for CTC / premium measurement."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel


def load_official_bpe_tokenizer(artifact_dir: Path) -> Tokenizer:
    vocab = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
    merges: list[tuple[str, str]] = []
    for line in (artifact_dir / "merges.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        left, right = line.split(" ", 1)
        merges.append((left, right))
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=False))
    # Shared encode path for premium ratios across arms; STAGE1 is used at train time.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


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
