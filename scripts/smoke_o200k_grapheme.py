"""Smoke tests for o200k grapheme-healed wrap."""

from __future__ import annotations

import unicodedata

import tiktoken

from src.grapheme_wrap import make_o200k_grapheme_encode
from src.tokenizers_registry import load_tokenizers


def test_english_space_prefixed_tokens_preserved() -> None:
    """Whitespace-only wraps break `` world``; heal wrap must not."""
    raw = tiktoken.get_encoding("o200k_base")
    wrap = make_o200k_grapheme_encode()
    for text in [
        "Hello world",
        "Hello, world!",
        "The quick brown fox",
        "OpenAI o200k tokenizer",
    ]:
        assert wrap(text) == raw.encode(unicodedata.normalize("NFKC", text)), text


def test_registry_loads_o200k_grapheme() -> None:
    toks = load_tokenizers(include=["o200k", "o200k_grapheme"])
    assert set(toks) == {"o200k", "o200k_grapheme"}
    text = "Hello world"
    assert toks["o200k_grapheme"].encode(text) == toks["o200k"].encode(text)


def test_mark_heavy_scripts_roundtrip_len() -> None:
    raw = tiktoken.get_encoding("o200k_base")
    wrap = make_o200k_grapheme_encode()
    samples = [
        "ନମସ୍କାର ବିଶ୍ୱ",
        "ሰላም ዓለም",
        "مَرْحَبًا بِالْعَالَمِ",
        "ภาษาไทยทดสอบ",
        "क्षत्रिय",
    ]
    for text in samples:
        norm = unicodedata.normalize("NFKC", text)
        # Heal is a no-op when o200k regex already respects graphemes.
        assert wrap(text) == raw.encode(norm), text


if __name__ == "__main__":
    test_english_space_prefixed_tokens_preserved()
    test_registry_loads_o200k_grapheme()
    test_mark_heavy_scripts_roundtrip_len()
    print("ok")
