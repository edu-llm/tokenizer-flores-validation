"""Parity-aware BPE with official STAGE1 pretok and HF-compatible artifacts.

The fair-max algorithm lives in ``train_parity_bpe_from_lang_freqs``. This module
locks pretok to the official SuperBPE STAGE1 regex (no NFKC) so the only
scientific difference versus official BPE is merge selection, then exports
``vocab.json`` / ``merges.txt`` / ``meta.json`` for OLMo tokenization.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import regex as re

from src.bpe_train import train_parity_bpe_from_lang_freqs

# Official SuperBPE stage-one whitespace-constrained pretok (no NFKC).
STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Hugging Face byte-level unicode map used by official BPE artifacts."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(2**8):
        if byte not in bs:
            bs.append(byte)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = bytes_to_unicode()


def encode_token_bytes(token: bytes) -> str:
    return "".join(_BYTE_ENCODER[b] for b in token)


def stage1_pattern() -> re.Pattern[str]:
    return re.compile(STAGE1_REGEX)


def pretokenize_stage1(text: str, pat: re.Pattern[str] | None = None) -> list[str]:
    if pat is None:
        pat = stage1_pattern()
    # Official path: no NFKC; keep the raw Unicode string for ByteLevel encoding.
    return [piece for piece in pat.findall(text) if piece]


def build_stage1_lang_freqs(
    by_lang: dict[str, list[str]],
    *,
    pat: re.Pattern[str] | None = None,
) -> dict[str, Counter[tuple[bytes, ...]]]:
    if pat is None:
        pat = stage1_pattern()
    out: dict[str, Counter[tuple[bytes, ...]]] = {}
    for lang, texts in by_lang.items():
        freqs: Counter[tuple[bytes, ...]] = Counter()
        for text in texts:
            for pretok in pretokenize_stage1(text, pat):
                atoms = tuple(bytes([b]) for b in pretok.encode("utf-8"))
                if atoms:
                    freqs[atoms] += 1
        if freqs:
            out[lang] = freqs
    return out


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


def export_official_bpe_artifacts(
    out_dir: Path,
    *,
    vocab: dict[bytes, int],
    merges: list[tuple[bytes, bytes]],
    meta: dict,
) -> None:
    """Write HF/superbpe-compatible vocab.json, merges.txt, and meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    surface_vocab = {encode_token_bytes(token): idx for token, idx in vocab.items()}
    (out_dir / "vocab.json").write_text(
        json.dumps(surface_vocab, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    merge_lines = ["#version: 0.2"]
    for left, right in merges:
        merge_lines.append(f"{encode_token_bytes(left)} {encode_token_bytes(right)}")
    (out_dir / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def train_parity_official(
    *,
    train_by_lang: dict[str, list[str]],
    dev_by_lang: dict[str, list[str]],
    target_vocab_size: int,
    output_dir: Path,
    train_bytes: int,
    train_files: Iterable[str],
) -> dict:
    train_freqs = build_stage1_lang_freqs(train_by_lang)
    dev_freqs = build_stage1_lang_freqs(dev_by_lang)
    shared = sorted(set(train_freqs) & set(dev_freqs))
    if not shared:
        raise ValueError("train and CR-dev share no languages after STAGE1 pretok")
    train_freqs = {lang: train_freqs[lang] for lang in shared}
    dev_freqs = {lang: dev_freqs[lang] for lang in shared}
    dev_n_lines = {lang: len(dev_by_lang[lang]) for lang in shared}

    vocab, merges = train_parity_bpe_from_lang_freqs(
        train_freqs,
        dev_freqs,
        target_vocab_size=target_vocab_size,
        dev_n_lines=dev_n_lines,
    )
    meta = {
        "arm": "parity",
        "pretok": "official_stage1",
        "nfkc": False,
        "total_bytes": train_bytes,
        "train_files": list(train_files),
        "languages": shared,
        "target_vocab_size": target_vocab_size,
        "vocab_size": len(vocab),
        "n_merges": len(merges),
        "merge_selection": "parity_fair_max_worst_cr_dev",
        "fair_max_score": "tokens_per_line",
        "dev_n_lines": dev_n_lines,
    }
    export_official_bpe_artifacts(output_dir, vocab=vocab, merges=merges, meta=meta)
    return meta
