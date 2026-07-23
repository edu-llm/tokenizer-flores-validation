"""Deterministic byte- and grapheme-seeded BPE trainer (o200k pretok recipe)."""

from __future__ import annotations

import base64
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Literal

import regex as re
import tiktoken

Unit = Literal["byte", "grapheme", "grapheme_constrained"]

# Token = (bytes, left_on_grapheme_boundary, right_on_grapheme_boundary)
GCToken = tuple[bytes, bool, bool]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def o200k_pattern() -> re.Pattern[str]:
    pat_str = tiktoken.get_encoding("o200k_base")._pat_str
    return re.compile(pat_str)


def pretokenize(text: str, pat: re.Pattern[str] | None = None) -> list[str]:
    if pat is None:
        pat = o200k_pattern()
    return pat.findall(normalize(text))


def pretoken_atoms(pretok: str, unit: Unit) -> tuple[bytes, ...]:
    if unit == "byte":
        return tuple(bytes([b]) for b in pretok.encode("utf-8"))
    return tuple(g.encode("utf-8") for g in re.findall(r"\X", pretok, flags=re.VERSION1))


def pretoken_gconstr_word(pretok: str) -> tuple[GCToken, ...]:
    """Byte atoms with grapheme-boundary flags for constrained training."""
    raw = pretok.encode("utf-8")
    left_bnds: list[bool] = []
    right_bnds: list[bool] = []
    for grapheme in re.findall(r"\X", pretok, flags=re.VERSION1):
        gb = grapheme.encode("utf-8")
        n = len(gb)
        for i in range(n):
            left_bnds.append(i == 0)
            right_bnds.append(i == n - 1)
    if len(left_bnds) != len(raw):
        raise ValueError("grapheme/byte length mismatch in pretoken")
    return tuple(
        (bytes([b]), left_bnds[i], right_bnds[i]) for i, b in enumerate(raw)
    )


def merge_allowed(left: GCToken, right: GCToken) -> bool:
    """Allow within-grapheme completion or whole-grapheme cross-boundary merges."""
    _, a_lb, a_rb = left
    _, b_lb, b_rb = right
    if not a_rb and not b_lb:
        return True
    if a_rb and b_lb:
        return a_lb and b_rb
    return False


def build_word_freqs(
    texts: list[str],
    unit: Unit,
    pat: re.Pattern[str] | None = None,
) -> Counter[tuple[bytes, ...]]:
    if pat is None:
        pat = o200k_pattern()
    freqs: Counter[tuple[bytes, ...]] = Counter()
    for text in texts:
        for pretok in pretokenize(text, pat):
            if pretok:
                freqs[pretoken_atoms(pretok, unit)] += 1
    return freqs


def build_gconstr_word_freqs(
    texts: list[str],
    pat: re.Pattern[str] | None = None,
) -> Counter[tuple[GCToken, ...]]:
    if pat is None:
        pat = o200k_pattern()
    freqs: Counter[tuple[GCToken, ...]] = Counter()
    for text in texts:
        for pretok in pretokenize(text, pat):
            if pretok:
                freqs[pretoken_gconstr_word(pretok)] += 1
    return freqs


def initial_vocab(word_freqs: Counter[tuple[bytes, ...]], unit: Unit) -> dict[bytes, int]:
    tokens: set[bytes] = set()
    for word in word_freqs:
        tokens.update(word)
    # Both arms seed the 256 single-byte alphabet (byte arm atoms; grapheme fallback).
    tokens.update(bytes([i]) for i in range(256))
    ordered = sorted(tokens)
    return {tok: idx for idx, tok in enumerate(ordered)}


def initial_gconstr_vocab(word_freqs: Counter[tuple[GCToken, ...]]) -> dict[bytes, int]:
    tokens: set[bytes] = set()
    for word in word_freqs:
        for tok_bytes, _, _ in word:
            tokens.add(tok_bytes)
    tokens.update(bytes([i]) for i in range(256))
    ordered = sorted(tokens)
    return {tok: idx for idx, tok in enumerate(ordered)}


def word_pairs(word: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    return [(word[i], word[i + 1]) for i in range(len(word) - 1)]


def gconstr_word_pairs(word: tuple[GCToken, ...]) -> list[tuple[bytes, bytes]]:
    pairs: list[tuple[bytes, bytes]] = []
    for i in range(len(word) - 1):
        left, right = word[i], word[i + 1]
        if merge_allowed(left, right):
            pairs.append((left[0], right[0]))
    return pairs


def merge_pair_in_word(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    merged: bytes,
) -> tuple[bytes, ...]:
    left, right = pair
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def merge_pair_in_gconstr_word(
    word: tuple[GCToken, ...],
    pair: tuple[bytes, bytes],
    merged: bytes,
) -> tuple[GCToken, ...]:
    left_b, right_b = pair
    out: list[GCToken] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1:
            tok_a, tok_b = word[i], word[i + 1]
            if (
                tok_a[0] == left_b
                and tok_b[0] == right_b
                and merge_allowed(tok_a, tok_b)
            ):
                out.append((merged, tok_a[1], tok_b[2]))
                i += 2
                continue
        out.append(word[i])
        i += 1
    return tuple(out)


def best_pair(counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes] | None:
    if not counts:
        return None
    max_count = max(counts.values())
    if max_count <= 0:
        return None
    candidates = [pair for pair, count in counts.items() if count == max_count]
    return min(candidates)


def _remove_word(
    word: tuple[bytes, ...],
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for pair in word_pairs(word):
        pair_counts[pair] -= freq
        if pair_counts[pair] <= 0:
            del pair_counts[pair]
        bucket = pair_index.get(pair)
        if bucket is not None:
            bucket.discard(word)
            if not bucket:
                del pair_index[pair]


def _add_word(
    word: tuple[bytes, ...],
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    for pair in word_pairs(word):
        pair_counts[pair] += freq
        pair_index.setdefault(pair, set()).add(word)


def _remove_gconstr_word(
    word: tuple[GCToken, ...],
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_index: dict[tuple[bytes, bytes], set[tuple[GCToken, ...]]],
) -> None:
    for pair in gconstr_word_pairs(word):
        pair_counts[pair] -= freq
        if pair_counts[pair] <= 0:
            del pair_counts[pair]
        bucket = pair_index.get(pair)
        if bucket is not None:
            bucket.discard(word)
            if not bucket:
                del pair_index[pair]


def _add_gconstr_word(
    word: tuple[GCToken, ...],
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_index: dict[tuple[bytes, bytes], set[tuple[GCToken, ...]]],
) -> None:
    for pair in gconstr_word_pairs(word):
        pair_counts[pair] += freq
        pair_index.setdefault(pair, set()).add(word)


def _train_bpe_standard(
    texts: list[str],
    *,
    unit: Literal["byte", "grapheme"],
    target_vocab_size: int,
    pat: re.Pattern[str] | None = None,
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    if pat is None:
        pat = o200k_pattern()

    word_freqs = build_word_freqs(texts, unit, pat)
    vocab = initial_vocab(word_freqs, unit)
    merges: list[tuple[bytes, bytes]] = []

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_index: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for word, freq in word_freqs.items():
        _add_word(word, freq, pair_counts, pair_index)

    while len(vocab) < target_vocab_size:
        pair = best_pair(pair_counts)
        if pair is None:
            break
        left, right = pair
        merged = left + right
        if merged in vocab:
            pair_counts[pair] = 0
            continue

        vocab[merged] = len(vocab)
        merges.append(pair)

        affected = list(pair_index.get(pair, ()))
        for word in affected:
            if word not in word_freqs:
                continue
            freq = word_freqs.pop(word)
            _remove_word(word, freq, pair_counts, pair_index)
            new_word = merge_pair_in_word(word, pair, merged)

            if new_word in word_freqs:
                existing = word_freqs.pop(new_word)
                _remove_word(new_word, existing, pair_counts, pair_index)
                freq += existing

            word_freqs[new_word] = freq
            _add_word(new_word, freq, pair_counts, pair_index)

    return vocab, merges


def _train_bpe_grapheme_constrained(
    texts: list[str],
    *,
    target_vocab_size: int,
    pat: re.Pattern[str] | None = None,
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    if pat is None:
        pat = o200k_pattern()

    word_freqs = build_gconstr_word_freqs(texts, pat)
    vocab = initial_gconstr_vocab(word_freqs)
    merges: list[tuple[bytes, bytes]] = []

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_index: dict[tuple[bytes, bytes], set[tuple[GCToken, ...]]] = {}
    for word, freq in word_freqs.items():
        _add_gconstr_word(word, freq, pair_counts, pair_index)

    while len(vocab) < target_vocab_size:
        pair = best_pair(pair_counts)
        if pair is None:
            break
        left, right = pair
        merged = left + right
        if merged in vocab:
            pair_counts[pair] = 0
            continue

        vocab[merged] = len(vocab)
        merges.append(pair)

        affected = list(pair_index.get(pair, ()))
        for word in affected:
            if word not in word_freqs:
                continue
            freq = word_freqs.pop(word)
            _remove_gconstr_word(word, freq, pair_counts, pair_index)
            new_word = merge_pair_in_gconstr_word(word, pair, merged)

            if new_word in word_freqs:
                existing = word_freqs.pop(new_word)
                _remove_gconstr_word(new_word, existing, pair_counts, pair_index)
                freq += existing

            word_freqs[new_word] = freq
            _add_gconstr_word(new_word, freq, pair_counts, pair_index)

    return vocab, merges


def train_bpe(
    texts: list[str],
    *,
    unit: Unit,
    target_vocab_size: int,
    pat: re.Pattern[str] | None = None,
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]]]:
    """Train BPE on *texts* until *target_vocab_size* tokens (including seed alphabet)."""
    if unit == "grapheme_constrained":
        return _train_bpe_grapheme_constrained(
            texts, target_vocab_size=target_vocab_size, pat=pat
        )
    return _train_bpe_standard(texts, unit=unit, target_vocab_size=target_vocab_size, pat=pat)


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _from_b64(text: str) -> bytes:
    return base64.standard_b64decode(text.encode("ascii"))


def save_artifact(
    out_dir: Path,
    *,
    vocab: dict[bytes, int],
    merges: list[tuple[bytes, bytes]],
    unit: Unit,
    target_vocab_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = _b64(token)

    payload = {
        "unit": unit,
        "target_vocab_size": target_vocab_size,
        "vocab_size": len(vocab),
        "vocab": id_to_token,
        "merges": [[_b64(left), _b64(right)] for left, right in merges],
    }
    (out_dir / "tokenizer.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_artifact(artifact_dir: Path) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]], Unit]:
    data = json.loads((artifact_dir / "tokenizer.json").read_text(encoding="utf-8"))
    unit: Unit = data["unit"]
    vocab: dict[bytes, int] = {}
    for idx, token_b64 in enumerate(data["vocab"]):
        vocab[_from_b64(token_b64)] = idx
    merges = [(_from_b64(left), _from_b64(right)) for left, right in data["merges"]]
    return vocab, merges, unit
