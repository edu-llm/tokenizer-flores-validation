"""Encoder and registry-facing loader for trained BPE artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import regex as re

from .bpe_train import load_artifact, o200k_pattern, pretokenize
from .tokenizers_registry import TokenizerSpec

Unit = Literal["byte", "grapheme"]


class BPEEncoder:
    def __init__(
        self,
        vocab: dict[bytes, int],
        merges: list[tuple[bytes, bytes]],
        unit: Unit,
    ) -> None:
        self.vocab = vocab
        self.unit = unit
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.id_to_token = {idx: token for token, idx in vocab.items()}
        self.pat = o200k_pattern()

    @classmethod
    def from_artifact(cls, artifact_dir: Path | str) -> BPEEncoder:
        vocab, merges, unit = load_artifact(Path(artifact_dir))
        return cls(vocab, merges, unit)

    def _initial_symbols(self, pretok: str) -> list[bytes]:
        if self.unit == "byte":
            return [bytes([b]) for b in pretok.encode("utf-8")]
        symbols: list[bytes] = []
        for grapheme in re.findall(r"\X", pretok, flags=re.VERSION1):
            gb = grapheme.encode("utf-8")
            if gb in self.vocab:
                symbols.append(gb)
            else:
                symbols.extend(bytes([b]) for b in gb)
        return symbols

    def _merge_symbols(self, symbols: list[bytes]) -> list[bytes]:
        if len(symbols) < 2:
            return symbols
        while True:
            best_rank: int | None = None
            best_i: int | None = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            merged = symbols[best_i] + symbols[best_i + 1]
            symbols = symbols[:best_i] + [merged] + symbols[best_i + 2 :]
        return symbols

    def encode_piece(self, pretok: str) -> list[int]:
        symbols = self._merge_symbols(self._initial_symbols(pretok))
        return [self.vocab[sym] for sym in symbols]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for pretok in pretokenize(text, self.pat):
            if pretok:
                ids.extend(self.encode_piece(pretok))
        return ids

    def surface(self, token_id: int) -> str:
        raw = self.id_to_token[token_id]
        return raw.decode("utf-8", errors="replace")


def load_bpe_spec(
    artifact_dir: Path | str,
    *,
    tokenizer_id: str,
    name: str | None = None,
) -> TokenizerSpec:
    """Load a trained BPE artifact as a TokenizerSpec (registry wiring hook)."""
    artifact_dir = Path(artifact_dir)
    encoder = BPEEncoder.from_artifact(artifact_dir)
    display = name or tokenizer_id
    return TokenizerSpec(
        id=tokenizer_id,
        name=display,
        source=str(artifact_dir.resolve()),
        is_frontier=False,
        encode=encoder.encode,
        surface=encoder.surface,
        leading_space_for_words=True,
    )
