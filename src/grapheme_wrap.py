"""Grapheme-aware wrap over frozen OpenAI o200k_base.

Preserves o200k's merge table and (when already legal) its regex pretok,
but refuses to cut pretokens inside a Unicode extended grapheme cluster
(UAX #29). Each healed pretok span is encoded with BPE only
(``Encoding._encode_single_piece``), skipping a second regex split.

This is deliberately *not* a whitespace-only pretok: replacing o200k's
regex with whitespace splits destroys space-prefixed tokens (e.g. `` world``)
and inflates English CTC, which can artifactually shrink token premiums.
"""

from __future__ import annotations

import unicodedata
from typing import List, Sequence, Tuple

import regex as re
import tiktoken

O200K_ENCODING = "o200k_base"


def normalize_nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def grapheme_cut_positions(text: str) -> set[int]:
    """Offsets where a UAX #29 extended grapheme cluster may begin/end."""
    cuts = {0, len(text)}
    for match in re.finditer(r"\X", text, flags=re.VERSION1):
        cuts.add(match.start())
        cuts.add(match.end())
    return cuts


def _regex_pretok_spans(text: str, pat: re.Pattern[str]) -> List[Tuple[int, int]]:
    """Return contiguous (start, end) spans covering ``text`` via o200k's regex.

    Raises if the pattern leaves gaps (should not happen for o200k_base).
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    for match in pat.finditer(text):
        if match.start() != pos:
            raise ValueError(
                f"o200k pretok gap at {pos}..{match.start()} in {text!r}"
            )
        spans.append((match.start(), match.end()))
        pos = match.end()
    if pos != len(text):
        raise ValueError(f"o200k pretok did not cover suffix at {pos}..{len(text)}")
    return spans


def heal_spans_to_graphemes(
    spans: Sequence[Tuple[int, int]],
    legal_cuts: set[int],
) -> List[Tuple[int, int]]:
    """Merge adjacent pretok spans until every boundary sits on a grapheme cut."""
    if not spans:
        return []

    healed: List[List[int]] = [[spans[0][0], spans[0][1]]]
    for start, end in spans[1:]:
        # Illegal start ⇒ this piece was cut out of the previous grapheme.
        if start not in legal_cuts:
            healed[-1][1] = end
        else:
            healed.append([start, end])

    # Ensure ends are legal by absorbing following pieces when needed.
    out: List[Tuple[int, int]] = []
    i = 0
    while i < len(healed):
        start, end = healed[i]
        while end not in legal_cuts and i + 1 < len(healed):
            i += 1
            end = healed[i][1]
        if end not in legal_cuts:
            # Extend to the next legal cut at or after end (always exists: len).
            end = min(c for c in legal_cuts if c >= end)
        out.append((start, end))
        i += 1
    return out


def make_o200k_grapheme_encode():
    """Return ``encode(text) -> list[int]`` for o200k + grapheme-healed pretok."""
    enc = tiktoken.get_encoding(O200K_ENCODING)
    pat = re.compile(enc._pat_str)

    def encode(text: str) -> List[int]:
        text = normalize_nfkc(text)
        if not text:
            return []
        legal = grapheme_cut_positions(text)
        raw_spans = _regex_pretok_spans(text, pat)
        spans = heal_spans_to_graphemes(raw_spans, legal)
        ids: List[int] = []
        for start, end in spans:
            # BPE only — do not re-apply o200k regex inside the span.
            ids.extend(enc._encode_single_piece(text[start:end]))
        return ids

    return encode
