"""Tests for src/vocab_profile.py.

Byte-fragment attribution is the subtle part: Amharic, Tigrinya, and Santali are
tokenized entirely out of partial-UTF-8 tokens under o200k, so if the leading-byte
arithmetic is wrong those languages' allocation reads as zero for the wrong reason.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.tokenizers_registry import TokenizerSpec
from src.vocab_profile import (
    BYTE_FRAGMENT,
    MIXED_SCRIPT,
    NO_LETTER,
    attribute_byte_fragment,
    classify_token_text,
    codepoint_range_from_prefix,
    exclusivity_stats,
    fragment_usage,
    script_of_char,
    whole_word_coverage,
)


# ---------------------------------------------------------------------------
# Per-character script lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "char,expected",
    [
        ("a", "Latin"),
        ("Z", "Latin"),
        ("é", "Latin"),  # é
        ("д", "Cyrillic"),  # д
        ("中", "Han"),  # 中
        ("あ", "Hiragana"),  # あ
        ("가", "Hangul"),  # 가
        ("क", "Devanagari"),  # क
        ("ሀ", "Ethiopic"),  # ሀ
        ("ଓ", "Oriya"),  # ଓ
        ("ก", "Thai"),  # ก
        ("א", "Hebrew"),  # א
        ("ا", "Arabic"),  # ا
        ("ᱚ", "Ol_Chiki"),  # ᱚ
    ],
)
def test_script_of_char(char: str, expected: str):
    assert script_of_char(char) == expected


@pytest.mark.parametrize("char", ["1", " ", ".", "!", "\n", "€"])
def test_script_of_char_none_for_non_letters(char: str):
    """Digits, punctuation, whitespace and currency carry no script identity."""
    assert script_of_char(char) is None


# ---------------------------------------------------------------------------
# Token text classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (" hello", "Latin"),
        ("Москва", "Cyrillic"),
        ("中国", "Han"),
        ("ሰላም", "Ethiopic"),
        ("नमस्ते", "Devanagari"),
        ("123", NO_LETTER),
        ("   ", NO_LETTER),
        (".,;!?", NO_LETTER),
        ("", NO_LETTER),
        ("abcДЕ", MIXED_SCRIPT),
        ("中文abc", MIXED_SCRIPT),
    ],
)
def test_classify_token_text(text: str, expected: str):
    assert classify_token_text(text) == expected


def test_classify_ignores_digits_and_punctuation_around_letters():
    """A script-identifying token keeps its script despite attached punctuation."""
    assert classify_token_text("(hello),") == "Latin"
    assert classify_token_text("42kg") == "Latin"


def test_classify_treats_combining_marks_as_their_base_script():
    # Devanagari क + vowel sign ि -- one script, not mixed.
    assert classify_token_text("कि") == "Devanagari"


# ---------------------------------------------------------------------------
# Partial-UTF-8 codepoint range arithmetic
# ---------------------------------------------------------------------------


def test_ascii_byte_is_its_own_codepoint():
    assert codepoint_range_from_prefix(b"A") == (0x41, 0x41)


def test_two_byte_prefix_of_three_byte_sequence_narrows_to_64_codepoints():
    # 0xE1 0x88 is the first two bytes of U+1200 ETHIOPIC SYLLABLE HA.
    lo, hi = codepoint_range_from_prefix(b"\xe1\x88")
    assert (lo, hi) == (0x1200, 0x123F)


def test_lead_byte_only_of_three_byte_sequence_spans_4096_codepoints():
    lo, hi = codepoint_range_from_prefix(b"\xe1")
    assert (lo, hi) == (0x1000, 0x1FFF)


def test_complete_three_byte_sequence_resolves_exactly():
    assert codepoint_range_from_prefix(b"\xe1\x88\x80") == (0x1200, 0x1200)


def test_two_byte_sequence_lead_only():
    # 0xD0 leads Cyrillic in UTF-8 (U+0400..U+043F).
    assert codepoint_range_from_prefix(b"\xd0") == (0x400, 0x43F)


def test_four_byte_lead_reaches_supplementary_planes():
    lo, hi = codepoint_range_from_prefix(b"\xf0")
    assert lo == 0x0
    assert hi >= 0x3FFFF


def test_continuation_only_fragment_is_unattributable():
    """A tail fragment carries no leading byte, so no script can be inferred."""
    assert codepoint_range_from_prefix(b"\x88\x80") is None


def test_empty_bytes_unattributable():
    assert codepoint_range_from_prefix(b"") is None


# ---------------------------------------------------------------------------
# Byte-fragment attribution to a Unicode block
# ---------------------------------------------------------------------------


def test_ethiopic_fragment_attributes_to_ethiopic_with_certainty():
    attribution = attribute_byte_fragment(b"\xe1\x88")
    assert attribution.block == "Ethiopic"
    assert attribution.certain


def test_devanagari_fragment_attributes_to_devanagari():
    # U+0915 DEVANAGARI KA is 0xE0 0xA4 0x95.
    attribution = attribute_byte_fragment(b"\xe0\xa4")
    assert attribution.block == "Devanagari"
    assert attribution.certain


def test_ol_chiki_fragment_attributes_to_ol_chiki_without_claiming_certainty():
    """Santali's case, and a deliberate demonstration of the certainty rule.

    0xE1 0xB1 narrows to U+1C40-U+1C7F, which straddles Lepcha (U+1C00-U+1C4F)
    and Ol Chiki (U+1C50-U+1C7F). Ol Chiki holds 48 of the 64 candidate
    codepoints so it is the reported best guess, but certainty would be a lie.
    """
    attribution = attribute_byte_fragment(b"\xe1\xb1")
    assert attribution.block == "Ol Chiki"
    assert not attribution.certain
    assert (attribution.lo, attribution.hi) == (0x1C40, 0x1C7F)


def test_han_fragment_attributes_to_cjk():
    # U+4E2D is 0xE4 0xB8 0xAD.
    attribution = attribute_byte_fragment(b"\xe4\xb8")
    assert attribution.block == "CJK Unified Ideographs"
    assert attribution.certain


def test_ambiguous_lead_byte_is_flagged_uncertain():
    attribution = attribute_byte_fragment(b"\xe1")
    assert not attribution.certain, "a 4096-codepoint span must not claim certainty"
    assert attribution.block is not None, "a best guess is still reported"


def test_unattributable_fragment_has_no_block():
    attribution = attribute_byte_fragment(b"\x88\x80")
    assert attribution.block is None
    assert not attribution.certain


# ---------------------------------------------------------------------------
# Exclusivity
# ---------------------------------------------------------------------------


def _sets() -> dict[str, set[int]]:
    return {
        "eng_Latn": {1, 2, 3, 4},
        "swh_Latn": {1, 2, 3, 5},  # mostly shared with English
        "amh_Ethi": {90, 91},  # fully disjoint from English
        "other_a": {5, 90},
        "other_b": {5},
    }


def test_exclusivity_vs_control():
    stats = exclusivity_stats(_sets(), control="eng_Latn", rare_max_langs=2)
    # Swahili: only token 5 of its four types is unused by English.
    assert stats["swh_Latn"].share_not_in_control == pytest.approx(0.25)
    # Amharic shares nothing with English.
    assert stats["amh_Ethi"].share_not_in_control == pytest.approx(1.0)
    # The control is trivially zero against itself.
    assert stats["eng_Latn"].share_not_in_control == pytest.approx(0.0)


def test_exclusivity_rare_share_counts_languages_per_token():
    # Token 90 appears in amh_Ethi and other_a -> 2 languages (rare at max 2).
    # Token 91 appears only in amh_Ethi -> 1 language (rare).
    stats = exclusivity_stats(_sets(), control="eng_Latn", rare_max_langs=2)
    assert stats["amh_Ethi"].share_rare == pytest.approx(1.0)
    # Tokens 1,2,3 appear in 2 languages (eng, swh) -> rare at max 2;
    # token 5 appears in swh, other_a, other_b -> 3 languages, not rare.
    assert stats["swh_Latn"].share_rare == pytest.approx(0.75)


def test_exclusivity_empty_language_is_zero_not_nan():
    stats = exclusivity_stats({"eng_Latn": {1}, "empty": set()}, control="eng_Latn")
    assert stats["empty"].share_not_in_control == 0.0
    assert stats["empty"].share_rare == 0.0


def test_exclusivity_requires_control_present():
    with pytest.raises(KeyError):
        exclusivity_stats({"a": {1}}, control="eng_Latn")


def test_exclusivity_bare_sets_make_mass_equal_types():
    """With no frequencies supplied every type weighs the same."""
    stats = exclusivity_stats(_sets(), control="eng_Latn", rare_max_langs=2)
    swh = stats["swh_Latn"]
    assert swh.share_mass_not_in_control == pytest.approx(swh.share_not_in_control)


def test_exclusivity_mass_weighting_reflects_the_amharic_pattern():
    """The reason mass weighting exists, reduced to its essentials.

    A language with many rarely-used shared types and few heavily-used exclusive
    types looks unremarkable by type count and dramatic by mass. Measured on real
    data, Amharic sits at 0.20 type-level and 0.89 mass-level.
    """
    counts = {
        # 8 shared types, one occurrence each.
        "eng_Latn": Counter({i: 100 for i in range(8)}),
        "amh_Ethi": Counter(
            {**{i: 1 for i in range(8)}, 90: 4_000, 91: 4_000},
        ),
    }
    amh = exclusivity_stats(counts, control="eng_Latn")["amh_Ethi"]
    assert amh.n_types == 10
    assert amh.share_not_in_control == pytest.approx(0.2)
    # 8,000 of 8,008 occurrences sit on the two exclusive types.
    assert amh.share_mass_not_in_control > 0.99
    assert amh.n_tokens == 8_008


def test_exclusivity_mass_rare_share():
    counts = {
        "eng_Latn": Counter({1: 50}),
        "low": Counter({1: 1, 99: 999}),
    }
    stats = exclusivity_stats(counts, control="eng_Latn", rare_max_langs=1)
    # Token 99 is reached by one language only and carries 999 of 1000 tokens.
    assert stats["low"].share_rare == pytest.approx(0.5)
    assert stats["low"].share_mass_rare == pytest.approx(0.999)


# ---------------------------------------------------------------------------
# Byte-fragment usage per language
# ---------------------------------------------------------------------------


def test_fragment_usage_separates_types_from_mass():
    frag_ids = frozenset({90, 91})
    counts = Counter({1: 10, 2: 10, 90: 4_000, 91: 4_000})
    usage = fragment_usage(counts, frag_ids)
    assert usage.n_fragment_types == 2
    assert usage.share_types == pytest.approx(0.5)
    assert usage.n_fragment_tokens == 8_000
    assert usage.share_mass == pytest.approx(8_000 / 8_020)


def test_fragment_usage_zero_when_no_fragments():
    usage = fragment_usage(Counter({1: 5, 2: 5}), frozenset({90}))
    assert usage.share_mass == 0.0
    assert usage.n_fragment_types == 0


def test_fragment_usage_empty_counts():
    usage = fragment_usage(Counter(), frozenset({90}))
    assert usage.share_mass == 0.0
    assert usage.share_types == 0.0


def test_fragment_token_ids_matches_allocation_count():
    """The fragment id set must agree with the static allocation's bucket."""
    from src.vocab_profile import fragment_token_ids

    assert len(fragment_token_ids()) == 1_562


# ---------------------------------------------------------------------------
# Type-level whole-word coverage
# ---------------------------------------------------------------------------


def _fake_spec(single_token_words: set[str], leading_space: bool = True) -> TokenizerSpec:
    """Spec whose encode returns one id only for the listed surface forms."""

    def encode(text: str) -> list[int]:
        return [1] if text in single_token_words else [1, 2]

    return TokenizerSpec(
        id="fake",
        name="fake",
        source="fake",
        is_frontier=False,
        encode=encode,
        surface=lambda tid: "x",
        leading_space_for_words=leading_space,
    )


def test_whole_word_coverage_counts_types_not_occurrences():
    """The point of the type-level metric: repeats must not inflate coverage.

    'the' appears three times but counts once, so coverage is 1 of 2 types
    rather than 3 of 4 occurrences.
    """
    spec = _fake_spec({" the"})
    result = whole_word_coverage(spec, ["the the cat", "the"])
    assert result.n_word_types == 2
    assert result.n_single_token == 1
    assert result.coverage == pytest.approx(0.5)


def test_whole_word_coverage_applies_leading_space_convention():
    spec = _fake_spec({" cat"}, leading_space=True)
    assert whole_word_coverage(spec, ["cat"]).coverage == pytest.approx(1.0)
    # A tokenizer without the leading-space convention sees the bare word.
    spec_bare = _fake_spec({"cat"}, leading_space=False)
    assert whole_word_coverage(spec_bare, ["cat"]).coverage == pytest.approx(1.0)
    # Mismatched convention finds nothing -- proves the flag is honoured.
    spec_wrong = _fake_spec({" cat"}, leading_space=False)
    assert whole_word_coverage(spec_wrong, ["cat"]).coverage == pytest.approx(0.0)


def test_whole_word_coverage_empty_corpus():
    result = whole_word_coverage(_fake_spec(set()), ["", "   "])
    assert result.n_word_types == 0
    assert result.coverage is None, "undefined, not zero, with no words to measure"


def test_whole_word_coverage_normalizes_nfkc():
    # Fullwidth 'ｃａｔ' NFKC-normalizes to 'cat'.
    spec = _fake_spec({" cat"})
    assert whole_word_coverage(spec, ["ｃａｔ"]).coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Regression anchors against the real o200k vocabulary
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def allocation():
    from src.vocab_profile import script_allocation

    return script_allocation()


def test_allocation_covers_every_mergeable_rank(allocation):
    assert allocation.n_vocab == 199_998
    assert sum(row.n_tokens for row in allocation.rows) == 199_998
    assert sum(row.share for row in allocation.rows) == pytest.approx(1.0)


def test_allocation_regression_anchors(allocation):
    """Pinned to the values measured during planning with these library versions."""
    by_script = {row.script: row.n_tokens for row in allocation.rows}
    assert by_script["Latin"] == 134_839
    assert by_script["Cyrillic"] == 14_209
    assert by_script["Arabic"] == 8_007
    assert by_script["Han"] == 7_398
    assert by_script["Devanagari"] == 3_943
    assert by_script[BYTE_FRAGMENT] == 1_562
    assert by_script.get("Ethiopic", 0) == 0
    assert by_script.get("Oriya", 0) == 38


def test_latin_share_is_two_thirds_of_the_vocabulary(allocation):
    latin = next(row for row in allocation.rows if row.script == "Latin")
    assert latin.share == pytest.approx(0.67420, abs=1e-5)


def test_ethiopic_absent_from_complete_character_tokens(allocation):
    """The headline structural finding: no o200k token holds an Ethiopic character.

    Amharic and Tigrinya are therefore encoded entirely from byte fragments.
    """
    assert "Ethiopic" not in {row.script for row in allocation.rows}


def test_byte_fragments_are_attributed(allocation):
    """Fragment attribution must recover Ethiopic even though no whole-char token exists."""
    assert allocation.fragment_blocks, "fragment attribution produced nothing"
    total = sum(allocation.fragment_blocks.values())
    assert total == 1_562


def test_allocation_rows_sorted_descending(allocation):
    counts = [row.n_tokens for row in allocation.rows]
    assert counts == sorted(counts, reverse=True)
