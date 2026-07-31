"""Vocabulary allocation profiling for a frozen tokenizer.

Two complementary views, because neither alone answers "what share of the
vocabulary does this language get":

*Static script allocation* walks every mergeable rank and classifies it by the
Unicode script of its characters. This is a property of the tokenizer alone. It
handles non-Latin scripts cleanly but cannot separate languages that share a
script -- all 67% of Latin lands in one bucket.

*Empirical active vocabulary* encodes a language's corpus and records which token
ids it actually reaches, plus how many of those are shared with the control
language. Exclusivity is what distinguishes Swahili from English when both draw
on the same Latin subwords.

Partial-UTF-8 tokens matter disproportionately. o200k_base contains no token
holding a complete Ethiopic character, so Amharic and Tigrinya are built entirely
from byte fragments; attributing those fragments to Unicode blocks via their
leading byte is the only way their allocation shows up at all.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import regex as re
import tiktoken

from .tokenizers_registry import O200K_ENCODING, TokenizerSpec

# Buckets that are not Unicode scripts.
BYTE_FRAGMENT = "<byte_fragment>"
MIXED_SCRIPT = "<mixed_script>"
NO_LETTER = "<no_letter>"
UNKNOWN_SCRIPT = "<unknown_script>"

# Unicode script property values tested per character. Deliberately broad: a
# script missing from this list would be silently folded into UNKNOWN_SCRIPT,
# and the "rows sum to n_vocab" test would still pass while the Latin and mixed
# counts drifted. Ordered roughly by expected frequency for a cheap early exit.
SCRIPT_NAMES: Sequence[str] = (
    "Latin",
    "Cyrillic",
    "Arabic",
    "Han",
    "Devanagari",
    "Hebrew",
    "Hangul",
    "Georgian",
    "Bengali",
    "Armenian",
    "Greek",
    "Thai",
    "Katakana",
    "Hiragana",
    "Malayalam",
    "Gujarati",
    "Kannada",
    "Telugu",
    "Tamil",
    "Gurmukhi",
    "Sinhala",
    "Myanmar",
    "Khmer",
    "Lao",
    "Tibetan",
    "Oriya",
    "Ethiopic",
    "Thaana",
    "Syriac",
    "Mongolian",
    "Cherokee",
    "Canadian_Aboriginal",
    "Tifinagh",
    "Nko",
    "Adlam",
    "Ol_Chiki",
    "Vai",
    "Bamum",
    "Yi",
    "Bopomofo",
    "Coptic",
    "Glagolitic",
    "Gothic",
    "Runic",
    "Ogham",
    "Deseret",
    "Osage",
    "Shavian",
    "Cham",
    "Tai_Le",
    "New_Tai_Lue",
    "Tai_Tham",
    "Tai_Viet",
    "Buginese",
    "Balinese",
    "Javanese",
    "Sundanese",
    "Batak",
    "Rejang",
    "Lepcha",
    "Limbu",
    "Meetei_Mayek",
    "Saurashtra",
    "Kayah_Li",
    "Lisu",
    "Mandaic",
    "Samaritan",
    "Brahmi",
    "Kharoshthi",
    "Phags_Pa",
    "Tagalog",
    "Hanunoo",
    "Buhid",
    "Tagbanwa",
    "Syloti_Nagri",
    "Chakma",
    "Sharada",
    "Takri",
    "Khojki",
    "Khudawadi",
    "Modi",
    "Tirhuta",
    "Siddham",
    "Grantha",
    "Newa",
    "Mongolian",
    "Cuneiform",
    "Egyptian_Hieroglyphs",
    "Anatolian_Hieroglyphs",
    "Linear_A",
    "Linear_B",
    "Cypriot",
    "Carian",
    "Lycian",
    "Lydian",
    "Old_Italic",
    "Old_Persian",
    "Old_Turkic",
    "Old_Hungarian",
    "Avestan",
    "Phoenician",
    "Imperial_Aramaic",
    "Inscriptional_Pahlavi",
    "Inscriptional_Parthian",
    "Psalter_Pahlavi",
    "Nabataean",
    "Palmyrene",
    "Hatran",
    "Elbasan",
    "Caucasian_Albanian",
    "Duployan",
    "Bassa_Vah",
    "Pahawh_Hmong",
    "Miao",
    "Mro",
    "Warang_Citi",
    "Mende_Kikakui",
    "Ahom",
    "Multani",
    "Marchen",
    "Soyombo",
    "Zanabazar_Square",
    "Dogra",
    "Gunjala_Gondi",
    "Masaram_Gondi",
    "Hanifi_Rohingya",
    "Sogdian",
    "Old_Sogdian",
    "Elymaic",
    "Nandinagari",
    "Nyiakeng_Puachue_Hmong",
    "Wancho",
    "Chorasmian",
    "Dives_Akuru",
    "Khitan_Small_Script",
    "Yezidi",
    "Cypro_Minoan",
    "Old_Uyghur",
    "Tangsa",
    "Toto",
    "Vithkuqi",
    "Kawi",
    "Nag_Mundari",
    "Tangut",
    "Nushu",
    "SignWriting",
    "Mahajani",
    "Sora_Sompeng",
    "Osmanya",
    "Medefaidrin",
    "Bhaiksuki",
    "Makasar",
    "Lao",
)

_SCRIPT_PATTERNS = {name: re.compile(r"\p{Script=%s}" % name) for name in dict.fromkeys(SCRIPT_NAMES)}

# Characters that confer no script identity. Script=Common and Script=Inherited
# hold digits, punctuation, whitespace, currency and most symbols; Script=Unknown
# (Zzzz) holds unassigned codepoints.
#
# Membership is decided by the Unicode Script property alone, deliberately *not*
# gated on general category being a letter or mark. Three kinds of character
# would otherwise be misclassified, each verified against o200k_base:
#   - U+02BB MODIFIER LETTER TURNED COMMA (as in "Hawaiʻi") is Lm/Common;
#     treating it as an unknown script made 94 Latin tokens read as mixed.
#   - Roman numerals U+2160.. are Nl, which \p{L} excludes, yet their script is
#     genuinely Latin (5 tokens).
#   - Two tokens pair an unassigned codepoint (U+4E50A, U+7968A) with "app";
#     Script=Unknown must not make those mixed-script either.
_NON_IDENTIFYING = re.compile(
    r"[\p{Script=Common}\p{Script=Inherited}\p{Script=Unknown}]"
)


@lru_cache(maxsize=1 << 18)
def script_of_char(char: str) -> str | None:
    """Unicode script property value of ``char``, or None if it carries none.

    Returns None for Common/Inherited characters so none of them can make a token
    look mixed-script. A character from a script absent from
    :data:`SCRIPT_NAMES` returns :data:`UNKNOWN_SCRIPT`, which surfaces as its own
    allocation row rather than being silently folded into another.
    """
    if _NON_IDENTIFYING.fullmatch(char):
        return None
    for name, pattern in _SCRIPT_PATTERNS.items():
        if pattern.match(char):
            return name
    return UNKNOWN_SCRIPT


def classify_token_text(text: str) -> str:
    """Bucket a decoded token by the script of its script-bearing characters."""
    if not text:
        return NO_LETTER
    # Fast path: every ASCII letter is Latin, so most of the vocabulary is
    # classified without touching the regex engine at all.
    if text.isascii():
        return "Latin" if any(ch.isalpha() for ch in text) else NO_LETTER
    scripts = {s for s in (script_of_char(ch) for ch in text) if s is not None}
    if not scripts:
        return NO_LETTER
    if len(scripts) == 1:
        return next(iter(scripts))
    return MIXED_SCRIPT


# ---------------------------------------------------------------------------
# Partial-UTF-8 attribution
# ---------------------------------------------------------------------------


def codepoint_range_from_prefix(raw: bytes) -> tuple[int, int] | None:
    """Codepoints a (possibly truncated) UTF-8 byte sequence could encode.

    A BPE vocabulary contains tokens that are fragments of multi-byte characters.
    The leading byte fixes the sequence length and the high bits of the
    codepoint; each present continuation byte adds six more bits. Missing
    continuation bytes leave a window of ``64 ** missing`` codepoints.

    Returns None when no leading byte is present (a tail-only fragment, which
    carries no script information whatsoever).
    """
    if not raw:
        return None
    first = raw[0]
    if first < 0x80:
        return (first, first)
    if first < 0xC0:
        # Continuation byte in leading position: this is a tail fragment.
        return None
    if first < 0xE0:
        n_total, bits = 2, first & 0x1F
    elif first < 0xF0:
        n_total, bits = 3, first & 0x0F
    elif first < 0xF8:
        n_total, bits = 4, first & 0x07
    else:
        return None

    have = 1
    for byte in raw[1:n_total]:
        if not 0x80 <= byte < 0xC0:
            break
        bits = (bits << 6) | (byte & 0x3F)
        have += 1

    missing = n_total - have
    lo = bits << (6 * missing)
    hi = lo + (1 << (6 * missing)) - 1
    return (lo, hi)


# Unicode blocks, enough to attribute fragments across every FLORES-200 script.
# Adjacent small blocks are kept separate so a fragment window straddling two of
# them is correctly reported as uncertain rather than confidently mislabelled.
UNICODE_BLOCKS: Sequence[tuple[int, int, str]] = (
    (0x0000, 0x007F, "Basic Latin"),
    (0x0080, 0x00FF, "Latin-1 Supplement"),
    (0x0100, 0x017F, "Latin Extended-A"),
    (0x0180, 0x024F, "Latin Extended-B"),
    (0x0250, 0x02AF, "IPA Extensions"),
    (0x02B0, 0x02FF, "Spacing Modifier Letters"),
    (0x0300, 0x036F, "Combining Diacritical Marks"),
    (0x0370, 0x03FF, "Greek and Coptic"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic Supplement"),
    (0x0780, 0x07BF, "Thaana"),
    (0x07C0, 0x07FF, "NKo"),
    (0x0800, 0x083F, "Samaritan"),
    (0x0840, 0x085F, "Mandaic"),
    (0x0860, 0x086F, "Syriac Supplement"),
    (0x0870, 0x089F, "Arabic Extended-B"),
    (0x08A0, 0x08FF, "Arabic Extended-A"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul Jamo"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x1380, 0x139F, "Ethiopic Supplement"),
    (0x13A0, 0x13FF, "Cherokee"),
    (0x1400, 0x167F, "Unified Canadian Aboriginal Syllabics"),
    (0x1680, 0x169F, "Ogham"),
    (0x16A0, 0x16FF, "Runic"),
    (0x1700, 0x171F, "Tagalog"),
    (0x1720, 0x173F, "Hanunoo"),
    (0x1740, 0x175F, "Buhid"),
    (0x1760, 0x177F, "Tagbanwa"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1800, 0x18AF, "Mongolian"),
    (0x18B0, 0x18FF, "Unified Canadian Aboriginal Syllabics Extended"),
    (0x1900, 0x194F, "Limbu"),
    (0x1950, 0x197F, "Tai Le"),
    (0x1980, 0x19DF, "New Tai Lue"),
    (0x19E0, 0x19FF, "Khmer Symbols"),
    (0x1A00, 0x1A1F, "Buginese"),
    (0x1A20, 0x1AAF, "Tai Tham"),
    (0x1B00, 0x1B7F, "Balinese"),
    (0x1B80, 0x1BBF, "Sundanese"),
    (0x1BC0, 0x1BFF, "Batak"),
    (0x1C00, 0x1C4F, "Lepcha"),
    (0x1C50, 0x1C7F, "Ol Chiki"),
    (0x1C80, 0x1C8F, "Cyrillic Extended-C"),
    (0x1CC0, 0x1CCF, "Sundanese Supplement"),
    (0x1CD0, 0x1CFF, "Vedic Extensions"),
    (0x1D00, 0x1D7F, "Phonetic Extensions"),
    (0x1D80, 0x1DBF, "Phonetic Extensions Supplement"),
    (0x1DC0, 0x1DFF, "Combining Diacritical Marks Supplement"),
    (0x1E00, 0x1EFF, "Latin Extended Additional"),
    (0x1F00, 0x1FFF, "Greek Extended"),
    (0x2000, 0x206F, "General Punctuation"),
    (0x2070, 0x209F, "Superscripts and Subscripts"),
    (0x20A0, 0x20CF, "Currency Symbols"),
    (0x20D0, 0x20FF, "Combining Marks for Symbols"),
    (0x2100, 0x214F, "Letterlike Symbols"),
    (0x2150, 0x218F, "Number Forms"),
    (0x2190, 0x21FF, "Arrows"),
    (0x2200, 0x22FF, "Mathematical Operators"),
    (0x2300, 0x23FF, "Miscellaneous Technical"),
    (0x2400, 0x243F, "Control Pictures"),
    (0x2460, 0x24FF, "Enclosed Alphanumerics"),
    (0x2500, 0x257F, "Box Drawing"),
    (0x2580, 0x259F, "Block Elements"),
    (0x25A0, 0x25FF, "Geometric Shapes"),
    (0x2600, 0x26FF, "Miscellaneous Symbols"),
    (0x2700, 0x27BF, "Dingbats"),
    (0x2800, 0x28FF, "Braille Patterns"),
    (0x2C00, 0x2C5F, "Glagolitic"),
    (0x2C60, 0x2C7F, "Latin Extended-C"),
    (0x2C80, 0x2CFF, "Coptic"),
    (0x2D00, 0x2D2F, "Georgian Supplement"),
    (0x2D30, 0x2D7F, "Tifinagh"),
    (0x2D80, 0x2DDF, "Ethiopic Extended"),
    (0x2DE0, 0x2DFF, "Cyrillic Extended-A"),
    (0x2E00, 0x2E7F, "Supplemental Punctuation"),
    (0x2E80, 0x2EFF, "CJK Radicals Supplement"),
    (0x2F00, 0x2FDF, "Kangxi Radicals"),
    (0x3000, 0x303F, "CJK Symbols and Punctuation"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3100, 0x312F, "Bopomofo"),
    (0x3130, 0x318F, "Hangul Compatibility Jamo"),
    (0x31F0, 0x31FF, "Katakana Phonetic Extensions"),
    (0x3200, 0x32FF, "Enclosed CJK Letters and Months"),
    (0x3300, 0x33FF, "CJK Compatibility"),
    (0x3400, 0x4DBF, "CJK Unified Ideographs Extension A"),
    (0x4DC0, 0x4DFF, "Yijing Hexagram Symbols"),
    (0x4E00, 0x9FFF, "CJK Unified Ideographs"),
    (0xA000, 0xA48F, "Yi Syllables"),
    (0xA490, 0xA4CF, "Yi Radicals"),
    (0xA4D0, 0xA4FF, "Lisu"),
    (0xA500, 0xA63F, "Vai"),
    (0xA640, 0xA69F, "Cyrillic Extended-B"),
    (0xA6A0, 0xA6FF, "Bamum"),
    (0xA700, 0xA71F, "Modifier Tone Letters"),
    (0xA720, 0xA7FF, "Latin Extended-D"),
    (0xA800, 0xA82F, "Syloti Nagri"),
    (0xA840, 0xA87F, "Phags-pa"),
    (0xA880, 0xA8DF, "Saurashtra"),
    (0xA8E0, 0xA8FF, "Devanagari Extended"),
    (0xA900, 0xA92F, "Kayah Li"),
    (0xA930, 0xA95F, "Rejang"),
    (0xA960, 0xA97F, "Hangul Jamo Extended-A"),
    (0xA980, 0xA9DF, "Javanese"),
    (0xAA00, 0xAA5F, "Cham"),
    (0xAA60, 0xAA7F, "Myanmar Extended-A"),
    (0xAA80, 0xAADF, "Tai Viet"),
    (0xABC0, 0xABFF, "Meetei Mayek"),
    (0xAC00, 0xD7AF, "Hangul Syllables"),
    (0xD7B0, 0xD7FF, "Hangul Jamo Extended-B"),
    (0xD800, 0xDFFF, "Surrogates"),
    (0xE000, 0xF8FF, "Private Use Area"),
    (0xF900, 0xFAFF, "CJK Compatibility Ideographs"),
    (0xFB00, 0xFB4F, "Alphabetic Presentation Forms"),
    (0xFB50, 0xFDFF, "Arabic Presentation Forms-A"),
    (0xFE00, 0xFE0F, "Variation Selectors"),
    (0xFE20, 0xFE2F, "Combining Half Marks"),
    (0xFE30, 0xFE4F, "CJK Compatibility Forms"),
    (0xFE70, 0xFEFF, "Arabic Presentation Forms-B"),
    (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms"),
    (0xFFF0, 0xFFFF, "Specials"),
    (0x10000, 0x1007F, "Linear B Syllabary"),
    (0x10280, 0x1029F, "Lycian"),
    (0x102A0, 0x102DF, "Carian"),
    (0x10300, 0x1032F, "Old Italic"),
    (0x10330, 0x1034F, "Gothic"),
    (0x10400, 0x1044F, "Deseret"),
    (0x10450, 0x1047F, "Shavian"),
    (0x10480, 0x104AF, "Osmanya"),
    (0x104B0, 0x104FF, "Osage"),
    (0x10800, 0x1083F, "Cypriot Syllabary"),
    (0x10840, 0x1085F, "Imperial Aramaic"),
    (0x10900, 0x1091F, "Phoenician"),
    (0x10A00, 0x10A5F, "Kharoshthi"),
    (0x10B00, 0x10B3F, "Avestan"),
    (0x10C00, 0x10C4F, "Old Turkic"),
    (0x10C80, 0x10CFF, "Old Hungarian"),
    (0x10E60, 0x10E7F, "Rumi Numeral Symbols"),
    (0x11000, 0x1107F, "Brahmi"),
    (0x11080, 0x110CF, "Kaithi"),
    (0x11100, 0x1114F, "Chakma"),
    (0x11180, 0x111DF, "Sharada"),
    (0x11280, 0x112AF, "Multani"),
    (0x112B0, 0x112FF, "Khudawadi"),
    (0x11300, 0x1137F, "Grantha"),
    (0x11400, 0x1147F, "Newa"),
    (0x11480, 0x114DF, "Tirhuta"),
    (0x11580, 0x115FF, "Siddham"),
    (0x11600, 0x1165F, "Modi"),
    (0x11680, 0x116CF, "Takri"),
    (0x11700, 0x1174F, "Ahom"),
    (0x118A0, 0x118FF, "Warang Citi"),
    (0x11A00, 0x11A4F, "Zanabazar Square"),
    (0x11A50, 0x11AAF, "Soyombo"),
    (0x11AC0, 0x11AFF, "Pau Cin Hau"),
    (0x11C00, 0x11C6F, "Bhaiksuki"),
    (0x11C70, 0x11CBF, "Marchen"),
    (0x11D00, 0x11D5F, "Masaram Gondi"),
    (0x11D60, 0x11DAF, "Gunjala Gondi"),
    (0x11EE0, 0x11EFF, "Makasar"),
    (0x12000, 0x123FF, "Cuneiform"),
    (0x13000, 0x1342F, "Egyptian Hieroglyphs"),
    (0x14400, 0x1467F, "Anatolian Hieroglyphs"),
    (0x16800, 0x16A3F, "Bamum Supplement"),
    (0x16A40, 0x16A6F, "Mro"),
    (0x16AD0, 0x16AFF, "Bassa Vah"),
    (0x16B00, 0x16B8F, "Pahawh Hmong"),
    (0x16F00, 0x16F9F, "Miao"),
    (0x17000, 0x187FF, "Tangut"),
    (0x18800, 0x18AFF, "Tangut Components"),
    (0x1B000, 0x1B0FF, "Kana Supplement"),
    (0x1B100, 0x1B12F, "Kana Extended-A"),
    (0x1B170, 0x1B2FF, "Nushu"),
    (0x1BC00, 0x1BC9F, "Duployan"),
    (0x1D000, 0x1D0FF, "Byzantine Musical Symbols"),
    (0x1D100, 0x1D1FF, "Musical Symbols"),
    (0x1D400, 0x1D7FF, "Mathematical Alphanumeric Symbols"),
    (0x1E000, 0x1E02F, "Combining Glagolitic"),
    (0x1E800, 0x1E8DF, "Mende Kikakui"),
    (0x1E900, 0x1E95F, "Adlam"),
    (0x1EE00, 0x1EEFF, "Arabic Mathematical Symbols"),
    (0x1F000, 0x1F02F, "Mahjong Tiles"),
    (0x1F0A0, 0x1F0FF, "Playing Cards"),
    (0x1F100, 0x1F1FF, "Enclosed Alphanumeric Supplement"),
    (0x1F200, 0x1F2FF, "Enclosed Ideographic Supplement"),
    (0x1F300, 0x1F5FF, "Miscellaneous Symbols and Pictographs"),
    (0x1F600, 0x1F64F, "Emoticons"),
    (0x1F650, 0x1F67F, "Ornamental Dingbats"),
    (0x1F680, 0x1F6FF, "Transport and Map Symbols"),
    (0x1F700, 0x1F77F, "Alchemical Symbols"),
    (0x1F780, 0x1F7FF, "Geometric Shapes Extended"),
    (0x1F900, 0x1F9FF, "Supplemental Symbols and Pictographs"),
    (0x1FA70, 0x1FAFF, "Symbols and Pictographs Extended-A"),
    (0x20000, 0x2A6DF, "CJK Unified Ideographs Extension B"),
    (0x2A700, 0x2B73F, "CJK Unified Ideographs Extension C"),
    (0x2F800, 0x2FA1F, "CJK Compatibility Ideographs Supplement"),
    (0x30000, 0x3134F, "CJK Unified Ideographs Extension G"),
)


@dataclass(frozen=True)
class FragmentAttribution:
    """Best-guess Unicode block for a partial-UTF-8 token.

    ``certain`` is True only when the candidate codepoint window lies wholly
    inside a single known block. A window straddling two blocks still reports the
    block holding most of it, flagged uncertain.
    """

    block: str | None
    certain: bool
    lo: int | None = None
    hi: int | None = None


def attribute_byte_fragment(raw: bytes) -> FragmentAttribution:
    """Attribute a byte-fragment token to a Unicode block via its leading byte."""
    span = codepoint_range_from_prefix(raw)
    if span is None:
        return FragmentAttribution(block=None, certain=False)
    lo, hi = span

    overlaps: List[tuple[int, str, bool]] = []
    for block_lo, block_hi, name in UNICODE_BLOCKS:
        start, end = max(lo, block_lo), min(hi, block_hi)
        if start <= end:
            contains_all = block_lo <= lo and hi <= block_hi
            overlaps.append((end - start + 1, name, contains_all))

    if not overlaps:
        return FragmentAttribution(block=None, certain=False, lo=lo, hi=hi)

    overlaps.sort(reverse=True)
    width, name, contains_all = overlaps[0]
    certain = contains_all and len(overlaps) == 1
    return FragmentAttribution(block=name, certain=certain, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Static script allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationRow:
    script: str
    n_tokens: int
    share: float


@dataclass
class Allocation:
    encoding: str
    n_vocab: int
    rows: List[AllocationRow]
    # Byte-fragment tokens grouped by best-guess Unicode block.
    fragment_blocks: Dict[str, int] = field(default_factory=dict)
    # Subset of the above whose codepoint window sits inside one block.
    fragment_blocks_certain: Dict[str, int] = field(default_factory=dict)
    mixed_combinations: Dict[str, int] = field(default_factory=dict)


def script_allocation(encoding_name: str = O200K_ENCODING) -> Allocation:
    """Classify every mergeable rank of a tiktoken encoding by script.

    Special tokens are excluded: they carry no language content. Counts therefore
    sum to ``len(enc._mergeable_ranks)`` rather than ``enc.n_vocab``.
    """
    enc = tiktoken.get_encoding(encoding_name)
    ranks = enc._mergeable_ranks

    buckets: Counter = Counter()
    fragment_blocks: Counter = Counter()
    fragment_certain: Counter = Counter()
    mixed_combinations: Counter = Counter()

    for raw in ranks:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            buckets[BYTE_FRAGMENT] += 1
            attribution = attribute_byte_fragment(raw)
            label = attribution.block or "<unattributable>"
            fragment_blocks[label] += 1
            if attribution.certain:
                fragment_certain[label] += 1
            continue

        bucket = classify_token_text(text)
        buckets[bucket] += 1
        if bucket == MIXED_SCRIPT:
            combo = "+".join(
                sorted({s for s in (script_of_char(c) for c in text) if s is not None})
            )
            mixed_combinations[combo] += 1

    total = len(ranks)
    rows = [
        AllocationRow(script=script, n_tokens=count, share=count / total)
        for script, count in sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return Allocation(
        encoding=encoding_name,
        n_vocab=total,
        rows=rows,
        fragment_blocks=dict(fragment_blocks.most_common()),
        fragment_blocks_certain=dict(fragment_certain.most_common()),
        mixed_combinations=dict(mixed_combinations.most_common()),
    )


# ---------------------------------------------------------------------------
# Empirical active vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exclusivity:
    """Type-level and mass-level views of how much vocabulary a language owns.

    The two diverge sharply for byte-level-tokenized languages and both are
    needed. Amharic under o200k reaches 676 types, of which 538 are shared with
    English -- so type-level exclusivity reads 0.20, sounding unremarkable. But
    the shared types are punctuation, digits and Latin names appearing a handful
    of times each, while the 76 exclusive Ethiopic byte fragments carry 88.8% of
    all Amharic tokens. Type counting weights a hapax equal to a fragment used
    36,918 times; mass weighting does not.
    """

    code: str
    n_types: int
    n_tokens: int
    n_not_in_control: int
    share_not_in_control: float
    n_rare: int
    share_rare: float
    # Share of token *occurrences* on types the control language never reaches.
    share_mass_not_in_control: float
    share_mass_rare: float


def exclusivity_stats(
    active_counts: Mapping[str, Mapping[int, float] | Counter | Set[int]],
    control: str,
    rare_max_langs: int = 5,
) -> Dict[str, Exclusivity]:
    """How much of each language's active vocabulary is its own.

    Static script allocation cannot separate languages sharing a script, so this
    is the metric Latin-script conclusions must rest on. ``share_rare`` counts
    types reached by at most ``rare_max_langs`` of the supplied languages.

    Accepts frequency mappings (preferred, enables the mass-weighted view) or
    bare id sets, in which case every type is weighted equally and the mass
    figures coincide with the type figures.
    """
    if control not in active_counts:
        raise KeyError(f"control language {control!r} absent from active_counts")

    def as_counts(value: Mapping[int, float] | Counter | Set[int]) -> Dict[int, float]:
        if isinstance(value, (set, frozenset)):
            return {int(t): 1.0 for t in value}
        return {int(t): float(c) for t, c in value.items() if c > 0}

    per_lang = {code: as_counts(value) for code, value in active_counts.items()}
    control_types = set(per_lang[control])

    langs_per_token: Counter = Counter()
    for counts in per_lang.values():
        langs_per_token.update(counts.keys())

    out: Dict[str, Exclusivity] = {}
    for code, counts in per_lang.items():
        n = len(counts)
        if n == 0:
            out[code] = Exclusivity(code, 0, 0, 0, 0.0, 0, 0.0, 0.0, 0.0)
            continue
        total_mass = sum(counts.values())
        new_types = [t for t in counts if t not in control_types]
        rare_types = [t for t in counts if langs_per_token[t] <= rare_max_langs]
        mass_new = sum(counts[t] for t in new_types)
        mass_rare = sum(counts[t] for t in rare_types)
        out[code] = Exclusivity(
            code=code,
            n_types=n,
            n_tokens=int(total_mass),
            n_not_in_control=len(new_types),
            share_not_in_control=len(new_types) / n,
            n_rare=len(rare_types),
            share_rare=len(rare_types) / n,
            share_mass_not_in_control=(mass_new / total_mass) if total_mass else 0.0,
            share_mass_rare=(mass_rare / total_mass) if total_mass else 0.0,
        )
    return out


@lru_cache(maxsize=8)
def fragment_token_ids(encoding_name: str = O200K_ENCODING) -> frozenset[int]:
    """Ids of tokens whose bytes are not valid standalone UTF-8.

    These are the pieces a language falls back to when the vocabulary holds no
    token for its characters.
    """
    enc = tiktoken.get_encoding(encoding_name)
    out = set()
    for raw, rank in enc._mergeable_ranks.items():
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            out.add(int(rank))
    return frozenset(out)


@dataclass(frozen=True)
class FragmentUsage:
    n_fragment_types: int
    n_fragment_tokens: int
    share_types: float
    # The headline Stage 1 number: what fraction of this language is raw bytes.
    share_mass: float


def fragment_usage(counts: Mapping[int, float] | Counter, fragment_ids: frozenset[int]) -> FragmentUsage:
    """How much of a language's encoded output is partial-UTF-8 byte fragments."""
    total_types = len(counts)
    total_mass = sum(counts.values())
    if total_types == 0 or total_mass <= 0:
        return FragmentUsage(0, 0, 0.0, 0.0)
    frag_types = [t for t in counts if int(t) in fragment_ids]
    frag_mass = sum(counts[t] for t in frag_types)
    return FragmentUsage(
        n_fragment_types=len(frag_types),
        n_fragment_tokens=int(frag_mass),
        share_types=len(frag_types) / total_types,
        share_mass=frag_mass / total_mass,
    )


@dataclass(frozen=True)
class WordCoverage:
    n_word_types: int
    n_single_token: int
    # None when the corpus yields no words at all -- undefined, not zero.
    coverage: float | None


def whole_word_coverage(spec: TokenizerSpec, sentences: Iterable[str]) -> WordCoverage:
    """Share of distinct word types that encode to exactly one token.

    Type-level by design. ``src/metrics.py:compute_strr`` is the occurrence-level
    sibling and is left untouched because published results depend on its current
    semantics; frequent short function words would otherwise dominate this
    measure and hide how the long tail of a language's lexicon actually fares.
    """
    word_types: Set[str] = set()
    for sentence in sentences:
        stripped = unicodedata.normalize("NFKC", sentence).strip()
        if stripped:
            word_types.update(stripped.split())

    if not word_types:
        return WordCoverage(n_word_types=0, n_single_token=0, coverage=None)

    single = 0
    for word in word_types:
        piece = f" {word}" if spec.leading_space_for_words else word
        if len(spec.encode(piece)) == 1:
            single += 1
    return WordCoverage(
        n_word_types=len(word_types),
        n_single_token=single,
        coverage=single / len(word_types),
    )


def encode_per_sentence(spec: TokenizerSpec, sentences: Sequence[str]) -> List[List[int]]:
    """Encode once and keep per-sentence id lists.

    Bootstrap draws resample sentences from this cache, so the expensive encode
    happens once per (language, tokenizer) rather than once per draw.
    """
    return [spec.encode(sentence) for sentence in sentences]


def active_vocabulary(per_sentence: Sequence[Sequence[int]]) -> Counter:
    """Token-id frequency table over a per-sentence encode cache."""
    counts: Counter = Counter()
    for ids in per_sentence:
        counts.update(ids)
    return counts
