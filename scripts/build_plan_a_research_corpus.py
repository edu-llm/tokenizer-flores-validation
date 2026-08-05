#!/usr/bin/env python3
"""Build the equal-byte Plan A tokenizer-training corpus.

Every language contributes **exactly** the same number of bytes. That is the
whole point: the superseded builder concatenated whatever it found in
filesystem order and produced a 6x skew (48 MB Amharic/Swahili against 8 MB
for twelve others), which confounded every cross-language premium with the
mix. See [plans/02-tokenizer-training.md §1.1](../plans/02-tokenizer-training.md).

Two views of the same bytes are written, because the two trainers take
different input shapes and the cross-check requires they see an identical
corpus:

``corpus/langs/{code}.txt``
    One file per language, for the official SuperBPE trainer, which reads a
    ``corpus_dir`` of ``*.txt``. Pass ``--corpus_dir corpus/langs`` and
    ``--num-bytes`` equal to the manifest's ``total_bytes`` so
    ``get_files_with_num_bytes`` takes every file whole and its
    ``random.seed(0)`` shuffle cannot change the mix.

    **These live in their own subdirectory on purpose.** The official trainer
    globs every ``*.txt`` in ``corpus_dir``; leaving ``train.txt`` beside them
    would silently train on the corpus twice over.

``corpus/train.txt``
    The byte-identical concatenation in ``PLAN_A_CODES`` order, for
    gigatoken's ``train_superbpe``, which accepts only in-memory bytes or a
    single mmapped path. Pass ``separator=b"\\n"``: gigatoken strips separator
    bytes and yields one document per line, which is the same unit HuggingFace's
    trainer sees when it reads the per-language files line by line. Using the
    default ``<|endoftext|>`` separator instead would leave the corpus as one
    document, costing all pretokenization parallelism.

Sources are the staged FineWeb-2 / FineWeb pulls, one ``{code}.txt`` per
language. A language that cannot supply its full share is a hard failure --
never a silent shrink, which is how the skew got in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# _fit_utf8 is the project's existing "longest valid UTF-8 prefix within N
# bytes" helper (src/benchmark.py:77); duplicating it here would create a
# second one, which CLAUDE.md forbids for sha256_file and applies equally.
from src.benchmark import _fit_utf8, atomic_write_json
from src.plan_a_langs import PLAN_A_CODES, PLAN_A_LANGS, SOURCES

DEFAULT_SOURCE_DIR = ROOT / "artifacts" / "plan_a" / "raw" / "fineweb2_samples"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "plan_a"
DEFAULT_CONFIG = ROOT / "configs" / "benchmarks" / "tokenizer_local.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tier",
        choices=("smoke", "pilot", "scale"),
        help="Read bytes_per_language from the config's tier table.",
    )
    p.add_argument(
        "--bytes-per-language",
        type=int,
        help="Override the tier budget. Exactly one of --tier / this is required.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args(argv)


def resolve_budget(args: argparse.Namespace) -> tuple[int, str | None]:
    """Per-language byte budget, from an explicit override or the tier table."""
    if (args.tier is None) == (args.bytes_per_language is None):
        raise ValueError("Pass exactly one of --tier or --bytes-per-language")
    if args.bytes_per_language is not None:
        if args.bytes_per_language <= 0:
            raise ValueError("--bytes-per-language must be positive")
        return args.bytes_per_language, None

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tier = config["tiers"][args.tier]
    budget = int(tier["bytes_per_language"])

    # The config carries a derived total; keep the two consistent rather than
    # trusting either alone.
    declared_total = int(tier["corpus_bytes"])
    if declared_total != budget * len(PLAN_A_CODES):
        raise ValueError(
            f"config tier {args.tier!r} is inconsistent: corpus_bytes="
            f"{declared_total:,} but bytes_per_language={budget:,} x "
            f"{len(PLAN_A_CODES)} languages = {budget * len(PLAN_A_CODES):,}"
        )

    ceiling = int(config["max_balanced_corpus_bytes"])
    if declared_total > ceiling:
        raise ValueError(
            f"tier {args.tier!r} needs {declared_total:,} bytes but "
            f"max_balanced_corpus_bytes is {ceiling:,}"
        )
    return budget, args.tier


def build_language_file(source: Path, dest: Path, budget: int) -> dict[str, object]:
    """Write exactly ``budget`` bytes of ``source`` to ``dest``.

    Whole lines are taken while they fit; the final line is cut to the longest
    valid UTF-8 prefix that still fits. That prefix can land 1-3 bytes short
    when the cut falls inside a multi-byte character, so the remainder is
    padded with newlines -- empty lines, immaterial as training text, and the
    only way to make ``max(bytes) == min(bytes)`` hold exactly. Padding is
    recorded per language rather than hidden.
    """
    if not source.is_file():
        raise FileNotFoundError(
            f"No staged source for this language: {source}. Pull it with "
            "scripts/pull_fineweb2_lang_samples.py before building."
        )
    available = source.stat().st_size
    if available < budget:
        raise ValueError(
            f"{source.name} has {available:,} bytes but the budget is "
            f"{budget:,}. Equal-byte corpora must fail here, not shrink: a "
            "short language silently reintroduces the mix skew."
        )

    written = 0
    lines_written = 0
    truncated_line = False
    digest = hashlib.sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8", errors="strict") as src, dest.open("wb") as out:
        for line in src:
            if written >= budget:
                break
            payload = line.rstrip("\r\n").encode("utf-8") + b"\n"
            if written + len(payload) > budget:
                payload = _fit_utf8(payload, budget - written)
                truncated_line = bool(payload)
                if not payload:
                    break
            out.write(payload)
            digest.update(payload)
            written += len(payload)
            lines_written += 1

        padding = budget - written
        # Padding exists only for a 1–3 byte UTF-8 mid-character cut. Anything
        # larger means the source ran dry after CRLF inflation or a shortfall
        # the st_size pre-filter missed — fail hard, do not pad with newlines.
        if padding > 3:
            raise ValueError(
                f"{source.name}: need {padding:,} bytes of padding to reach "
                f"budget {budget:,} (wrote {written:,}). Padding > 3 means the "
                "source ran dry (often CRLF-inflated on-disk size); re-pull "
                "with newline='\\n', do not absorb the shortfall."
            )
        if padding > 0:
            pad = b"\n" * padding
            out.write(pad)
            digest.update(pad)
            written += padding

    if written != budget:
        raise AssertionError(f"{dest.name}: wrote {written:,}, expected {budget:,}")

    return {
        "bytes": written,
        "sha256": digest.hexdigest(),
        "lines": lines_written,
        "final_line_truncated": truncated_line,
        "padding_bytes": padding,
        "source": str(source),
        "source_bytes_available": available,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    budget, tier = resolve_budget(args)

    corpus_dir = args.output_dir / "corpus"
    # Per-language files get their own directory: the official trainer globs
    # *.txt in corpus_dir, so train.txt must not sit beside them or it trains
    # on everything twice.
    langs_dir = corpus_dir / "langs"
    langs_dir.mkdir(parents=True, exist_ok=True)

    per_language: dict[str, dict[str, object]] = {}
    for code in PLAN_A_CODES:
        source = args.source_dir / f"{code}.txt"
        per_language[code] = build_language_file(
            source, langs_dir / f"{code}.txt", budget
        )
    stray = sorted(p.name for p in langs_dir.glob("*.txt") if p.stem not in PLAN_A_CODES)
    if stray:
        raise AssertionError(
            f"Unexpected .txt files in {langs_dir}: {stray}. The official "
            "trainer globs *.txt and would train on them too."
        )

    sizes = {code: int(meta["bytes"]) for code, meta in per_language.items()}
    if max(sizes.values()) != min(sizes.values()):
        raise AssertionError(f"Corpus is not equal-byte: {sizes}")

    # train.txt is the byte-identical concatenation, so the two trainers see
    # the same corpus and the cross-check compares tokenizers, not inputs.
    train_txt = corpus_dir / "train.txt"
    total_digest = hashlib.sha256()
    with train_txt.open("wb") as out:
        for code in PLAN_A_CODES:
            payload = (langs_dir / f"{code}.txt").read_bytes()
            out.write(payload)
            total_digest.update(payload)

    total = train_txt.stat().st_size
    expected_total = budget * len(PLAN_A_CODES)
    if total != expected_total:
        raise AssertionError(f"train.txt is {total:,}, expected {expected_total:,}")

    manifest = {
        "schema_version": 2,
        "kind": "plan_a_equal_byte_corpus",
        "tier": tier,
        "bytes_per_language": budget,
        "total_bytes": total,
        "languages": PLAN_A_CODES,
        "per_language": per_language,
        "corpus_dir_for_official_trainer": str(langs_dir),
        "train_txt": {
            "path": str(train_txt),
            "sha256": total_digest.hexdigest(),
            "order": PLAN_A_CODES,
            "separator": "\\n",
        },
        "equal_bytes_verified": True,
        "notes": [
            "Pass --corpus-dir corpus/langs and --num-bytes total_bytes to the "
            "official trainer so get_files_with_num_bytes takes every "
            "per-language file whole. train.txt is deliberately NOT in that "
            "directory: the trainer globs *.txt and would double-count.",
            "Pass separator=b'\\n' to gigatoken so documents are lines, "
            "matching what HuggingFace's trainer reads from the split files.",
            "train.txt is the byte-identical concatenation of the per-language "
            "files in PLAN_A_CODES order.",
        ],
    }
    atomic_write_json(corpus_dir / "manifest.json", manifest)

    for lang in PLAN_A_LANGS:
        meta = per_language[lang.code]
        print(
            f"  {lang.code:10s} {meta['bytes']:>13,} bytes  "
            f"{meta['lines']:>9,} lines  (of {meta['source_bytes_available']:,} available)"
        )
    print(f"\n{len(PLAN_A_CODES)} languages x {budget:,} = {total:,} bytes")
    print(f"Wrote {corpus_dir}/  (per-language .txt, train.txt, manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
