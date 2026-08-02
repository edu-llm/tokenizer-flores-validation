"""The Plan A corpus must be exactly equal-byte across languages.

Guards [plans/02-tokenizer-training.md §5.6](../plans/02-tokenizer-training.md).
The superseded builder produced a 6x skew that confounded every cross-language
premium with the mix, so "roughly balanced" is not good enough to assert.

These also replace the retired ``meta.json:train_files`` assertion on the
gigatoken path: gigatoken takes one mmapped file, so the official trainer's
whole-file selection hazard cannot arise, and the corpus manifest becomes the
place where balance is proven.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_plan_a_research_corpus import main as build_main
from src.plan_a_langs import PLAN_A_CODES

BUDGET = 4096


def _write_sources(source_dir: Path, *, codes=PLAN_A_CODES, per_lang_bytes=20_000) -> None:
    """Stage a plausible multi-script source file per language."""
    source_dir.mkdir(parents=True, exist_ok=True)
    # Multi-byte scripts matter here: the truncation path cuts on a UTF-8
    # boundary and can land short, which is what the padding covers.
    bodies = {
        "eng_Latn": "the quick brown fox jumps over the lazy dog",
        "hun_Latn": "a gyors barna róka átugorja a lusta kutyát",
        "zho_Hans": "敏捷的棕色狐狸跳过了那只懒狗并且继续向前奔跑",
        "hin_Deva": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूद गई",
        "swh_Latn": "mbweha wa kahawia mwepesi anaruka juu ya mbwa mvivu",
        "hat_Latn": "rena mawon rapid la sote sou chen parese a",
    }
    for code in codes:
        body = bodies.get(code, "generic filler line for this language")
        line = (body + "\n").encode("utf-8")
        reps = per_lang_bytes // len(line) + 2
        (source_dir / f"{code}.txt").write_bytes(line * reps)


def _build(tmp_path: Path, **kwargs) -> dict:
    source_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    _write_sources(source_dir, **kwargs)
    rc = build_main(
        [
            "--bytes-per-language",
            str(BUDGET),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    return json.loads((out_dir / "corpus" / "manifest.json").read_text(encoding="utf-8"))


def test_every_language_gets_exactly_the_same_bytes(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    sizes = {code: manifest["per_language"][code]["bytes"] for code in PLAN_A_CODES}
    assert set(sizes) == set(PLAN_A_CODES)
    assert max(sizes.values()) == min(sizes.values()) == BUDGET, sizes


def test_per_language_files_are_exactly_the_budget_on_disk(tmp_path: Path) -> None:
    """The manifest could agree with itself and still lie about the files."""
    manifest = _build(tmp_path)
    langs = Path(manifest["corpus_dir_for_official_trainer"])
    for code in PLAN_A_CODES:
        assert (langs / f"{code}.txt").stat().st_size == BUDGET


def test_official_corpus_dir_holds_exactly_the_language_files(tmp_path: Path) -> None:
    """The official trainer globs *.txt in corpus_dir and sums what it finds.

    train.txt beside the per-language files would train on the corpus twice
    over, and nothing would error -- the same silent-failure shape as the
    get_files_with_num_bytes prefix hazard.
    """
    manifest = _build(tmp_path)
    langs = Path(manifest["corpus_dir_for_official_trainer"])
    found = sorted(p.name for p in langs.glob("*.txt"))
    assert found == sorted(f"{code}.txt" for code in PLAN_A_CODES)

    # What the trainer would total up must equal what we tell it to read.
    assert sum(p.stat().st_size for p in langs.glob("*.txt")) == manifest["total_bytes"]

    train_txt = Path(manifest["train_txt"]["path"])
    assert train_txt.parent != langs
    assert not (langs / "train.txt").exists()


def test_train_txt_is_the_byte_identical_concatenation(tmp_path: Path) -> None:
    """The two trainers must see the same corpus or the cross-check is void."""
    manifest = _build(tmp_path)
    train_txt = Path(manifest["train_txt"]["path"])
    langs = Path(manifest["corpus_dir_for_official_trainer"])

    concatenated = b"".join(
        (langs / f"{code}.txt").read_bytes() for code in PLAN_A_CODES
    )
    assert train_txt.read_bytes() == concatenated
    assert train_txt.stat().st_size == BUDGET * len(PLAN_A_CODES)
    assert manifest["total_bytes"] == BUDGET * len(PLAN_A_CODES)
    assert (
        hashlib.sha256(concatenated).hexdigest() == manifest["train_txt"]["sha256"]
    )


def test_every_file_is_valid_utf8(tmp_path: Path) -> None:
    """Truncation cuts mid-line, so it must respect character boundaries."""
    manifest = _build(tmp_path)
    langs = Path(manifest["corpus_dir_for_official_trainer"])
    for code in PLAN_A_CODES:
        (langs / f"{code}.txt").read_text(encoding="utf-8")  # raises if invalid
    Path(manifest["train_txt"]["path"]).read_text(encoding="utf-8")


def test_short_language_is_a_hard_failure_not_a_silent_shrink(tmp_path: Path) -> None:
    """The skew got in because a short language was quietly accepted."""
    source_dir = tmp_path / "raw"
    _write_sources(source_dir)
    # hat_Latn is the binding language in the real set; starve it here.
    (source_dir / "hat_Latn.txt").write_bytes(b"too little\n")

    with pytest.raises(ValueError, match="budget"):
        build_main(
            [
                "--bytes-per-language",
                str(BUDGET),
                "--source-dir",
                str(source_dir),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_missing_language_is_a_hard_failure(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    _write_sources(source_dir)
    (source_dir / "hin_Deva.txt").unlink()

    with pytest.raises(FileNotFoundError, match="hin_Deva"):
        build_main(
            [
                "--bytes-per-language",
                str(BUDGET),
                "--source-dir",
                str(source_dir),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_tier_budget_matches_the_config(tmp_path: Path) -> None:
    """resolve_budget cross-checks bytes_per_language against corpus_bytes."""
    from scripts.build_plan_a_research_corpus import DEFAULT_CONFIG

    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for name, tier in config["tiers"].items():
        assert tier["corpus_bytes"] == tier["bytes_per_language"] * len(PLAN_A_CODES), (
            f"tier {name} is inconsistent with the {len(PLAN_A_CODES)}-language scope"
        )
