"""Tests for Plan B UniMax pool tokenize → ``.u32le.bin``."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_tiny_pool(lang_dir: Path, lang: str, n_docs: int = 20) -> None:
    lang_dir.mkdir(parents=True, exist_ok=True)
    path = lang_dir / "part-00000.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n_docs):
            text = f"{lang} document number {i} with some filler text for bytes."
            row = {"id": f"{lang}-{i:04d}", "text": text, "digest": f"d{i}"}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tiny_tokenizer(dir_path: Path, *, vocab_extra: dict[str, int] | None = None) -> None:
    """Write a minimal HF tokenizer.json that can encode ASCII-ish text."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    dir_path.mkdir(parents=True, exist_ok=True)
    # Train a tiny BPE on a few lines so encode works without gigatoken artifacts.
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=200, special_tokens=["<unk>"])
    corpus = [
        "eng_Latn document number with some filler text for bytes.",
        "hun_Latn document number with some filler text for bytes.",
        "hello world tokenizer test abc def ghi jkl mno",
    ] * 20
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(str(dir_path / "tokenizer.json"))
    # vocab.json for READY digest helpers
    vocab = tok.get_vocab()
    if vocab_extra:
        vocab.update(vocab_extra)
    (dir_path / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (dir_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")


@pytest.fixture
def tiny_env(tmp_path: Path):
    pools = tmp_path / "pools"
    langs = ["eng_Latn", "hun_Latn", "zho_Hans", "hin_Deva", "swh_Latn", "hat_Latn"]
    for lang in langs:
        _write_tiny_pool(pools / lang, lang, n_docs=30)

    tok_root = tmp_path / "tokenizers"
    _tiny_tokenizer(tok_root / "bpe")
    # Second arm: retrain with different seed corpus so token counts differ,
    # but bytes stay identical.
    _tiny_tokenizer(
        tok_root / "superbpe",
        vocab_extra=None,
    )
    # Force a distinct superbpe by appending extra training signal
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=220, special_tokens=["<unk>"])
    corpus = [
        "superbpe extra merges for whitespace bridging words together now",
        "hat_Latn document number with some filler text for bytes.",
    ] * 30
    tok.train_from_iterator(corpus, trainer=trainer)
    superbpe_dir = tok_root / "superbpe"
    tok.save(str(superbpe_dir / "tokenizer.json"))
    (superbpe_dir / "vocab.json").write_text(
        json.dumps(tok.get_vocab(), ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (superbpe_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")

    # Minimal mixture with tiny allocations so smoke finishes instantly.
    alloc_bytes = 400  # ~a few docs each
    mixture = {
        "kind": "plan_b_mixture",
        "schema_version": 1,
        "allocation": {
            lang: {
                "bytes": alloc_bytes,
                "tokens": alloc_bytes / 4.0,
                "epochs_of_available": 1.0,
                "passes_over_pool": 1.0 if lang != "hat_Latn" else 2.0,
                "capped": lang == "hat_Latn",
                "share": 1.0 / 6.0,
            }
            for lang in langs
        },
        "languages": langs,
    }
    mix_path = tmp_path / "mixture.json"
    mix_path.write_text(json.dumps(mixture), encoding="utf-8")
    return {
        "pools": pools,
        "tok_root": tok_root,
        "mixture": mix_path,
        "langs": langs,
        "out": tmp_path / "tokens",
    }


def test_tokenize_pools_both_arms_val_byte_match(tiny_env):
    mod = _load_script("run_plan_b_tokenize_pools")
    rc = mod.main(
        [
            "--pools-dir",
            str(tiny_env["pools"]),
            "--mixture",
            str(tiny_env["mixture"]),
            "--tokenizers-root",
            str(tiny_env["tok_root"]),
            "--output-dir",
            str(tiny_env["out"]),
            "--budget-scale",
            "1.0",
            "--val-docs",
            "5",
            "--shard-bytes",
            "1048576",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (tiny_env["out"] / "tokenize_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["equal_bytes_ok"] is True
    bpe_bytes = manifest["arms"]["bpe"]["total_train_utf8_bytes"]
    sbpe_bytes = manifest["arms"]["superbpe"]["total_train_utf8_bytes"]
    assert bpe_bytes == sbpe_bytes
    assert bpe_bytes > 0

    for arm in ("bpe", "superbpe"):
        for lang in tiny_env["langs"]:
            lang_dir = tiny_env["out"] / arm / "tokens" / lang
            trains = sorted(lang_dir.glob("train-*.u32le.bin"))
            vals = sorted(lang_dir.glob("val-*.u32le.bin"))
            assert trains, f"missing train for {arm}/{lang}"
            assert vals, f"missing val for {arm}/{lang}"
            for p in trains + vals:
                size = p.stat().st_size
                assert size % 4 == 0
                raw = p.read_bytes()
                n = size // 4
                ids = struct.unpack("<" + "I" * n, raw)
                assert len(ids) == n

    # Val ids excluded from train sample metadata.
    for lang, meta in manifest["languages"].items():
        assert len(meta["val_ids"]) == 5
        assert meta["sampled_bytes"] >= 400 or meta["sampled_docs"] >= 1


def test_holdout_excludes_val_from_train(tmp_path: Path):
    mod = _load_script("run_plan_b_tokenize_pools")
    lang = "eng_Latn"
    _write_tiny_pool(tmp_path / lang, lang, n_docs=12)
    val, val_ids, train_n, _ = mod.holdout_val(tmp_path / lang, lang, val_docs=3)
    assert len(val) == 3
    assert train_n == 9
    assert len(val_ids) == 3
    train_ids = [d["id"] for d in mod.iter_train_docs(tmp_path / lang, lang, val_ids)]
    assert len(train_ids) == 9
    assert not any(i in val_ids for i in train_ids)
    assert val[-1]["id"] == "eng_Latn-0011"


def test_build_mixture_reads_null_available_bytes(tmp_path: Path):
    mod = _load_script("build_plan_b_mixture")
    man = {
        "kind": "plan_b_pools_manifest",
        "languages": {
            "eng_Latn": {
                "bytes": 1e9,
                "available_bytes": None,
                "unbounded": True,
            },
            "hat_Latn": {"bytes": 0.772e9, "available_bytes": 0.772e9},
            "swh_Latn": {"bytes": 4.577e9, "available_bytes": 4.577e9},
            "hin_Deva": {"bytes": 1e9, "available_bytes": 34.4e9},
            "hun_Latn": {"bytes": 1e9, "available_bytes": 98.6e9},
            "zho_Hans": {"bytes": 1e9, "available_bytes": 1622e9},
        },
    }
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    out = tmp_path / "mixture.json"
    assert mod.main(["--pools-manifest", str(man_path), "--result", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["available_bytes"]["eng_Latn"] is None  # serialized inf → null
