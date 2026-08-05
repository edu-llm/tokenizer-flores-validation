"""Smoke tests for Plan B pool targets, mixture builder, and preflight wiring."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.benchmark import atomic_write_json
from src.plan_b_mixture import (
    PLANNING_AVAILABLE_BYTES,
    PLANNING_EPOCH_CAP,
    acquire_pool_bytes,
    planning_budget_bytes,
    unimax_allocation,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_acquire_targets_match_planning_table():
    budget = planning_budget_bytes()
    alloc = unimax_allocation(PLANNING_AVAILABLE_BYTES, budget, PLANNING_EPOCH_CAP)
    acquire = acquire_pool_bytes(alloc, PLANNING_AVAILABLE_BYTES)
    assert acquire["hat_Latn"] == pytest.approx(0.772e9)
    assert acquire["swh_Latn"] == pytest.approx(4.577e9)
    assert acquire["eng_Latn"] == pytest.approx(alloc["eng_Latn"].bytes * 1.15)
    assert sum(acquire.values()) == pytest.approx(74e9, rel=0.05)


def test_build_plan_b_mixture_script_writes_kind(tmp_path: Path):
    mod = _load_script("build_plan_b_mixture")
    out = tmp_path / "mixture.json"
    assert mod.main(["--result", str(out)]) == 0
    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["kind"] == "plan_b_mixture"
    assert man["allocation"]["hat_Latn"]["capped"] is True
    assert man["budget_bytes"]["derived"] == pytest.approx(78e9)


def test_preflight_requires_mixture_and_derives_budget(tmp_path: Path):
    mod = _load_script("run_plan_b_preflight")
    ready = {
        "kind": "plan_a_ready",
        "arms": {"bpe": {"path": "x"}, "superbpe": {"path": "y"}},
    }
    materialization = {
        "arms": {"bpe": {"n": 1}, "superbpe": {"n": 1}},
        "context_tokens_byte_matched": {
            "bpe_context_tokens": 2048,
            "superbpe_context_tokens": 1800,
        },
    }
    mixture = {
        "kind": "plan_b_mixture",
        "budget_bytes": {"derived": 78_000_000_000},
    }
    ready_p = tmp_path / "READY.json"
    mat_p = tmp_path / "mat.json"
    mix_p = tmp_path / "mixture.json"
    out_p = tmp_path / "preflight.json"
    atomic_write_json(ready_p, ready)
    atomic_write_json(mat_p, materialization)
    atomic_write_json(mix_p, mixture)

    assert (
        mod.main(
            [
                "--ready",
                str(ready_p),
                "--materialization",
                str(mat_p),
                "--mixture",
                str(mix_p),
                "--result",
                str(out_p),
            ]
        )
        == 0
    )
    sched = json.loads(out_p.read_text(encoding="utf-8"))
    assert sched["equal_byte_target"] == 78_000_000_000
    assert "mixture_sha256" in sched


def test_shard_writer_roundtrip_jsonl_zst(tmp_path: Path):
    zstd = pytest.importorskip("zstandard")
    mod = _load_script("pull_plan_b_pools")
    lang_dir = tmp_path / "eng_Latn"
    w = mod._ShardWriter(lang_dir, shard_bytes=50)
    w.write({"id": "a", "text": "hello world"}, len("hello world"))
    w.write({"id": "b", "text": "more text here"}, len("more text here"))
    w.close()
    parts = sorted(lang_dir.glob("part-*.jsonl.zst"))
    assert parts
    # stream_writer frames omit content size — decompress via stream_reader.
    with parts[0].open("rb") as fh:
        with zstd.ZstdDecompressor().stream_reader(fh) as reader:
            raw = reader.read()
    lines = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x]
    assert lines[0]["text"] == "hello world"
    assert lines[1]["id"] == "b"


def test_emit_jobs_reads_tie_from_config(tmp_path: Path):
    mod = _load_script("emit_plan_b_olmo_jobs")
    preflight = {
        "status": "ready_for_olmo_training",
        "context_tokens_byte_matched": {
            "bpe_context_tokens": 2048,
            "superbpe_context_tokens": 1800,
        },
        "equal_byte_target": 78_000_000_000,
        "equal_flop_baseline": "bpe",
        "continue_to_equal_flops": ["superbpe"],
        "pairwise_deltas": ["superbpe-bpe"],
    }
    materialization = {"arms": {"bpe": {}, "superbpe": {}}}
    config = {
        "architecture": {
            "tie_word_embeddings": False,
            "dtype": "bf16",
            "vocab_size": 100000,
        }
    }
    for name, obj in (
        ("pre.json", preflight),
        ("mat.json", materialization),
        ("cfg.json", config),
    ):
        atomic_write_json(tmp_path / name, obj)

    out = tmp_path / "jobs.json"
    assert (
        mod.main(
            [
                "--preflight",
                str(tmp_path / "pre.json"),
                "--materialization",
                str(tmp_path / "mat.json"),
                "--config",
                str(tmp_path / "cfg.json"),
                "--result",
                str(out),
            ]
        )
        == 0
    )
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["jobs"][0]["tie_word_embeddings"] is False
    assert bundle["jobs"][0]["job_name"].endswith("-untied")
