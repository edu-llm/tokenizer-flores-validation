#!/usr/bin/env python3
"""Measure BPE UTF-8 bytes/token on Plan B unique pools.

Loads gigatoken BPE ``tokenizer.json``, encodes a fixed per-language sample of
pool documents, and writes ``artifacts/plan_b/bytes_per_token.json``.

Prefer streaming a sample from S3/local shards rather than loading the full
~74 GB pools on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json
from src.plan_a_langs import PLAN_A_CODES
from src.plan_b_pool_io import iter_docs

DEFAULT_SAMPLE_BYTES = 8_000_000  # ~8 MB UTF-8 text per language
OUT = ROOT / "artifacts" / "plan_b" / "bytes_per_token.json"
DEFAULT_TOK = (
    ROOT / "artifacts" / "plan_a" / "scale" / "tokenizers" / "bpe" / "tokenizer.json"
)
DEFAULT_POOLS = ROOT / "artifacts" / "plan_b" / "pools"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pools-dir", type=Path, default=DEFAULT_POOLS)
    p.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOK)
    p.add_argument("--sample-bytes", type=int, default=DEFAULT_SAMPLE_BYTES)
    p.add_argument("--langs", nargs="*", default=list(PLAN_A_CODES))
    p.add_argument("--result", type=Path, default=OUT)
    p.add_argument(
        "--arm",
        default="bpe",
        help="Arm label recorded in the output (measurement uses --tokenizer-json)",
    )
    return p.parse_args(argv)


def measure_language(
    lang_dir: Path,
    lang: str,
    encode,
    *,
    sample_bytes: int,
) -> dict[str, Any]:
    if sample_bytes < 1:
        raise ValueError("sample_bytes must be positive")
    total_bytes = 0
    total_tokens = 0
    n_docs = 0
    for doc in iter_docs(lang_dir, lang):
        text = doc["text"]
        raw = text.encode("utf-8")
        if total_bytes >= sample_bytes:
            break
        # Take whole docs; stop once we would far exceed the budget after this doc
        # only if we already have some sample.
        if total_bytes > 0 and total_bytes + len(raw) > sample_bytes * 1.25:
            break
        ids = encode(text)
        total_bytes += len(raw)
        total_tokens += len(ids)
        n_docs += 1
        if total_bytes >= sample_bytes:
            break
    if total_tokens <= 0 or total_bytes <= 0:
        raise RuntimeError(f"{lang}: no tokens measured under {lang_dir}")
    bpt = total_bytes / total_tokens
    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "documents": n_docs,
        "bytes_per_token": bpt,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.tokenizer_json.is_file():
        raise SystemExit(f"missing tokenizer: {args.tokenizer_json}")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer_json))

    def encode(text: str) -> list[int]:
        return tok.encode(text).ids

    per: dict[str, float] = {}
    detail: dict[str, Any] = {}
    for lang in args.langs:
        lang_dir = args.pools_dir / lang
        if not lang_dir.is_dir():
            raise SystemExit(f"missing pool dir: {lang_dir}")
        stats = measure_language(
            lang_dir, lang, encode, sample_bytes=args.sample_bytes
        )
        per[lang] = float(stats["bytes_per_token"])
        detail[lang] = stats
        print(
            f"{lang}: bpt={per[lang]:.4f} "
            f"({stats['bytes']} bytes / {stats['tokens']} tokens, "
            f"{stats['documents']} docs)",
            flush=True,
        )

    mean = sum(per.values()) / len(per)
    payload = {
        "kind": "plan_b_bytes_per_token",
        "schema_version": 1,
        "arm": args.arm,
        "tokenizer_json": str(args.tokenizer_json.resolve()),
        "pools_dir": str(args.pools_dir.resolve()),
        "sample_bytes_target": args.sample_bytes,
        "per_language": per,
        "mean": mean,
        "detail": detail,
    }
    atomic_write_json(args.result, payload)
    print(json.dumps({"mean": mean, "per_language": per}, indent=2), flush=True)
    print(f"wrote {args.result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
