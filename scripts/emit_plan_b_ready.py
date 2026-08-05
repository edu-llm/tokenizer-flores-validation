#!/usr/bin/env python3
"""Emit a minimal Plan A READY.json for Plan B tokenize / preflight consumers.

Binds local (or published) BPE + SuperBPE artifact directories and vocab digests.
Does not require calibration freeze — suitable for smoke / research scratch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import SCHEMA_VERSION, atomic_write_json, sha256_file

DEFAULT_TOK_ROOT = ROOT / "artifacts" / "plan_a" / "scale" / "tokenizers"
DEFAULT_OUT = ROOT / "artifacts" / "plan_b" / "READY.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokenizers-root",
        type=Path,
        default=DEFAULT_TOK_ROOT,
        help="Directory containing bpe/ and superbpe/ artifact dirs",
    )
    p.add_argument("--bpe-dir", type=Path, help="Override BPE artifact dir")
    p.add_argument("--superbpe-dir", type=Path, help="Override SuperBPE artifact dir")
    p.add_argument("--result", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stage", default="smoke", choices=("pilot", "final", "smoke"))
    return p.parse_args(argv)


def _arm_digest(directory: Path) -> dict[str, str]:
    vocab = directory / "vocab.json"
    merges = directory / "merges.txt"
    tok_json = directory / "tokenizer.json"
    if not vocab.is_file():
        raise FileNotFoundError(f"{directory}: missing vocab.json")
    if not tok_json.is_file():
        raise FileNotFoundError(f"{directory}: missing tokenizer.json")
    out: dict[str, str] = {
        "directory": str(directory.resolve()),
        "path": str(directory.resolve()),
        "vocab_sha256": sha256_file(vocab),
        "tokenizer_json": str(tok_json.resolve()),
    }
    if merges.is_file():
        out["merges_sha256"] = sha256_file(merges)
    meta = directory / "meta.json"
    if meta.is_file():
        out["meta_sha256"] = sha256_file(meta)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bpe_dir = args.bpe_dir or (args.tokenizers_root / "bpe")
    superbpe_dir = args.superbpe_dir or (args.tokenizers_root / "superbpe")
    ready = {
        "schema_version": SCHEMA_VERSION,
        "kind": "plan_a_ready",
        "stage": args.stage,
        "arms": {
            "bpe": _arm_digest(bpe_dir),
            "superbpe": _arm_digest(superbpe_dir),
        },
        "note": "Minimal READY for Plan B tokenize; not the frozen Plan A handoff.",
    }
    atomic_write_json(args.result, ready)
    print(json.dumps(ready, indent=2, sort_keys=True))
    print(f"wrote {args.result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
