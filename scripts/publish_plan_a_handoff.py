#!/usr/bin/env python3
"""Publish Plan A READY.json binding three tokenizer digests and verification reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import SCHEMA_VERSION, atomic_write_json, sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpe-dir", type=Path, required=True)
    parser.add_argument("--superbpe-dir", type=Path, required=True)
    parser.add_argument("--parity-dir", type=Path, required=True)
    parser.add_argument("--bpe-superbpe-verification", type=Path, required=True)
    parser.add_argument("--parity-bpe-verification", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--mixture-shares", type=Path, help="Optional frozen language shares JSON")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stage", default="pilot", choices=("pilot", "final", "smoke"))
    return parser.parse_args(argv)


def _arm_digest(directory: Path) -> dict[str, str]:
    required = ("vocab.json", "merges.txt", "meta.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} missing {missing}")
    return {
        "directory": str(directory.resolve()),
        "vocab_sha256": sha256_file(directory / "vocab.json"),
        "merges_sha256": sha256_file(directory / "merges.txt"),
        "meta_sha256": sha256_file(directory / "meta.json"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bpe_superbpe = json.loads(args.bpe_superbpe_verification.read_text(encoding="utf-8"))
    parity_bpe = json.loads(args.parity_bpe_verification.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))

    if not bpe_superbpe.get("passed"):
        raise RuntimeError("BPE↔SuperBPE verification did not pass")
    if not parity_bpe.get("passed"):
        raise RuntimeError("Parity↔BPE contract verification did not pass")
    if not calibration.get("freeze") and args.stage == "final":
        raise RuntimeError("Calibration is not frozen; refuse final READY.json")

    mixture = None
    if args.mixture_shares is not None:
        mixture = json.loads(args.mixture_shares.read_text(encoding="utf-8"))
    elif "updated_shares" in calibration:
        mixture = calibration["updated_shares"]

    ready = {
        "schema_version": SCHEMA_VERSION,
        "kind": "plan_a_ready",
        "stage": args.stage,
        "arms": {
            "bpe": _arm_digest(args.bpe_dir),
            "superbpe": _arm_digest(args.superbpe_dir),
            "parity": _arm_digest(args.parity_dir),
        },
        "verifications": {
            "bpe_superbpe": {
                "path": str(args.bpe_superbpe_verification.resolve()),
                "passed": True,
                "sha256": sha256_file(args.bpe_superbpe_verification),
            },
            "parity_bpe": {
                "path": str(args.parity_bpe_verification.resolve()),
                "passed": True,
                "sha256": sha256_file(args.parity_bpe_verification),
            },
        },
        "calibration": {
            "path": str(args.calibration.resolve()),
            "sha256": sha256_file(args.calibration),
            "freeze": bool(calibration.get("freeze")),
            "shared_premium_formula": "(r_bpe * r_superbpe * r_parity) ** (1/3)",
        },
        "corpus_manifest": {
            "path": str(args.corpus_manifest.resolve()),
            "sha256": sha256_file(args.corpus_manifest),
            "actual_bytes": corpus.get("actual_bytes"),
            "records": corpus.get("records"),
        },
        "mixture_shares": mixture,
        "plan_b_inputs": [
            "tokenizers/final/{bpe,superbpe,parity}",
            "manifests/mixture_shares.json",
            "handoff/READY.json",
        ],
    }
    atomic_write_json(args.result, ready)
    print(json.dumps(ready, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
