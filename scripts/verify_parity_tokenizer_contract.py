#!/usr/bin/env python3
"""Verify Parity vs BPE contracts: vocab size, specials/alphabet, intentional merge divergence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json, sha256_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpe-dir", type=Path, required=True)
    parser.add_argument("--parity-dir", type=Path, required=True)
    parser.add_argument("--expected-vocab-size", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def _vocab(path: Path) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in value.items()}


def _merges(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bpe_vocab = _vocab(args.bpe_dir / "vocab.json")
    parity_vocab = _vocab(args.parity_dir / "vocab.json")
    bpe_merges = _merges(args.bpe_dir / "merges.txt")
    parity_merges = _merges(args.parity_dir / "merges.txt")

    # Prefer unicode-map singles: one mapped char per seeded byte atom.
    bpe_atoms = {tok for tok in bpe_vocab if len(tok) == 1}
    parity_atoms = {tok for tok in parity_vocab if len(tok) == 1}
    # Official BPE may seed only corpus-observed atoms; Parity seeds all 256 bytes.
    bpe_atoms_subset_of_parity = bpe_atoms.issubset(parity_atoms)

    common_prefix = 0
    for left, right in zip(bpe_merges, parity_merges):
        if left != right:
            break
        common_prefix += 1

    checks = {
        "bpe_vocab_size": len(bpe_vocab) == args.expected_vocab_size,
        "parity_vocab_size": len(parity_vocab) == args.expected_vocab_size,
        "parity_full_byte_alphabet": len(parity_atoms) >= 256,
        "bpe_atoms_subset_of_parity": bpe_atoms_subset_of_parity,
        "artifacts_present": all(
            (args.parity_dir / name).is_file()
            for name in ("vocab.json", "merges.txt", "meta.json")
        ),
        "merges_differ_by_design": bpe_merges != parity_merges,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "kind": "parity_bpe_contract_verification",
        "passed": passed,
        "checks": checks,
        "counts": {
            "bpe_merges": len(bpe_merges),
            "parity_merges": len(parity_merges),
            "common_merge_prefix": common_prefix,
            "bpe_atoms": len(bpe_atoms),
            "parity_atoms": len(parity_atoms),
        },
        "hashes": {
            "bpe_vocab": sha256_file(args.bpe_dir / "vocab.json"),
            "bpe_merges": sha256_file(args.bpe_dir / "merges.txt"),
            "parity_vocab": sha256_file(args.parity_dir / "vocab.json"),
            "parity_merges": sha256_file(args.parity_dir / "merges.txt"),
        },
        "notes": [
            "Parity and BPE share STAGE1 pretok and vocab size; merge lists diverge by design.",
            "SuperBPE prefix verification remains a separate BPE↔SuperBPE check.",
        ],
    }
    atomic_write_json(args.result, result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
