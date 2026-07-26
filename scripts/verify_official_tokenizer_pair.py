#!/usr/bin/env python3
"""Verify that a SuperBPE artifact differs only after its inherited BPE prefix."""

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
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--superbpe-dir", type=Path, required=True)
    parser.add_argument("--transition-vocab-size", type=int, required=True)
    parser.add_argument("--expected-vocab-size", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def _merges(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def _vocab(path: Path) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected vocab object: {path}")
    return {str(token): int(token_id) for token, token_id in value.items()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_vocab = _vocab(args.baseline_dir / "vocab.json")
    superbpe_vocab = _vocab(args.superbpe_dir / "vocab.json")
    baseline_merges = _merges(args.baseline_dir / "merges.txt")
    superbpe_merges = _merges(args.superbpe_dir / "merges.txt")
    inherited_file = args.superbpe_dir / "requested_initial_merges.txt"
    inherited_merges = _merges(inherited_file) if inherited_file.is_file() else []

    initial_vocab_size = len(baseline_vocab) - len(baseline_merges)
    expected_inherited = args.transition_vocab_size - initial_vocab_size
    shared_id_mismatches = [
        token
        for token, token_id in baseline_vocab.items()
        if token_id < args.transition_vocab_size
        and superbpe_vocab.get(token) != token_id
    ]
    common_prefix_length = 0
    for baseline_merge, superbpe_merge in zip(baseline_merges, superbpe_merges):
        if baseline_merge != superbpe_merge:
            break
        common_prefix_length += 1

    checks = {
        "baseline_vocab_size": len(baseline_vocab) == args.expected_vocab_size,
        "superbpe_vocab_size": len(superbpe_vocab) == args.expected_vocab_size,
        "transition_is_valid": 0 < expected_inherited <= len(baseline_merges),
        "saved_inherited_prefix_matches": (
            inherited_merges == baseline_merges[:expected_inherited]
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "kind": "official_tokenizer_pair_verification",
        "passed": passed,
        "checks": checks,
        "counts": {
            "initial_vocab_size": initial_vocab_size,
            "expected_inherited_merges": expected_inherited,
            "baseline_merges": len(baseline_merges),
            "superbpe_merges": len(superbpe_merges),
            "final_common_merge_prefix": common_prefix_length,
            "shared_id_mismatches": len(shared_id_mismatches),
        },
        "hashes": {
            "baseline_vocab": sha256_file(args.baseline_dir / "vocab.json"),
            "baseline_merges": sha256_file(args.baseline_dir / "merges.txt"),
            "superbpe_vocab": sha256_file(args.superbpe_dir / "vocab.json"),
            "superbpe_merges": sha256_file(args.superbpe_dir / "merges.txt"),
        },
        "mismatch_examples": shared_id_mismatches[:20],
        "notes": [
            "The official patched trainer may drop inherited merges that are "
            "unreachable under stage-two pretokenization.",
            "Final merge-prefix length and token-ID mismatches are reported, "
            "not treated as parity failures.",
        ],
    }
    atomic_write_json(args.result, result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

