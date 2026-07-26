#!/usr/bin/env python3
"""Validate Plan A READY.json and emit Plan B equal-byte / equal-FLOP schedule."""

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
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--target-train-bytes", type=int, default=50_000_000_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ready = json.loads(args.ready.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization.read_text(encoding="utf-8"))
    if ready.get("kind") != "plan_a_ready":
        raise ValueError("READY.json kind must be plan_a_ready")
    for arm in ("bpe", "superbpe", "parity"):
        if arm not in ready.get("arms", {}):
            raise ValueError(f"READY missing {arm}")
        if arm not in materialization.get("arms", {}):
            raise ValueError(f"materialization missing {arm}")

    context = materialization.get("context_tokens_byte_matched", {})
    schedule = {
        "schema_version": 1,
        "kind": "plan_b_preflight_schedule",
        "ready_sha256": sha256_file(args.ready),
        "materialization_sha256": sha256_file(args.materialization),
        "arms": ["bpe", "superbpe", "parity"],
        "context_tokens_byte_matched": context,
        "equal_byte_target": args.target_train_bytes,
        "equal_flop_baseline": "bpe",
        "continue_to_equal_flops": ["superbpe", "parity"],
        "pairwise_deltas": ["superbpe-bpe", "parity-bpe", "superbpe-parity"],
        "status": "ready_for_olmo_training",
        "notes": [
            "Full OLMo-1B training runs on the shared B200 after this CPU preflight.",
            "Checkpoint selection is budget-based (equal bytes / equal FLOPs), not best-on-benchmark.",
        ],
    }
    atomic_write_json(args.result, schedule)
    print(json.dumps(schedule, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
