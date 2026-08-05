#!/usr/bin/env python3
"""Validate Plan A READY.json and emit Plan B equal-byte / equal-FLOP schedule.

``--target-train-bytes`` is derived from ``mixture.json`` ``budget_bytes.derived`` —
there is no silent 50 GB default (that was the superseded 50B-token plan).
"""

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
    parser.add_argument(
        "--mixture",
        type=Path,
        required=True,
        help="artifacts/plan_b/mixture.json from build_plan_b_mixture.py",
    )
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def _budget_from_mixture(mixture: dict) -> int:
    if mixture.get("kind") != "plan_b_mixture":
        raise ValueError("mixture kind must be plan_b_mixture")
    budget = mixture.get("budget_bytes") or {}
    derived = budget.get("derived")
    if derived is None:
        raise ValueError("mixture.json missing budget_bytes.derived")
    return int(round(float(derived)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ready = json.loads(args.ready.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization.read_text(encoding="utf-8"))
    mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
    if ready.get("kind") != "plan_a_ready":
        raise ValueError("READY.json kind must be plan_a_ready")
    for arm in ("bpe", "superbpe"):
        if arm not in ready.get("arms", {}):
            raise ValueError(f"READY missing {arm}")
        if arm not in materialization.get("arms", {}):
            raise ValueError(f"materialization missing {arm}")

    target_train_bytes = _budget_from_mixture(mixture)
    context = materialization.get("context_tokens_byte_matched", {})
    schedule = {
        "schema_version": 1,
        "kind": "plan_b_preflight_schedule",
        "ready_sha256": sha256_file(args.ready),
        "materialization_sha256": sha256_file(args.materialization),
        "mixture_sha256": sha256_file(args.mixture),
        "arms": ["bpe", "superbpe"],
        "context_tokens_byte_matched": context,
        "equal_byte_target": target_train_bytes,
        "equal_flop_baseline": "bpe",
        "continue_to_equal_flops": ["superbpe"],
        "pairwise_deltas": ["superbpe-bpe"],
        "status": "ready_for_olmo_training",
        "notes": [
            "Full OLMo-1B training runs on the shared B200 after this CPU preflight.",
            "Checkpoint selection is budget-based (equal bytes / equal FLOPs), not best-on-benchmark.",
            "equal_byte_target comes from mixture.json budget_bytes.derived (UniMax).",
        ],
    }
    atomic_write_json(args.result, schedule)
    print(json.dumps(schedule, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
