#!/usr/bin/env python3
"""Emit three matched OLMo-1B job specs from Plan B preflight (AWS/B200 handoff)."""

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
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmarks/plan_b_olmo.json")
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    materialization = json.loads(args.materialization.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if preflight.get("status") != "ready_for_olmo_training":
        raise RuntimeError("preflight status must be ready_for_olmo_training")

    jobs = []
    for arm in ("bpe", "superbpe", "parity"):
        context_key = "bpe_context_tokens" if arm == "bpe" else f"{arm}_context_tokens"
        jobs.append(
            {
                "job_name": f"olmo1b-{arm}-100k-tied",
                "arm": arm,
                "architecture": config["architecture"],
                "context_tokens": preflight["context_tokens_byte_matched"][context_key],
                "shard_summary": materialization["arms"][arm],
                "equal_byte_target": preflight["equal_byte_target"],
                "equal_flop_baseline": preflight["equal_flop_baseline"],
                "continue_to_equal_flops": arm in preflight["continue_to_equal_flops"],
                "seed": 0,
                "dtype": config["architecture"]["dtype"],
                "tie_word_embeddings": True,
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "plan_b_olmo_job_bundle",
        "preflight_sha256": sha256_file(args.preflight),
        "materialization_sha256": sha256_file(args.materialization),
        "config_sha256": sha256_file(args.config),
        "jobs": jobs,
        "pairwise_deltas": preflight["pairwise_deltas"],
        "execution_notes": [
            "Submit these jobs to the shared B200 allocation sequentially unless partitions are proven equivalent.",
            "Do not start until FSx/NVMe shard staging matches materialization hashes.",
            "Full model training is out of band of this CPU repository; this bundle is the immutable launch contract.",
        ],
    }
    atomic_write_json(args.result, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
