#!/usr/bin/env python3
"""Calibrate shared language shares from BPE / SuperBPE / Parity token premiums."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json
from src.premium_calibration import (
    damp_shares,
    max_share_delta,
    shared_premiums,
    target_token_shares,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--premiums",
        type=Path,
        required=True,
        help="JSON mapping arm -> language -> token_premium",
    )
    parser.add_argument(
        "--prior-shares",
        type=Path,
        help="Optional JSON language -> share prior; defaults to uniform",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--freeze-threshold", type=float, default=0.05)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arm_premiums = json.loads(args.premiums.read_text(encoding="utf-8"))
    required = {"bpe", "superbpe", "parity"}
    missing = required - set(arm_premiums)
    if missing:
        raise ValueError(f"premiums JSON missing arms: {sorted(missing)}")

    shared = shared_premiums(arm_premiums)
    target = target_token_shares(shared)
    if args.prior_shares is not None:
        prior = {
            str(k): float(v)
            for k, v in json.loads(args.prior_shares.read_text(encoding="utf-8")).items()
        }
    else:
        n = len(target)
        prior = {lang: 1.0 / n for lang in target}

    updated = damp_shares(prior, target, alpha=args.alpha)
    delta = max_share_delta(prior, updated)
    result = {
        "schema_version": 1,
        "kind": "three_arm_premium_calibration",
        "arms": sorted(required),
        "shared_premiums": shared,
        "target_shares": target,
        "prior_shares": prior,
        "updated_shares": updated,
        "max_share_delta": delta,
        "freeze": delta < args.freeze_threshold,
        "freeze_threshold": args.freeze_threshold,
        "alpha": args.alpha,
    }
    atomic_write_json(args.result, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
