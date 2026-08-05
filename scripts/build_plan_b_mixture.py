#!/usr/bin/env python3
"""Build Plan B ``mixture.json`` from UniMax + measured bytes/token.

Reads availability (pool manifest or planning defaults), optional measured
per-language bytes/token, and writes the allocation artifact that
``run_plan_b_preflight.py`` consumes.

Spec: ``plans/03-model-pretraining.md`` §7.2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json
from src.plan_a_langs import PLAN_A_CODES
from src.plan_b_mixture import (
    PLANNING_AVAILABLE_BYTES,
    PLANNING_BYTES_PER_TOKEN,
    PLANNING_EPOCH_CAP,
    PLANNING_HEADROOM,
    PLANNING_TARGET_TOKENS,
    acquire_pool_bytes,
    allocation_shares,
    passes_over_pool,
    planning_budget_bytes,
    unimax_allocation,
    unique_pool_bytes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result",
        type=Path,
        default=ROOT / "artifacts" / "plan_b" / "mixture.json",
    )
    p.add_argument(
        "--pools-manifest",
        type=Path,
        help="Optional pretrain/pools/manifest.json; supplies measured unique_pool_bytes",
    )
    p.add_argument(
        "--bytes-per-token-json",
        type=Path,
        help="Optional JSON {lang: float} measured on the BPE arm; default is planning 3.9",
    )
    p.add_argument("--target-tokens", type=float, default=PLANNING_TARGET_TOKENS)
    p.add_argument("--epoch-cap", type=float, default=PLANNING_EPOCH_CAP)
    p.add_argument("--headroom", type=float, default=PLANNING_HEADROOM)
    p.add_argument(
        "--arm",
        default="bpe",
        help="Arm that anchors target_tokens (default bpe)",
    )
    return p.parse_args(argv)


def _load_available(pools_manifest: Path | None) -> tuple[dict[str, float], str]:
    if pools_manifest is None:
        return dict(PLANNING_AVAILABLE_BYTES), "planning_defaults"
    man = json.loads(pools_manifest.read_text(encoding="utf-8"))
    langs = man.get("languages") or man.get("langs") or {}
    avail: dict[str, float] = {}
    for code in PLAN_A_CODES:
        entry = langs.get(code) or {}
        # Explicit null available_bytes + unbounded → FineWeb-scale infinity.
        if entry.get("unbounded"):
            avail[code] = math.inf
            continue
        # Prefer declared available_bytes; fall back to pulled unique bytes.
        # Do not use .get(key, default) when the key may be present with null.
        raw = entry.get("available_bytes")
        if raw is None:
            raw = entry.get("bytes")
        if raw is None:
            raise ValueError(f"pools manifest missing bytes for {code}")
        avail[code] = float(raw)
        if entry.get("drawn_in_full") and avail[code] <= 0:
            raise ValueError(f"{code} marked drawn_in_full but bytes={avail[code]}")
    return avail, str(pools_manifest)


def _load_bpt(path: Path | None) -> tuple[dict[str, float], float, str]:
    if path is None:
        per = {code: PLANNING_BYTES_PER_TOKEN for code in PLAN_A_CODES}
        return per, PLANNING_BYTES_PER_TOKEN, "planning_constant_3.9"
    data = json.loads(path.read_text(encoding="utf-8"))
    if "per_language" in data:
        per = {k: float(v) for k, v in data["per_language"].items()}
        mean = float(data.get("mean", sum(per.values()) / len(per)))
    else:
        per = {k: float(v) for k, v in data.items()}
        mean = sum(per.values()) / len(per)
    for code in PLAN_A_CODES:
        if code not in per:
            raise ValueError(f"bytes-per-token missing {code}")
    return per, mean, str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    available, avail_src = _load_available(args.pools_manifest)
    bpt_per, bpt_mean, bpt_src = _load_bpt(args.bytes_per_token_json)

    # Budget from mean measured BPT (or planning constant) × target tokens.
    assumed_budget = planning_budget_bytes(
        target_tokens=args.target_tokens,
        bytes_per_token=PLANNING_BYTES_PER_TOKEN,
    )
    budget = float(args.target_tokens) * bpt_mean
    alloc = unimax_allocation(available, budget, args.epoch_cap)
    pools = unique_pool_bytes(alloc, available)
    acquire = acquire_pool_bytes(alloc, available, headroom=args.headroom)
    pops = passes_over_pool(alloc, pools)
    shares = allocation_shares(alloc)

    payload = {
        "kind": "plan_b_mixture",
        "schema_version": 1,
        "target_tokens": args.target_tokens,
        "target_tokens_arm": args.arm,
        "epoch_cap": args.epoch_cap,
        "headroom": args.headroom,
        "bytes_per_token": {
            "assumed": PLANNING_BYTES_PER_TOKEN,
            "measured_mean": bpt_mean,
            "per_language": bpt_per,
            "source": bpt_src,
            "arm_measured_on": args.arm,
        },
        "budget_bytes": {
            "assumed": assumed_budget,
            "derived": budget,
        },
        "available_bytes": {
            lang: (None if not math.isfinite(v) else v) for lang, v in available.items()
        },
        "available_source": avail_src,
        "unique_pool_bytes": pools,
        "acquire_pool_bytes": acquire,
        "allocation": {
            lang: {
                "bytes": a.bytes,
                "tokens": a.bytes / bpt_per[lang],
                "epochs_of_available": a.epochs_of_available,
                "passes_over_pool": pops[lang],
                "capped": a.capped,
                "share": shares[lang],
            }
            for lang, a in alloc.items()
        },
        "languages": list(PLAN_A_CODES),
    }
    atomic_write_json(args.result, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
