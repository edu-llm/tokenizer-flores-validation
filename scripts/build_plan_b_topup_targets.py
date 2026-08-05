#!/usr/bin/env python3
"""Build absolute acquire targets for Plan B pool top-up from mixture.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mixture",
        type=Path,
        default=ROOT / "artifacts" / "plan_b" / "mixture.json",
    )
    p.add_argument(
        "--result",
        type=Path,
        default=ROOT / "artifacts" / "plan_b" / "topup_targets.json",
    )
    p.add_argument(
        "--langs",
        nargs="*",
        default=["eng_Latn", "hin_Deva", "hun_Latn", "zho_Hans"],
        help="Languages to include (default: the four short rich langs)",
    )
    args = p.parse_args(argv)
    mix = json.loads(args.mixture.read_text(encoding="utf-8"))
    acquire = mix["acquire_pool_bytes"]
    # Include all mixture langs so append merges cleanly; non-listed keep prior.
    targets = {lang: int(round(float(acquire[lang]))) for lang in mix["languages"]}
    # Restrict file to requested langs for the pull --langs flag, but keep full map
    # available under "all".
    payload = {
        "kind": "plan_b_topup_targets",
        "from_mixture": str(args.mixture.resolve()),
        "langs": list(args.langs),
        "targets": {lang: targets[lang] for lang in args.langs},
        "all_acquire_targets": targets,
    }
    atomic_write_json(args.result, payload)
    # Also write flat {lang: bytes} next to it for --targets-json.
    flat = args.result.with_name(args.result.stem + "_flat.json")
    atomic_write_json(flat, {lang: targets[lang] for lang in args.langs})
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.result} and {flat}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
