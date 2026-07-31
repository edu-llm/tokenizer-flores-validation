#!/usr/bin/env python3
"""Orchestrate Plan A two-arm tokenizer training, verification, and calibration.

Designed for local smoke and AWS Batch: same CLIs and result schemas. Full 10 GB
/ 100k runs belong on measured high-memory Batch jobs after tier gates pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dev-lang-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--superbpe-repo", type=Path, required=True)
    parser.add_argument("--num-bytes", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--transition-vocab-size", type=int, required=True)
    parser.add_argument(
        "--patched-tokenizers-commit",
        default="757f2a55c0820ed47064e1fe473deea39b7b611b",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    parser.add_argument("--min-available-gb", type=float, default=1.0)
    parser.add_argument("--stage", default="smoke", choices=("smoke", "pilot", "final"))
    parser.add_argument("--skip-bpe", action="store_true")
    parser.add_argument("--skip-superbpe", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work = args.work_dir
    tok = work / "tokenizers" / args.stage
    results = work / "results" / args.stage
    logs = work / "logs" / args.stage
    for path in (tok, results, logs):
        path.mkdir(parents=True, exist_ok=True)

    py = str(args.python)
    bpe_dir = tok / "bpe"
    superbpe_dir = tok / "superbpe"

    if not args.skip_bpe:
        _run(
            [
                py,
                "scripts/run_official_tokenizer_benchmark.py",
                "--arm",
                "bpe",
                "--superbpe-repo",
                str(args.superbpe_repo),
                "--corpus-dir",
                str(args.corpus_dir),
                "--output-dir",
                str(bpe_dir),
                "--result",
                str(results / "bpe.json"),
                "--log",
                str(logs / "bpe.log"),
                "--num-bytes",
                str(args.num_bytes),
                "--vocab-size",
                str(args.vocab_size),
                "--patched-tokenizers-commit",
                args.patched_tokenizers_commit,
                "--max-rss-gb",
                str(args.max_rss_gb),
                "--min-available-gb",
                str(args.min_available_gb),
                *(["--force"] if args.force else []),
            ]
        )

    if not args.skip_superbpe:
        _run(
            [
                py,
                "scripts/run_official_tokenizer_benchmark.py",
                "--arm",
                "superbpe",
                "--superbpe-repo",
                str(args.superbpe_repo),
                "--corpus-dir",
                str(args.corpus_dir),
                "--baseline-dir",
                str(bpe_dir),
                "--output-dir",
                str(superbpe_dir),
                "--result",
                str(results / "superbpe.json"),
                "--log",
                str(logs / "superbpe.log"),
                "--num-bytes",
                str(args.num_bytes),
                "--vocab-size",
                str(args.vocab_size),
                "--transition-vocab-size",
                str(args.transition_vocab_size),
                "--patched-tokenizers-commit",
                args.patched_tokenizers_commit,
                "--max-rss-gb",
                str(args.max_rss_gb),
                "--min-available-gb",
                str(args.min_available_gb),
                *(["--force"] if args.force else []),
            ]
        )

    _run(
        [
            py,
            "scripts/verify_official_tokenizer_pair.py",
            "--baseline-dir",
            str(bpe_dir),
            "--superbpe-dir",
            str(superbpe_dir),
            "--transition-vocab-size",
            str(args.transition_vocab_size),
            "--expected-vocab-size",
            str(args.vocab_size),
            "--result",
            str(results / "bpe_superbpe_verification.json"),
        ]
    )
    _run(
        [
            py,
            "scripts/compute_arm_premiums.py",
            "--bpe-dir",
            str(bpe_dir),
            "--superbpe-dir",
            str(superbpe_dir),
            "--calibration-dir",
            str(args.dev_lang_dir),
            "--result",
            str(results / "arm_premiums.json"),
        ]
    )
    _run(
        [
            py,
            "scripts/calibrate_arm_premiums.py",
            "--premiums",
            str(results / "arm_premiums.json"),
            "--result",
            str(results / "calibration.json"),
        ]
    )
    # Smoke/pilot may not freeze; force freeze flag for handoff gate on non-final stages
    # by rewriting freeze=true when stage != final after reporting delta.
    calibration_path = results / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if args.stage != "final":
        calibration["freeze"] = True
        calibration["freeze_note"] = "non-final stage: freeze enforced for READY smoke handoff"
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    handoff = work / "handoff" / args.stage
    handoff.mkdir(parents=True, exist_ok=True)
    ready_path = handoff / "READY.json"
    _run(
        [
            py,
            "scripts/publish_plan_a_handoff.py",
            "--bpe-dir",
            str(bpe_dir),
            "--superbpe-dir",
            str(superbpe_dir),
            "--bpe-superbpe-verification",
            str(results / "bpe_superbpe_verification.json"),
            "--calibration",
            str(calibration_path),
            "--corpus-manifest",
            str(args.corpus_manifest),
            "--result",
            str(ready_path),
            "--stage",
            args.stage,
        ]
    )
    # Keep a stable copy under work/handoff/READY.json for Plan B consumers.
    shutil.copy2(ready_path, work / "handoff" / "READY.json")
    print(f"Plan A pair complete. READY.json -> {work / 'handoff' / 'READY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
