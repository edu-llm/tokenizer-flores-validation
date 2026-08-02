#!/usr/bin/env python3
"""Train one Plan A arm with supergigatoken, emitting the official artifact set.

A **sibling** of ``run_official_tokenizer_benchmark.py``, not a replacement.
The official entrypoint stays pinned and runnable: the smoke-tier cross-check
trains both trainers on the same corpus and compares per-language premiums, and
that comparison is the gate for trusting anything downstream.

Both entrypoints write ``vocab.json``, ``merges.txt``, ``tokenizer.json``,
``meta.json`` and (for the SuperBPE arm) ``requested_initial_merges.txt``, so
``verify_official_tokenizer_pair.py``, ``eval_plan_a_flores_compression.py``
and ``calibrate_arm_premiums.py`` work against either without branching.

Two differences from the official path are deliberate and load-bearing:

* **Corpus shape.** gigatoken takes a single mmapped file, not a ``corpus_dir``
  of ``*.txt``. The official trainer's ``get_files_with_num_bytes`` whole-file
  selection hazard therefore cannot arise here -- but the equal-byte guarantee
  moves entirely into corpus construction, so this script verifies the corpus
  manifest instead of asserting on ``meta.json:train_files``.
* **Stage-1 scheme.** Passed explicitly (default ``superbpe_stage1``, the
  official ``STAGE1_REGEX``). gigatoken's own default is ``gpt2``, whose
  letter runs exclude ``\\p{M}``; that fragments Devanagari and, because
  gigatoken's stage 2 applies no regex at all, lets stage 2 repair the damage
  in the SuperBPE arm only -- inflating that arm's apparent gain for exactly
  the scripts using combining marks. ``--pretokenizer gpt2`` is still
  accepted so the cross-check can measure that effect deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_official_tokenizer_benchmark import STAGE1_REGEX
from src.benchmark import run_monitored_command, sha256_file

WORKER = Path(__file__).resolve().parent / "_gigatoken_train_worker.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("bpe", "superbpe"), required=True)
    p.add_argument(
        "--gigatoken-repo",
        type=Path,
        required=True,
        help="supergigatoken checkout; its commit is recorded as provenance",
    )
    p.add_argument(
        "--gigatoken-python",
        type=Path,
        help="Python with gigatoken importable. Defaults to the checkout's .venv.",
    )
    p.add_argument(
        "--corpus-file",
        type=Path,
        required=True,
        help="Single concatenated train.txt from build_plan_a_research_corpus.py",
    )
    p.add_argument(
        "--corpus-manifest",
        type=Path,
        help="Corpus manifest.json; defaults to a sibling of --corpus-file. "
        "Verified for equal bytes -- the gigatoken path's replacement for the "
        "official trainer's meta.json:train_files assertion.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument(
        "--transition-vocab-size",
        type=int,
        help="Required for SuperBPE; exact vocabulary size inherited from stage 1",
    )
    p.add_argument(
        "--expected-commit",
        help="Hard-fail unless the checkout is at this commit. Defaults to "
        "gigatoken_pipeline.commit from --config; pass 'any' to skip.",
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "benchmarks" / "tokenizer_local.json")
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit an uncommitted worktree. Off by default: a dirty tree "
        "means the recorded commit does not describe the code that ran.",
    )
    p.add_argument("--pretokenizer", default="superbpe_stage1")
    p.add_argument("--separator", default="\n")
    p.add_argument("--max-unit-len", type=int, default=128)
    p.add_argument("--projection-bytes", type=int, default=10_000_000_000)
    p.add_argument("--max-rss-gb", type=float)
    p.add_argument("--min-available-gb", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty(repo: Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists (use --force): {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def verify_corpus_manifest(manifest_path: Path, corpus_file: Path) -> dict:
    """Equal-byte check, replacing the official path's train_files assertion."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sizes = {
        code: int(meta["bytes"]) for code, meta in manifest["per_language"].items()
    }
    if not sizes:
        raise ValueError(f"{manifest_path} lists no languages")
    if max(sizes.values()) != min(sizes.values()):
        raise ValueError(
            f"Corpus is not equal-byte, refusing to train on a skewed mix: {sizes}"
        )
    actual = corpus_file.stat().st_size
    if actual != int(manifest["total_bytes"]):
        raise ValueError(
            f"{corpus_file} is {actual:,} bytes but the manifest says "
            f"{manifest['total_bytes']:,}; corpus and manifest disagree."
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.gigatoken_repo.resolve()
    if not (repo / "Cargo.toml").is_file():
        raise FileNotFoundError(f"Not a gigatoken checkout: {repo}")

    commit = _git_commit(repo)
    dirty = _git_dirty(repo)

    # Provenance is the whole point of pinning: a tier trained on unpinned or
    # uncommitted code cannot be reproduced or compared against another tier.
    expected = args.expected_commit
    if expected is None and args.config.is_file():
        config = json.loads(args.config.read_text(encoding="utf-8"))
        expected = config.get("gigatoken_pipeline", {}).get("commit")
    if expected and expected != "any" and expected != "UNPINNED":
        if commit != expected:
            raise RuntimeError(
                f"Expected gigatoken commit {expected}, got {commit}. Bumping "
                "the pin invalidates completed tiers -- re-run "
                "scripts/compare_plan_a_trainers.py before changing it."
            )
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            f"{repo} has uncommitted changes, so commit {commit} does not "
            "describe the code that would run. Commit them, or pass "
            "--allow-dirty for a throwaway experiment."
        )

    interpreter = args.gigatoken_python or (repo / ".venv" / "Scripts" / "python.exe")
    if not Path(interpreter).is_file():
        raise FileNotFoundError(
            f"No gigatoken interpreter at {interpreter}; pass --gigatoken-python"
        )

    corpus_file = args.corpus_file.resolve()
    if not corpus_file.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    manifest_path = args.corpus_manifest or (corpus_file.parent / "manifest.json")
    corpus_manifest = verify_corpus_manifest(Path(manifest_path).resolve(), corpus_file)
    input_bytes = corpus_file.stat().st_size

    if args.vocab_size <= 256:
        raise ValueError("vocab-size must exceed 256")
    if args.arm == "superbpe":
        if args.transition_vocab_size is None:
            raise ValueError("SuperBPE requires --transition-vocab-size")
        if args.transition_vocab_size >= args.vocab_size:
            raise ValueError("transition-vocab-size must be smaller than vocab-size")

    output_dir = args.output_dir.resolve()
    _prepare_output(output_dir, args.force)

    spec = {
        "arm": args.arm,
        "output_dir": str(output_dir),
        "corpus_file": str(corpus_file),
        "vocab_size": args.vocab_size,
        "transition_vocab_size": args.transition_vocab_size,
        "pretokenizer": args.pretokenizer,
        "separator": args.separator,
        "max_unit_len": args.max_unit_len,
        "stage1_regex": STAGE1_REGEX,
    }

    metadata = {
        "trainer": "gigatoken",
        "arm": args.arm,
        "vocab_size": args.vocab_size,
        "transition_vocab_size": args.transition_vocab_size,
        "gigatoken_commit": commit,
        # A dirty checkout means the recorded commit does not describe the code
        # that ran; surface it rather than silently pinning a lie.
        "gigatoken_worktree_dirty": dirty,
        "pretokenizer": args.pretokenizer,
        "separator": args.separator,
        "max_unit_len": args.max_unit_len,
        "stage1_regex_sha256": hashlib.sha256(
            STAGE1_REGEX.encode("utf-8")
        ).hexdigest(),
        "corpus_sha256": sha256_file(corpus_file),
        "corpus_manifest": {
            "path": str(manifest_path),
            "bytes_per_language": corpus_manifest.get("bytes_per_language"),
            "total_bytes": corpus_manifest.get("total_bytes"),
            "languages": corpus_manifest.get("languages"),
            "equal_bytes_verified": True,
        },
        "stage2_note": (
            "gigatoken stage 2 applies no regex: units are bounded by "
            "separator, newline and max_unit_len. Its superword space is a "
            "strict superset of the official STAGE2_REGEX."
        ),
    }

    result = run_monitored_command(
        name=f"gigatoken-{args.arm}-{args.vocab_size}-{args.pretokenizer}",
        command=[str(interpreter), str(WORKER), json.dumps(spec)],
        result_path=args.result,
        log_path=args.log,
        input_bytes=input_bytes,
        projection_bytes=args.projection_bytes,
        cwd=repo,
        output_path=output_dir,
        max_rss_gb=args.max_rss_gb,
        min_available_gb=args.min_available_gb,
        metadata=metadata,
    )
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
