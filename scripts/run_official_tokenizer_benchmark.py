#!/usr/bin/env python3
"""Benchmark official-pipeline BPE or SuperBPE with resource telemetry.

The script is platform-neutral and is intended to run unchanged on a laptop or
inside AWS Batch. SuperBPE continuation is refused unless the patched
``tokenizers`` commit is explicitly recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import run_monitored_command, sha256_file

OFFICIAL_SUPERBPE_COMMIT = "bbd09768fc28a875cef48e6bdd66e3a17454628e"
OFFICIAL_TOKENIZERS_COMMIT = "757f2a55c0820ed47064e1fe473deea39b7b611b"

STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)
STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("bpe", "superbpe"), required=True)
    parser.add_argument("--superbpe-repo", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--num-bytes", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument(
        "--transition-vocab-size",
        type=int,
        help="Required for SuperBPE; exact vocabulary size inherited from BPE",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Required for SuperBPE; completed BPE artifact with vocab/merges/meta",
    )
    parser.add_argument("--projection-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--max-rss-gb", type=float)
    parser.add_argument("--min-available-gb", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--patched-tokenizers-commit",
        default=os.environ.get("SUPERBPE_TOKENIZERS_COMMIT"),
        help="Provenance guard; required for SuperBPE continuation",
    )
    return parser.parse_args(argv)


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _prepare_output(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output already exists (use --force): {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _load_merges(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty merges file: {path}")
    if lines[0].startswith("#"):
        return lines[0], [line for line in lines[1:] if line]
    return "#version: 0.2", [line for line in lines if line]


def _prepare_superbpe_prefix(
    baseline_dir: Path,
    output_dir: Path,
    transition_vocab_size: int,
) -> dict[str, int]:
    required = ("vocab.json", "merges.txt", "meta.json")
    missing = [name for name in required if not (baseline_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Baseline is missing {missing}: {baseline_dir}")

    vocab = json.loads((baseline_dir / "vocab.json").read_text(encoding="utf-8"))
    header, merges = _load_merges(baseline_dir / "merges.txt")
    initial_vocab_size = len(vocab) - len(merges)
    inherited_merges = transition_vocab_size - initial_vocab_size
    if inherited_merges <= 0 or inherited_merges > len(merges):
        raise ValueError(
            "transition vocab must inherit between 1 and all baseline merges; "
            f"initial={initial_vocab_size}, transition={transition_vocab_size}, "
            f"available_merges={len(merges)}"
        )

    selected = merges[:inherited_merges]
    selected_text = "\n".join([header, *selected]) + "\n"
    (output_dir / "merges.txt").write_text(
        selected_text,
        encoding="utf-8",
    )
    # The official training script copies merges.txt with the Unix `cp`
    # command, which is unavailable on native Windows. Keep a portable,
    # immutable record of the exact prefix requested for continuation.
    (output_dir / "requested_initial_merges.txt").write_text(
        selected_text,
        encoding="utf-8",
    )
    shutil.copy2(baseline_dir / "meta.json", output_dir / "meta.json")
    return {
        "baseline_vocab_size": len(vocab),
        "initial_vocab_size": initial_vocab_size,
        "transition_vocab_size": transition_vocab_size,
        "inherited_merges": inherited_merges,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.superbpe_repo.resolve()
    if not (repo / "train_tokenizer.py").is_file():
        raise FileNotFoundError(f"Not an official SuperBPE checkout: {repo}")
    repo_commit = _git_commit(repo)
    if repo_commit != OFFICIAL_SUPERBPE_COMMIT:
        raise RuntimeError(
            f"Expected SuperBPE commit {OFFICIAL_SUPERBPE_COMMIT}, got {repo_commit}"
        )
    if args.num_bytes <= 0 or args.vocab_size <= 256:
        raise ValueError("num-bytes must be positive and vocab-size must exceed 256")
    if args.patched_tokenizers_commit != OFFICIAL_TOKENIZERS_COMMIT:
        raise RuntimeError(
            "Both arms require the pinned patched tokenizers fork; set "
            f"--patched-tokenizers-commit {OFFICIAL_TOKENIZERS_COMMIT}"
        )

    output_dir = args.output_dir.resolve()
    _prepare_output(output_dir, args.force)
    prefix_metadata: dict[str, int] | None = None

    if args.arm == "superbpe":
        if args.baseline_dir is None or args.transition_vocab_size is None:
            raise ValueError("SuperBPE requires --baseline-dir and --transition-vocab-size")
        if args.transition_vocab_size >= args.vocab_size:
            raise ValueError("transition-vocab-size must be smaller than vocab-size")
        prefix_metadata = _prepare_superbpe_prefix(
            args.baseline_dir.resolve(),
            output_dir,
            args.transition_vocab_size,
        )
        regex = STAGE2_REGEX
    else:
        regex = STAGE1_REGEX

    command = [
        sys.executable,
        "-m",
        "train_tokenizer",
        "--output_dir",
        str(output_dir),
        "--vocab_size",
        str(args.vocab_size),
        "--regex_string",
        regex,
    ]
    if args.arm == "bpe":
        command.extend(
            [
                "--corpus_dir",
                str(args.corpus_dir.resolve()),
                "--num_bytes",
                str(args.num_bytes),
            ]
        )

    metadata = {
        "arm": args.arm,
        "vocab_size": args.vocab_size,
        "transition": prefix_metadata,
        "superbpe_commit": repo_commit,
        "patched_tokenizers_commit": args.patched_tokenizers_commit,
        "regex_sha256": hashlib.sha256(regex.encode("utf-8")).hexdigest(),
        "baseline_merges_sha256": (
            sha256_file(args.baseline_dir / "merges.txt")
            if args.baseline_dir is not None
            else None
        ),
    }
    pythonpath = [str(repo)]
    compatibility_shim = None
    if importlib.util.find_spec("simdjson") is None:
        compatibility_dir = ROOT / "compat" / "superbpe"
        pythonpath.insert(0, str(compatibility_dir))
        compatibility_shim = "stdlib-json-for-pysimdjson-metadata-only"
    metadata["compatibility_shim"] = compatibility_shim

    result = run_monitored_command(
        name=f"official-{args.arm}-{args.vocab_size}",
        command=command,
        result_path=args.result,
        log_path=args.log,
        input_bytes=args.num_bytes,
        projection_bytes=args.projection_bytes,
        cwd=repo,
        output_path=output_dir,
        max_rss_gb=args.max_rss_gb,
        min_available_gb=args.min_available_gb,
        metadata=metadata,
        env={"PYTHONPATH": os.pathsep.join(pythonpath)},
    )
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

