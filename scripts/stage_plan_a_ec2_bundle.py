#!/usr/bin/env python3
"""Stage Plan A EC2 code + vendored gigatoken pin to landing scratch.

Uploads:
  $CORPUS_S3_ROOT/code/          — docker build context
  $CORPUS_S3_ROOT/vendor/supergigatoken-00e61db.tar.gz

Default CORPUS_S3_ROOT is the existing FineWeb scratch prefix. Does not touch
edullm-data or frozen airlock versions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = [
    Path(r"c:\Users\aryan\projects\supergigatoken"),
    Path.home() / "projects" / "supergigatoken",
    ROOT.parent / "supergigatoken",
]
PIN = "00e61db6e885aedd179ae34540caa6b561e3c185"
# edullm-downloader (EC2 instance profile) can Put/Get edullm-datasets, not
# edullm-landing. Keep research staging on the datasets bucket.
DEFAULT_ROOT = "s3://edullm-datasets/_scratch/plan-a-fineweb"
CODE_DIRS = ("src", "scripts", "compat", "configs", "docker", "vendor")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-s3-root",
        default=os.environ.get("CORPUS_S3_ROOT", DEFAULT_ROOT),
    )
    p.add_argument("--gigatoken-repo", type=Path, default=None)
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "sbsandbox"))
    p.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    p.add_argument("--skip-code", action="store_true")
    p.add_argument("--skip-vendor", action="store_true")
    return p.parse_args(argv)


def _aws(profile: str, region: str, *args: str) -> None:
    cmd = ["aws", *args, "--profile", profile, "--region", region]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _resolve_gigatoken(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    for cand in _CANDIDATES:
        if (cand / "Cargo.toml").is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "supergigatoken checkout not found; pass --gigatoken-repo"
    )


def _clone_pin(repo: Path, dest: Path) -> None:
    """Clone ``repo`` at PIN into ``dest`` (includes .git for pin verification)."""
    if subprocess.call(["git", "-C", str(repo), "cat-file", "-e", f"{PIN}^{{commit}}"]):
        raise RuntimeError(f"{repo} does not contain pin {PIN}")
    subprocess.check_call(["git", "clone", "--no-checkout", str(repo), str(dest)])
    subprocess.check_call(["git", "-C", str(dest), "checkout", "--detach", PIN])
    got = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
    ).strip()
    if got != PIN:
        raise RuntimeError(f"clone HEAD {got} != {PIN}")


def _make_tarball(clone: Path, dest: Path) -> None:
    exclude_dirs = {
        ".venv",
        "target",
        ".pytest_cache",
        ".ruff_cache",
        ".cursor",
        ".vscode",
        "__pycache__",
        "notebooks",
        "assets",
        ".cache",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        for path in clone.rglob("*"):
            rel = path.relative_to(clone)
            if any(p in exclude_dirs for p in rel.parts):
                continue
            if path.is_file() or path.is_symlink():
                tf.add(path, arcname=str(Path("gigatoken") / rel))
    print(f"Wrote {dest} ({dest.stat().st_size:,} bytes)", flush=True)


def _copytree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", ".venv", ".venv*"
        ),
    )


def _sync_code(root: str, profile: str, region: str) -> None:
    with tempfile.TemporaryDirectory(prefix="plan-a-code-") as tmp:
        staging = Path(tmp)
        for name in CODE_DIRS:
            src = ROOT / name
            if not src.exists():
                raise FileNotFoundError(src)
            _copytree(src, staging / name)
        (staging / "README.ec2.md").write_text(
            "Plan A EC2 docker build context. Build:\n"
            "  docker build -f docker/tokenizer-benchmark/Dockerfile "
            "-t tokenizer-benchmark .\n",
            encoding="utf-8",
        )
        _aws(
            profile,
            region,
            "s3",
            "sync",
            str(staging),
            f"{root.rstrip('/')}/code/",
            "--delete",
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.corpus_s3_root.rstrip("/")
    if not args.skip_vendor:
        repo = _resolve_gigatoken(args.gigatoken_repo)
        with tempfile.TemporaryDirectory(prefix="plan-a-vendor-") as tmp:
            tmp_path = Path(tmp)
            clone = tmp_path / "clone"
            _clone_pin(repo, clone)
            tar_path = tmp_path / "supergigatoken-00e61db.tar.gz"
            _make_tarball(clone, tar_path)
            _aws(
                args.profile,
                args.region,
                "s3",
                "cp",
                str(tar_path),
                f"{root}/vendor/supergigatoken-00e61db.tar.gz",
            )
    if not args.skip_code:
        _sync_code(root, args.profile, args.region)
    print(f"Staged bundle under {root}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
