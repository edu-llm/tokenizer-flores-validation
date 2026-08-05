#!/usr/bin/env python3
"""Rebuild ``train.txt`` from per-language ``langs/*.txt`` + manifest order.

The landing scratch stash for scale ships ``langs/`` + ``manifest.json`` but not
always ``train.txt`` (gigatoken needs the concatenation). Reconstruct on the
EC2 host from the manifest's ``train_txt.order`` and verify against
``train_txt.sha256`` before training.

Example (on the instance, after syncing scratch):

    python scripts/materialize_plan_a_train_txt.py \\
        --corpus-dir /data/plan-a/corpus/scale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory with langs/ and manifest.json (train.txt written beside them)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing train.txt even if the hash already matches",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = args.corpus_dir
    manifest_path = corpus / "manifest.json"
    langs = corpus / "langs"
    train_txt = corpus / "train.txt"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = manifest["train_txt"]["order"]
    expected_sha = manifest["train_txt"]["sha256"]
    expected_total = int(manifest["total_bytes"])

    if train_txt.is_file() and not args.force:
        digest = hashlib.sha256(train_txt.read_bytes()).hexdigest()
        if digest == expected_sha and train_txt.stat().st_size == expected_total:
            print(f"OK: {train_txt} already matches manifest sha256")
            return 0
        raise SystemExit(
            f"{train_txt} exists but does not match manifest "
            f"(size={train_txt.stat().st_size}, sha256={digest}); "
            "pass --force to rebuild"
        )

    digest = hashlib.sha256()
    written = 0
    with train_txt.open("wb") as out:
        for code in order:
            path = langs / f"{code}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            chunk = path.read_bytes()
            out.write(chunk)
            digest.update(chunk)
            written += len(chunk)

    got = digest.hexdigest()
    if written != expected_total or got != expected_sha:
        train_txt.unlink(missing_ok=True)
        raise SystemExit(
            f"reconstructed train.txt mismatch: wrote {written} sha256={got}; "
            f"manifest expects {expected_total} sha256={expected_sha}"
        )

    print(f"Wrote {train_txt} ({written} bytes, sha256={got})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
