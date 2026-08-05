#!/usr/bin/env python3
"""UniMax-sample Plan B pools and write matched ``.u32le.bin`` token trees.

Hold out trailing val docs per language **before** UniMax sampling, then draw
the same UTF-8 document stream for both arms (equal-bytes policy). Packs
uint32 little-endian shards under::

    <output-dir>/<arm>/tokens/<lang>/{train,val}-NNNNN.u32le.bin

Uses gigatoken ``Tokenizer.from_file`` (proven on the FLORES suite), not the
official patched fork rebuild.

``--budget-scale 1.0`` is the full mixture allocation; use ``0.001`` (or similar)
for laptop / smoke runs.

Streaming: train docs are not loaded into RAM — the pool is re-read for each
UniMax pass (needed for hat/swh epoch caps). Both arms encode lockstep so the
document byte stream is identical by construction.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json, sha256_file
from src.plan_a_langs import PLAN_A_CODES
from src.plan_b_pool_io import iter_docs

DEFAULT_POOLS = ROOT / "artifacts" / "plan_b" / "pools"
DEFAULT_MIXTURE = ROOT / "artifacts" / "plan_b" / "mixture.json"
DEFAULT_TOK_ROOT = ROOT / "artifacts" / "plan_a" / "scale" / "tokenizers"
DEFAULT_OUT = ROOT / "artifacts" / "plan_b" / "tokens"
DEFAULT_SHARD_BYTES = 512 * 1024 * 1024  # ~512 MiB of packed uint32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pools-dir", type=Path, default=DEFAULT_POOLS)
    p.add_argument("--mixture", type=Path, default=DEFAULT_MIXTURE)
    p.add_argument(
        "--tokenizers-root",
        type=Path,
        default=DEFAULT_TOK_ROOT,
        help="Root with bpe/ and superbpe/ dirs (each containing tokenizer.json)",
    )
    p.add_argument("--ready", type=Path, help="Optional READY.json; overrides tokenizers-root")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Multiply mixture allocation bytes (1.0 = full; smoke ≈ 0.001)",
    )
    p.add_argument("--val-docs", type=int, default=50)
    p.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    p.add_argument("--langs", nargs="*", default=list(PLAN_A_CODES))
    p.add_argument(
        "--arms",
        nargs="*",
        default=["bpe", "superbpe"],
        help="Tokenizer arms to materialize (default both)",
    )
    return p.parse_args(argv)


def _load_tokenizer(artifact_dir: Path):
    from tokenizers import Tokenizer

    path = artifact_dir / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return Tokenizer.from_file(str(path))


def _resolve_arm_dirs(
    *,
    tokenizers_root: Path,
    ready_path: Path | None,
) -> dict[str, Path]:
    if ready_path is not None:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("kind") != "plan_a_ready":
            raise ValueError(f"READY kind must be plan_a_ready, got {ready.get('kind')}")
        out: dict[str, Path] = {}
        for arm in ("bpe", "superbpe"):
            entry = ready.get("arms", {}).get(arm) or {}
            raw = entry.get("directory") or entry.get("path")
            if not raw:
                raise ValueError(f"READY.json missing arms.{arm}.directory")
            out[arm] = Path(raw)
        return out
    return {
        "bpe": tokenizers_root / "bpe",
        "superbpe": tokenizers_root / "superbpe",
    }


def holdout_val(
    lang_dir: Path,
    lang: str,
    *,
    val_docs: int,
) -> tuple[list[dict[str, str]], set[str], int, int]:
    """One streaming pass: trailing ``val_docs`` held out.

    Returns ``(val_list, val_ids, train_doc_count, train_utf8_bytes)``.
    Train documents are not retained — callers re-stream excluding ``val_ids``.
    """
    if val_docs < 1:
        raise ValueError("val_docs must be >= 1")
    buf: deque[dict[str, str]] = deque(maxlen=val_docs)
    train_docs = 0
    train_bytes = 0
    total = 0
    for doc in iter_docs(lang_dir, lang):
        total += 1
        if len(buf) == val_docs:
            oldest = buf[0]
            train_docs += 1
            train_bytes += len(oldest["text"].encode("utf-8"))
        buf.append(doc)
    if total <= val_docs:
        raise RuntimeError(
            f"{lang}: only {total} docs; need more than val_docs={val_docs}"
        )
    val_list = list(buf)
    val_ids = {d["id"] for d in val_list}
    return val_list, val_ids, train_docs, train_bytes


def iter_train_docs(
    lang_dir: Path, lang: str, val_ids: set[str]
) -> Iterator[dict[str, str]]:
    for doc in iter_docs(lang_dir, lang):
        if doc["id"] in val_ids:
            continue
        yield doc


def _pack_u32le(ids: list[int]) -> bytes:
    if not ids:
        return b""
    buf = array.array("I", ids)
    if sys.byteorder != "little":
        buf.byteswap()
    return buf.tobytes()


class U32ShardWriter:
    def __init__(
        self,
        out_dir: Path,
        split: str,
        *,
        max_bytes: int,
    ) -> None:
        self.out_dir = out_dir
        self.split = split
        self.max_bytes = max_bytes
        self.shard_idx = 0
        self.token_count = 0
        self.byte_count = 0
        self.paths: list[str] = []
        self._fh: Any = None
        self._path: Path | None = None
        self._shard_bytes = 0

    def _open_next(self) -> None:
        self.close()
        self._path = self.out_dir / f"{self.split}-{self.shard_idx:05d}.u32le.bin"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("wb")
        self.paths.append(self._path.name)
        self.shard_idx += 1
        self._shard_bytes = 0

    def write_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        if self._fh is None:
            self._open_next()
        assert self._fh is not None
        payload = _pack_u32le(ids)
        self._fh.write(payload)
        n = len(ids)
        self.token_count += n
        self.byte_count += len(payload)
        self._shard_bytes += len(payload)
        if self.split == "train" and self._shard_bytes >= self.max_bytes:
            self._open_next()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def tokenize_docs(
    docs: list[dict[str, str]],
    encode: Callable[[str], list[int]],
    writer: U32ShardWriter,
) -> tuple[int, int]:
    utf8 = 0
    tokens = 0
    for doc in docs:
        text = doc["text"]
        utf8 += len(text.encode("utf-8"))
        ids = encode(text)
        tokens += len(ids)
        writer.write_ids(ids)
    return utf8, tokens


def unimax_tokenize_train(
    lang_dir: Path,
    lang: str,
    *,
    val_ids: set[str],
    target_bytes: int,
    passes_over_pool: float,
    encoders: dict[str, Callable[[str], list[int]]],
    writers: dict[str, U32ShardWriter],
) -> tuple[int, int, dict[str, int], float]:
    """Stream UniMax train draw; encode lockstep into per-arm writers.

    Returns ``(utf8_bytes, n_docs, tokens_by_arm, passes_used)``.
    """
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    max_passes = max(1, int(math.ceil(passes_over_pool - 1e-12)))
    got = 0
    n_docs = 0
    tokens_by_arm = {arm: 0 for arm in writers}
    passes_used = 0.0
    pool_bytes_seen = 0

    for pass_i in range(max_passes):
        if got >= target_bytes:
            break
        for doc in iter_train_docs(lang_dir, lang, val_ids):
            text = doc["text"]
            raw_n = len(text.encode("utf-8"))
            if pass_i == 0:
                pool_bytes_seen += raw_n
            if got >= target_bytes:
                break
            for arm, encode in encoders.items():
                ids = encode(text)
                tokens_by_arm[arm] += len(ids)
                writers[arm].write_ids(ids)
            got += raw_n
            n_docs += 1
            if got >= target_bytes:
                break
        passes_used = float(pass_i + 1)

    if got < 1:
        raise RuntimeError(f"{lang}: UniMax sample produced zero bytes")
    if pool_bytes_seen > 0:
        passes_used = got / pool_bytes_seen
    return got, n_docs, tokens_by_arm, passes_used


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_scale <= 0:
        raise SystemExit("--budget-scale must be positive")
    if args.val_docs < 1:
        raise SystemExit("--val-docs must be >= 1")

    mixture = json.loads(args.mixture.read_text(encoding="utf-8"))
    if mixture.get("kind") != "plan_b_mixture":
        raise SystemExit(f"expected plan_b_mixture, got {mixture.get('kind')}")
    allocation = mixture["allocation"]

    arm_dirs = _resolve_arm_dirs(
        tokenizers_root=args.tokenizers_root, ready_path=args.ready
    )
    for arm in args.arms:
        if arm not in arm_dirs:
            raise SystemExit(f"unknown arm {arm!r}")
        if not (arm_dirs[arm] / "tokenizer.json").is_file():
            raise SystemExit(f"missing tokenizer.json under {arm_dirs[arm]}")

    tokenizers = {arm: _load_tokenizer(arm_dirs[arm]) for arm in args.arms}

    def make_encode(arm: str) -> Callable[[str], list[int]]:
        tok = tokenizers[arm]

        def encode(text: str) -> list[int]:
            return tok.encode(text).ids

        return encode

    encoders = {arm: make_encode(arm) for arm in args.arms}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "kind": "plan_b_tokenize_pools",
        "schema_version": 1,
        "budget_scale": args.budget_scale,
        "val_docs": args.val_docs,
        "mixture": str(args.mixture.resolve()),
        "pools_dir": str(args.pools_dir.resolve()),
        "arms": {
            arm: {
                "tokenizer_directory": str(arm_dirs[arm].resolve()),
                "tokenizer_json_sha256": sha256_file(arm_dirs[arm] / "tokenizer.json"),
                "langs": {},
                "total_train_tokens": 0,
                "total_train_utf8_bytes": 0,
                "total_val_tokens": 0,
                "total_val_utf8_bytes": 0,
            }
            for arm in args.arms
        },
        "languages": {},
    }

    for lang in args.langs:
        lang_dir = args.pools_dir / lang
        if not lang_dir.is_dir():
            raise SystemExit(f"missing pool dir: {lang_dir}")
        alloc = allocation.get(lang)
        if not alloc:
            raise SystemExit(f"mixture missing allocation for {lang}")
        target = int(math.floor(float(alloc["bytes"]) * args.budget_scale))
        if target < 1:
            target = 1
        passes = float(alloc.get("passes_over_pool", 1.0))

        val_list, val_ids, train_pool_docs, train_pool_bytes = holdout_val(
            lang_dir, lang, val_docs=args.val_docs
        )

        train_writers = {
            arm: U32ShardWriter(
                args.output_dir / arm / "tokens" / lang,
                "train",
                max_bytes=args.shard_bytes,
            )
            for arm in args.arms
        }
        val_writers = {
            arm: U32ShardWriter(
                args.output_dir / arm / "tokens" / lang,
                "val",
                max_bytes=args.shard_bytes,
            )
            for arm in args.arms
        }

        try:
            # Val first (same docs → both arms).
            val_utf8 = 0
            val_tokens_by_arm = {arm: 0 for arm in args.arms}
            for doc in val_list:
                text = doc["text"]
                val_utf8 += len(text.encode("utf-8"))
                for arm, encode in encoders.items():
                    ids = encode(text)
                    val_tokens_by_arm[arm] += len(ids)
                    val_writers[arm].write_ids(ids)

            sampled_bytes, n_docs, train_tokens_by_arm, passes_used = (
                unimax_tokenize_train(
                    lang_dir,
                    lang,
                    val_ids=val_ids,
                    target_bytes=target,
                    passes_over_pool=passes,
                    encoders=encoders,
                    writers=train_writers,
                )
            )
        finally:
            for w in train_writers.values():
                w.close()
            for w in val_writers.values():
                w.close()

        report["languages"][lang] = {
            "allocation_bytes": float(alloc["bytes"]),
            "target_bytes": target,
            "train_pool_docs": train_pool_docs,
            "train_pool_bytes": train_pool_bytes,
            "val_docs": len(val_list),
            "sampled_docs": n_docs,
            "sampled_bytes": sampled_bytes,
            "passes_over_pool_declared": passes,
            "passes_used": passes_used,
            "val_ids": [d["id"] for d in val_list],
        }

        for arm in args.arms:
            tw = train_writers[arm]
            vw = val_writers[arm]
            if tw.byte_count % 4 != 0 or vw.byte_count % 4 != 0:
                raise RuntimeError(f"{arm}/{lang}: packed size not divisible by 4")
            report["arms"][arm]["langs"][lang] = {
                "train_utf8_bytes": sampled_bytes,
                "train_tokens": train_tokens_by_arm[arm],
                "train_file_bytes": tw.byte_count,
                "train_shards": tw.paths,
                "val_utf8_bytes": val_utf8,
                "val_tokens": val_tokens_by_arm[arm],
                "val_file_bytes": vw.byte_count,
                "val_shards": vw.paths,
            }
            report["arms"][arm]["total_train_tokens"] += train_tokens_by_arm[arm]
            report["arms"][arm]["total_train_utf8_bytes"] += sampled_bytes
            report["arms"][arm]["total_val_tokens"] += val_tokens_by_arm[arm]
            report["arms"][arm]["total_val_utf8_bytes"] += val_utf8
            print(
                f"  {arm}/{lang}: train_tokens={train_tokens_by_arm[arm]} "
                f"val_tokens={val_tokens_by_arm[arm]}",
                flush=True,
            )

        print(
            f"{lang}: sample {sampled_bytes} bytes "
            f"({n_docs} docs, passes≈{passes_used:.3f}) val={len(val_list)}",
            flush=True,
        )

    if len(args.arms) >= 2:
        ref = args.arms[0]
        ref_bytes = report["arms"][ref]["total_train_utf8_bytes"]
        for arm in args.arms[1:]:
            other = report["arms"][arm]["total_train_utf8_bytes"]
            if other != ref_bytes:
                raise SystemExit(
                    f"equal-bytes violated: {ref}={ref_bytes} vs {arm}={other}"
                )
        report["equal_bytes_train_utf8"] = ref_bytes
        report["equal_bytes_ok"] = True

    out_manifest = args.output_dir / "tokenize_manifest.json"
    atomic_write_json(out_manifest, report)
    print(
        json.dumps(
            {
                "equal_bytes_ok": report.get("equal_bytes_ok"),
                "equal_bytes_train_utf8": report.get("equal_bytes_train_utf8"),
                "arms": {
                    a: {
                        "total_train_tokens": report["arms"][a]["total_train_tokens"],
                        "total_train_utf8_bytes": report["arms"][a][
                            "total_train_utf8_bytes"
                        ],
                    }
                    for a in args.arms
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
