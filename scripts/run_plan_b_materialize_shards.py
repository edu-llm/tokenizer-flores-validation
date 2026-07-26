#!/usr/bin/env python3
"""Materialize matched OLMo shard manifests for BPE / SuperBPE / Parity.

Reads Plan A READY.json, tokenizes the same ordered documents with each arm,
and writes aligned shard JSONL manifests. Runs unchanged locally or on AWS Batch
after S3 staging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import atomic_write_json, sha256_file
from src.official_bpe_encode import load_official_bpe_tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="Ordered UTF-8 text file, one document per line",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--shard-size", type=int, default=1000)
    return parser.parse_args(argv)


def _load_ready(path: Path) -> dict:
    ready = json.loads(path.read_text(encoding="utf-8"))
    if ready.get("kind") != "plan_a_ready":
        raise ValueError(f"Expected plan_a_ready, got {ready.get('kind')}")
    for arm in ("bpe", "superbpe", "parity"):
        if arm not in ready.get("arms", {}):
            raise ValueError(f"READY.json missing arm {arm}")
    return ready


def _iter_documents(path: Path, max_documents: int | None):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\r\n")
            if not text:
                continue
            yield text
            count += 1
            if max_documents is not None and count >= max_documents:
                break


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ready = _load_ready(args.ready)
    documents = list(_iter_documents(args.documents, args.max_documents))
    if not documents:
        raise ValueError("No documents found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc_manifest = {
        "schema_version": 1,
        "kind": "plan_b_document_manifest",
        "n_documents": len(documents),
        "documents_sha256": hashlib.sha256(
            "\n".join(documents).encode("utf-8")
        ).hexdigest(),
        "source_uri": args.documents.resolve().as_uri(),
        "ready_sha256": sha256_file(args.ready),
    }
    atomic_write_json(args.output_dir / "documents.json", doc_manifest)

    summary_arms: dict[str, dict] = {}
    for arm in ("bpe", "superbpe", "parity"):
        artifact_dir = Path(ready["arms"][arm]["directory"])
        tokenizer = load_official_bpe_tokenizer(artifact_dir)
        arm_dir = args.output_dir / "shards" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        shard_rows: list[dict] = []
        total_tokens = 0
        total_bytes = 0
        written_shards: list[str] = []

        def flush() -> None:
            nonlocal shard_idx, shard_rows
            if not shard_rows:
                return
            shard_path = arm_dir / f"shard_{shard_idx:05d}.jsonl"
            with shard_path.open("w", encoding="utf-8") as handle:
                for row in shard_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_shards.append(shard_path.name)
            shard_idx += 1
            shard_rows = []

        for doc_id, text in enumerate(documents):
            ids = tokenizer.encode(text).ids
            raw_bytes = len(text.encode("utf-8"))
            total_tokens += len(ids)
            total_bytes += raw_bytes
            shard_rows.append(
                {
                    "doc_id": doc_id,
                    "n_tokens": len(ids),
                    "n_bytes": raw_bytes,
                    "token_ids": ids,
                    "tokenizer_arm": arm,
                    "tokenizer_vocab_sha256": ready["arms"][arm]["vocab_sha256"],
                }
            )
            if len(shard_rows) >= args.shard_size:
                flush()
        flush()

        arm_summary = {
            "arm": arm,
            "n_documents": len(documents),
            "n_tokens": total_tokens,
            "n_bytes": total_bytes,
            "bytes_per_token": (total_bytes / total_tokens) if total_tokens else None,
            "shards": written_shards,
            "tokenizer_directory": str(artifact_dir),
            "tokenizer_vocab_sha256": ready["arms"][arm]["vocab_sha256"],
        }
        atomic_write_json(arm_dir / "summary.json", arm_summary)
        summary_arms[arm] = arm_summary

    # Byte-matched context suggestion relative to BPE 2048.
    bpe_bpt = summary_arms["bpe"]["bytes_per_token"] or 1.0
    context = {"bpe_context_tokens": 2048}
    for arm in ("superbpe", "parity"):
        bpt = summary_arms[arm]["bytes_per_token"] or bpe_bpt
        context[f"{arm}_context_tokens"] = max(1, int(round(2048 * bpe_bpt / bpt)))

    overview = {
        "schema_version": 1,
        "kind": "plan_b_materialization",
        "ready": str(args.ready.resolve()),
        "arms": summary_arms,
        "context_tokens_byte_matched": context,
        "equal_byte_checkpoint": "compare after identical raw document bytes",
        "equal_flop_baseline": "bpe equal-byte endpoint",
    }
    atomic_write_json(args.output_dir / "materialization.json", overview)
    print(json.dumps(overview, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
