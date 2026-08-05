#!/usr/bin/env python3
"""Pull Plan B unique text pools as sharded ``part-*.jsonl.zst``.

Streams FineWeb / FineWeb-2 (and the fallback ladder for availability-bound
languages), preserves one JSON document per line, deduplicates across rungs by
text digest, and writes ``artifacts/plan_b/pools/<lang>/`` plus
``manifest.json``.

Default per-language byte targets come from UniMax planning acquisition
(``src.plan_b_mixture.acquire_pool_bytes``) — ~74 GB total with 15% headroom on
unbounded languages. Run on EC2 in the training region; do not pull 65 GB over
a residential link.

Requires: ``datasets``, ``zstandard``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plan_a_langs import PLAN_A_CODES, SOURCES
from src.plan_b_mixture import (
    PLANNING_AVAILABLE_BYTES,
    PLANNING_EPOCH_CAP,
    acquire_pool_bytes,
    planning_budget_bytes,
    unimax_allocation,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

OUT = ROOT / "artifacts" / "plan_b" / "pools"
DEFAULT_SHARD_BYTES = 256 * 1024 * 1024  # uncompressed UTF-8 text budget per part

# Fallback ladder (plans/01-data-sourcing.md §3). Rung 1 is always SOURCES.
FALLBACK_LADDER: dict[str, list[tuple[str, str | None]]] = {
    "hat_Latn": [
        ("HPLT/HPLT2.0_cleaned", "hat_Latn"),
        ("cis-lmu/GlotCC-V1", "hat-Latn"),
        ("wikimedia/wikipedia", "20231101.ht"),
    ],
    "swh_Latn": [
        ("HPLT/HPLT2.0_cleaned", "swh_Latn"),
        ("cis-lmu/GlotCC-V1", "swh-Latn"),
        ("wikimedia/wikipedia", "20231101.sw"),
        ("Adeptschneider/CiviVox-Swahili-text-corpus-v2.0", None),
    ],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--langs", nargs="*", default=list(PLAN_A_CODES))
    p.add_argument(
        "--targets-json",
        type=Path,
        help="Optional {lang: bytes} override; default is UniMax acquire targets",
    )
    p.add_argument(
        "--shard-bytes",
        type=int,
        default=DEFAULT_SHARD_BYTES,
        help="Rotate part-NNNNN.jsonl.zst after this many UTF-8 text bytes",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a language if its dir already has part-*.jsonl.zst",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help=(
            "Top up existing part-*.jsonl.zst toward --targets-json absolute byte "
            "totals without deleting prior shards (dedupe by digest across old+new)"
        ),
    )
    p.add_argument(
        "--no-ladder",
        action="store_true",
        help="Do not walk fallback rungs when FineWeb runs short",
    )
    return p.parse_args(argv)


def _default_targets() -> dict[str, int]:
    budget = planning_budget_bytes()
    alloc = unimax_allocation(PLANNING_AVAILABLE_BYTES, budget, PLANNING_EPOCH_CAP)
    acquire = acquire_pool_bytes(alloc, PLANNING_AVAILABLE_BYTES)
    return {lang: int(round(n)) for lang, n in acquire.items()}


def _text_from_row(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    for key in ("text", "content", "raw_content", "article"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def _doc_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stream_split(dataset_id: str, subset: str | None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": "train", "streaming": True}
    if subset:
        ds = load_dataset(dataset_id, name=subset, **kwargs)
    else:
        ds = load_dataset(dataset_id, **kwargs)
    for row in ds:
        yield row  # type: ignore[misc]


class _ShardWriter:
    def __init__(
        self,
        lang_dir: Path,
        *,
        shard_bytes: int,
        start_part_idx: int = 0,
        initial_bytes: int = 0,
        initial_docs: int = 0,
        initial_shards: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            import zstandard as zstd
        except ImportError as e:
            raise SystemExit(
                "pull_plan_b_pools.py requires 'zstandard' (pip install zstandard)"
            ) from e
        self._zstd = zstd
        self.lang_dir = lang_dir
        self.shard_bytes = shard_bytes
        self.lang_dir.mkdir(parents=True, exist_ok=True)
        self.part_idx = start_part_idx
        self.bytes_written = initial_bytes
        self.docs_written = initial_docs
        self.shard_text_bytes = 0
        self.shards: list[dict[str, Any]] = list(initial_shards or [])
        self._fh: Any = None
        self._compressor: Any = None
        self._path: Path | None = None
        self._buf = io.BytesIO()

    def _open_next(self) -> None:
        self.close_part()
        self._path = self.lang_dir / f"part-{self.part_idx:05d}.jsonl.zst"
        self._fh = self._path.open("wb")
        self._compressor = self._zstd.ZstdCompressor(level=3).stream_writer(self._fh)
        self.part_idx += 1
        self.shard_text_bytes = 0

    def write(self, obj: dict[str, Any], text_bytes: int) -> None:
        if self._compressor is None:
            self._open_next()
        assert self._compressor is not None
        line = (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        self._compressor.write(line)
        self.bytes_written += text_bytes
        self.docs_written += 1
        self.shard_text_bytes += text_bytes
        if self.shard_text_bytes >= self.shard_bytes:
            self.close_part()

    def close_part(self) -> None:
        if self._compressor is None or self._path is None:
            return
        self._compressor.close()
        self._fh.close()
        self.shards.append(
            {
                "path": self._path.name,
                "sha256": sha256_file(self._path),
                "text_bytes": self.shard_text_bytes,
            }
        )
        self._compressor = None
        self._fh = None
        self._path = None

    def close(self) -> None:
        self.close_part()


def _rungs_for(lang: str, *, use_ladder: bool) -> list[tuple[str, str | None, int]]:
    """Return [(dataset_id, subset, rung_number), ...] starting at rung 1."""
    ds, cfg = SOURCES[lang]
    rungs = [(ds, cfg, 1)]
    if use_ladder:
        for i, (did, subset) in enumerate(FALLBACK_LADDER.get(lang, []), start=2):
            rungs.append((did, subset, i))
    return rungs


def pull_language(
    lang: str,
    dest_dir: Path,
    *,
    target_bytes: int,
    shard_bytes: int,
    use_ladder: bool,
    seen: set[str] | None = None,
    start_part_idx: int = 0,
    initial_bytes: int = 0,
    initial_docs: int = 0,
    initial_shards: list[dict[str, Any]] | None = None,
    skip_stream_rows: int = 0,
) -> dict[str, Any]:
    seen = set() if seen is None else seen
    writer = _ShardWriter(
        dest_dir,
        shard_bytes=shard_bytes,
        start_part_idx=start_part_idx,
        initial_bytes=initial_bytes,
        initial_docs=initial_docs,
        initial_shards=initial_shards,
    )
    rungs_used: list[int] = []
    rung_stats: list[dict[str, Any]] = []
    truncated = False
    baseline_bytes = initial_bytes
    rows_to_skip = max(0, int(skip_stream_rows))

    try:
        for dataset_id, subset, rung in _rungs_for(lang, use_ladder=use_ladder):
            if writer.bytes_written >= target_bytes:
                break
            added = 0
            added_bytes = 0
            skipped_dup = 0
            skipped_prefix = 0
            status = "ok"
            try:
                stream = _stream_split(dataset_id, subset)
                for row in stream:
                    # Fast-forward past rows already materialized in a prior pull.
                    # Digest dedupe still applies after the prefix for safety.
                    if rows_to_skip > 0:
                        rows_to_skip -= 1
                        skipped_prefix += 1
                        if skipped_prefix % 100_000 == 0:
                            print(
                                f"  {lang} rung{rung}: skipped_prefix={skipped_prefix} "
                                f"(remaining_skip≈{rows_to_skip})",
                                flush=True,
                            )
                        continue
                    text = _text_from_row(row)
                    if text is None:
                        continue
                    # Preserve document boundaries: keep internal newlines in JSON string.
                    digest = _doc_digest(text)
                    if digest in seen:
                        skipped_dup += 1
                        continue
                    encoded_len = len(text.encode("utf-8"))
                    if writer.bytes_written + encoded_len > target_bytes:
                        truncated = True
                        break
                    seen.add(digest)
                    writer.write(
                        {
                            "id": f"{lang}-r{rung}-{len(seen):08d}",
                            "text": text,
                            "rung": rung,
                            "digest": digest,
                        },
                        encoded_len,
                    )
                    added += 1
                    added_bytes += encoded_len
                    if added % 1000 == 0:
                        print(
                            f"  {lang} rung{rung}: {writer.docs_written} docs / "
                            f"{writer.bytes_written} bytes",
                            flush=True,
                        )
                    if writer.bytes_written >= target_bytes:
                        truncated = True
                        break
            except Exception as exc:
                status = f"failed: {exc}"
                print(f"  {lang} rung{rung} FAILED: {exc}", flush=True)

            if added > 0 or skipped_prefix > 0:
                rungs_used.append(rung)
            rung_stats.append(
                {
                    "rung": rung,
                    "dataset_id": dataset_id,
                    "subset": subset,
                    "documents": added,
                    "text_bytes": added_bytes,
                    "duplicates_skipped": skipped_dup,
                    "prefix_rows_skipped": skipped_prefix,
                    "status": status,
                }
            )
            if writer.bytes_written >= target_bytes:
                break
    finally:
        writer.close()

    planning_avail = PLANNING_AVAILABLE_BYTES.get(lang)
    unbounded = planning_avail == float("inf")
    # truncated_at_target means the next document would exceed the acquire budget —
    # i.e. we filled the target (possibly a few docs under due to whole-document stops).
    target_met = writer.bytes_written >= int(0.99 * target_bytes) or truncated
    if unbounded:
        drawn_in_full = False
    elif planning_avail is not None:
        drawn_in_full = writer.bytes_written >= 0.95 * float(planning_avail)
    else:
        drawn_in_full = False

    return {
        "status": "staged" if writer.docs_written else "empty",
        "target_bytes": target_bytes,
        "bytes": writer.bytes_written,
        "documents": writer.docs_written,
        "bytes_added": writer.bytes_written - baseline_bytes,
        "shards": writer.shards,
        "rungs_used": rungs_used,
        "rung_stats": rung_stats,
        "truncated_at_target": truncated,
        "target_met": bool(target_met),
        "drawn_in_full": bool(drawn_in_full),
        "available_bytes": None if unbounded else planning_avail,
        "unbounded": unbounded,
    }


def _load_existing_pool_state(lang_dir: Path, lang: str) -> tuple[
    set[str], int, int, int, list[dict[str, Any]]
]:
    """Scan existing part-*.jsonl.zst for digests + byte totals (append mode)."""
    import zstandard as zstd

    parts = sorted(lang_dir.glob("part-*.jsonl.zst"))
    seen: set[str] = set()
    total_bytes = 0
    total_docs = 0
    shards: list[dict[str, Any]] = []
    for part in parts:
        shard_text = 0
        dctx = zstd.ZstdDecompressor()
        with part.open("rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    text = obj.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    digest = obj.get("digest")
                    if not isinstance(digest, str) or not digest:
                        digest = _doc_digest(text)
                    seen.add(digest)
                    n = len(text.encode("utf-8"))
                    shard_text += n
                    total_bytes += n
                    total_docs += 1
        shards.append(
            {
                "path": part.name,
                "sha256": sha256_file(part),
                "text_bytes": shard_text,
            }
        )
        print(
            f"  {lang}: scanned {part.name} (+{shard_text} bytes, "
            f"{total_docs} docs cumulative)",
            flush=True,
        )
    next_idx = 0
    for part in parts:
        # part-00012.jsonl.zst → 12
        stem = part.name
        if stem.startswith("part-") and ".jsonl" in stem:
            try:
                next_idx = max(next_idx, int(stem.split("-", 1)[1].split(".", 1)[0]) + 1)
            except ValueError:
                pass
    return seen, total_bytes, total_docs, next_idx, shards


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.append and args.skip_existing:
        raise SystemExit("--append and --skip-existing are mutually exclusive")
    if args.targets_json:
        targets = {
            k: int(v)
            for k, v in json.loads(args.targets_json.read_text(encoding="utf-8")).items()
        }
    else:
        targets = _default_targets()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior: dict[str, Any] = {}
    prior_path = args.output_dir / "manifest.json"
    if args.append and prior_path.is_file():
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "plan_b_pools_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shard_bytes": args.shard_bytes,
        "append_mode": bool(args.append),
        "languages": dict(prior.get("languages") or {}),
    }

    for lang in args.langs:
        if lang not in SOURCES:
            report["languages"][lang] = {"status": "unknown_lang"}
            continue
        if lang not in targets:
            raise SystemExit(f"no target bytes for {lang}")
        dest = args.output_dir / lang
        existing = sorted(dest.glob("part-*.jsonl.zst")) if dest.is_dir() else []
        if args.skip_existing and existing:
            print(f"{lang}: skip-existing ({len(existing)} parts)", flush=True)
            report["languages"][lang] = {
                "status": "skipped_existing",
                "shards": [{"path": p.name, "sha256": sha256_file(p)} for p in existing],
            }
            continue

        seen: set[str] | None = None
        start_part_idx = 0
        initial_bytes = 0
        initial_docs = 0
        initial_shards: list[dict[str, Any]] | None = None

        if args.append and existing:
            print(
                f"{lang}: append mode — scanning {len(existing)} existing parts "
                f"toward target {targets[lang]}",
                flush=True,
            )
            prior_lang = (prior.get("languages") or {}).get(lang) or {}
            prior_shards = prior_lang.get("shards") or []
            prior_docs = int(prior_lang.get("documents") or 0)
            prior_bytes = int(prior_lang.get("bytes") or 0)
            # Prefer prior manifest when shard names match on disk (avoids multi-GB rescan).
            prior_names = [s.get("path") for s in prior_shards if isinstance(s, dict)]
            disk_names = [p.name for p in existing]
            if (
                prior_docs > 0
                and prior_bytes > 0
                and prior_names == disk_names
            ):
                print(
                    f"{lang}: using prior manifest state "
                    f"(docs={prior_docs} bytes={prior_bytes}); digest set starts empty "
                    f"+ skip_stream_rows={prior_docs}",
                    flush=True,
                )
                seen = set()
                initial_bytes = prior_bytes
                initial_docs = prior_docs
                initial_shards = list(prior_shards)
                start_part_idx = 0
                for name in disk_names:
                    if name.startswith("part-") and ".jsonl" in name:
                        try:
                            start_part_idx = max(
                                start_part_idx,
                                int(name.split("-", 1)[1].split(".", 1)[0]) + 1,
                            )
                        except ValueError:
                            pass
            else:
                seen, initial_bytes, initial_docs, start_part_idx, initial_shards = (
                    _load_existing_pool_state(dest, lang)
                )
            if initial_bytes >= int(0.99 * targets[lang]):
                print(
                    f"{lang}: already at/above target "
                    f"({initial_bytes} >= {targets[lang]}); skipping pull",
                    flush=True,
                )
                report["languages"][lang] = {
                    "status": "already_met",
                    "target_bytes": targets[lang],
                    "bytes": initial_bytes,
                    "documents": initial_docs,
                    "bytes_added": 0,
                    "shards": initial_shards,
                    "target_met": True,
                    "drawn_in_full": False,
                    "available_bytes": (
                        None
                        if PLANNING_AVAILABLE_BYTES.get(lang) == float("inf")
                        else PLANNING_AVAILABLE_BYTES.get(lang)
                    ),
                    "unbounded": PLANNING_AVAILABLE_BYTES.get(lang) == float("inf"),
                }
                continue
        elif dest.exists() and not args.append:
            for p in dest.glob("part-*.jsonl.zst"):
                p.unlink()

        print(
            f"{lang}: pulling up to {targets[lang]} bytes "
            f"(have {initial_bytes}, skip_stream_rows={initial_docs})",
            flush=True,
        )
        stats = pull_language(
            lang,
            dest,
            target_bytes=targets[lang],
            shard_bytes=args.shard_bytes,
            use_ladder=not args.no_ladder,
            seen=seen,
            start_part_idx=start_part_idx,
            initial_bytes=initial_bytes,
            initial_docs=initial_docs,
            initial_shards=initial_shards,
            skip_stream_rows=initial_docs if args.append else 0,
        )
        report["languages"][lang] = stats
        print(
            f"{lang}: done docs={stats['documents']} bytes={stats['bytes']} "
            f"added={stats.get('bytes_added', stats['bytes'])} "
            f"rungs={stats['rungs_used']} target_met={stats['target_met']}",
            flush=True,
        )

    report["total_bytes"] = sum(
        int(v.get("bytes") or 0) for v in report["languages"].values()
    )
    manifest_path = args.output_dir / "manifest.json"
    atomic_write_json(manifest_path, report)
    print(f"wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
