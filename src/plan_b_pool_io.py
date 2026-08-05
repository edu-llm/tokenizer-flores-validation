"""Read Plan B research pool shards (``part-*.jsonl[.gz|.zst]``).

Mirrors the document iteration used by ``edullm-data``
``stage_fineweb2_unimax_pools.py`` so tokenize / measure see the same docs.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterator, TextIO


def open_text_stream(path: Path) -> TextIO:
    name = path.name.lower()
    if name.endswith(".jsonl.zst"):
        try:
            import zstandard as zstd
        except ImportError as e:
            raise ImportError(
                f"{path}: .jsonl.zst requires the 'zstandard' package"
            ) from e
        fh = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", newline="\n")
    if name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="\n")
    if name.endswith(".jsonl"):
        return path.open("r", encoding="utf-8", newline="\n")
    raise ValueError(f"unsupported pool shard extension: {path}")


def iter_pool_parts(lang_dir: Path) -> list[Path]:
    parts = sorted(lang_dir.glob("part-*.jsonl"))
    parts += sorted(lang_dir.glob("part-*.jsonl.gz"))
    parts += sorted(lang_dir.glob("part-*.jsonl.zst"))
    for name in ("docs.jsonl", "docs.jsonl.gz", "docs.jsonl.zst"):
        p = lang_dir / name
        if p.is_file():
            parts.append(p)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in parts:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def doc_from_line(line: str, *, lang: str, seq: int) -> dict[str, str] | None:
    text_line = line.rstrip("\r\n")
    if not text_line:
        return None
    text: str
    doc_id: str | None = None
    if text_line.startswith("{"):
        try:
            obj = json.loads(text_line)
        except json.JSONDecodeError:
            text = text_line
        else:
            if not isinstance(obj, dict):
                return None
            raw = obj.get("text")
            if not isinstance(raw, str) or not raw.strip():
                return None
            text = raw
            if isinstance(obj.get("id"), str) and obj["id"]:
                doc_id = obj["id"]
    else:
        text = text_line
    if not text.strip():
        return None
    return {"id": doc_id or f"{lang}-{seq:08d}", "text": text}


def iter_docs(lang_dir: Path, lang: str) -> Iterator[dict[str, str]]:
    parts = iter_pool_parts(lang_dir)
    if not parts:
        raise FileNotFoundError(
            f"{lang_dir}: no part-*.jsonl(.gz|.zst) pool shards found"
        )
    seq = 0
    for part in parts:
        with open_text_stream(part) as fin:
            for line in fin:
                doc = doc_from_line(line, lang=lang, seq=seq)
                if doc is None:
                    continue
                seq += 1
                yield doc
