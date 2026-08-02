#!/usr/bin/env python3
"""Build equal-content CR-dev for Plan A premium calibration.

FLORES Plan A languages use parallel ``dev`` line indices ``0..N-1``.
``nah_Latn`` / ``yua_Latn`` use AmericasNLP calibration/dev when present; otherwise
a reserved CR pool carved from staged raw (never overlapping train after the
research-corpus builder applies the manifest exclusions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plan_a_langs import PLAN_A_CODES

PLAN_A_FLORES_LANGS = list(PLAN_A_CODES)

# All six languages exist in the local FLORES-200 extraction for both dev and
# devtest, so CR-dev is uniformly parallel. The former AmericasNLP reserved
# pool for nah_Latn / yua_Latn is unreachable under the 6-language scope --
# both languages were dropped. See plans/02-tokenizer-training.md §3.
PLAN_A_LANGS = PLAN_A_FLORES_LANGS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--flores-root",
        type=Path,
        default=ROOT / "data" / "flores200_dataset",
    )
    p.add_argument(
        "--americasnlp-dir",
        type=Path,
        default=ROOT / "data" / "americasnlp",
        help="Optional AmericasNLP root with per-lang calibration/dev files",
    )
    p.add_argument(
        "--reserved-pool-dir",
        type=Path,
        default=ROOT / "artifacts" / "plan_a" / "raw" / "fineweb2_samples",
        help="Fallback CR pool for nah/yua when AmericasNLP is absent",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "plan_a" / "research_cpu" / "cr_dev",
    )
    p.add_argument(
        "--n-lines",
        type=int,
        default=None,
        help="Override N; default is min available lines across all CR sources",
    )
    return p.parse_args()


def _read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.rstrip("\r\n")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sentence_units_with_doc_span(
    lines: list[str],
    *,
    max_chars: int = 280,
    n_units: int | None = None,
) -> tuple[list[str], int]:
    """Expand docs into sentence units; return (units, raw_docs_consumed)."""
    units: list[str] = []
    docs_used = 0
    for line in lines:
        docs_used += 1
        parts = re.split(r"(?<=[.!?。！？])\s+", line)
        buf = ""
        for part in parts:
            piece = part.strip()
            if not piece:
                continue
            if not buf:
                buf = piece
            elif len(buf) + 1 + len(piece) <= max_chars:
                buf = f"{buf} {piece}"
            else:
                units.append(buf)
                buf = piece
            while len(buf) > max_chars:
                units.append(buf[:max_chars].rstrip())
                buf = buf[max_chars:].lstrip()
            if n_units is not None and len(units) >= n_units:
                return units[:n_units], docs_used
        if buf:
            units.append(buf)
        if n_units is not None and len(units) >= n_units:
            return units[:n_units], docs_used
    return units, docs_used


def _sentence_units(lines: list[str], *, max_chars: int = 280) -> list[str]:
    units, _ = _sentence_units_with_doc_span(lines, max_chars=max_chars)
    return units


def _find_americas_file(americas_dir: Path, lang: str) -> Path | None:
    if not americas_dir.is_dir():
        return None
    candidates = [
        americas_dir / lang / "calibration.txt",
        americas_dir / lang / "dev.txt",
        americas_dir / f"{lang}.calibration.txt",
        americas_dir / f"{lang}.dev.txt",
        americas_dir / f"{lang}.txt",
        americas_dir / "calibration" / f"{lang}.txt",
        americas_dir / "dev" / f"{lang}.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    # Fuzzy: any file whose stem starts with lang under the tree.
    matches = sorted(americas_dir.rglob(f"{lang}*"))
    for path in matches:
        if path.is_file() and path.suffix.lower() in {".txt", ".tsv", ".jsonl"}:
            return path
    return None


def _load_source_lines(
    *,
    lang: str,
    flores_root: Path,
    americas_dir: Path,
    reserved_pool_dir: Path,
) -> tuple[list[str], dict]:
    if lang in PLAN_A_FLORES_LANGS:
        path = flores_root / "dev" / f"{lang}.dev"
        if not path.is_file():
            raise FileNotFoundError(f"FLORES dev missing for {lang}: {path}")
        lines = _read_nonempty_lines(path)
        meta = {
            "source_uri": str(path.resolve()),
            "source_kind": "flores_dev",
            "parallel": True,
            "available_lines": len(lines),
            "reserved_line_start": 0,
            "reserved_line_end": None,  # filled after N chosen
        }
        return lines, meta

    americas = _find_americas_file(americas_dir, lang)
    if americas is not None:
        lines = _read_nonempty_lines(americas)
        meta = {
            "source_uri": str(americas.resolve()),
            "source_kind": "americasnlp_calibration",
            "parallel": False,
            "available_lines": len(lines),
            "reserved_line_start": 0,
            "reserved_line_end": None,
        }
        return lines, meta

    pool = reserved_pool_dir / f"{lang}.txt"
    if not pool.is_file():
        raise FileNotFoundError(
            f"No AmericasNLP file for {lang} under {americas_dir} and no "
            f"reserved CR pool at {pool}. Provide --americasnlp-dir or "
            f"--reserved-pool-dir with {lang}.txt."
        )
    raw_docs = _read_nonempty_lines(pool)
    lines = _sentence_units(raw_docs)
    meta = {
        "source_uri": str(pool.resolve()),
        "source_kind": "reserved_raw_pool",
        "parallel": False,
        "available_lines": len(lines),
        "raw_docs": len(raw_docs),
        "reserved_line_start": 0,
        "reserved_line_end": None,
        "train_exclude": True,
        "train_exclude_mode": "source_uri_prefix_docs",
        "note": (
            "Sentence units carved from reserved FineWeb-style docs for CR-dev. "
            "Research corpus builder excludes the contributing raw docs "
            "(first docs needed to produce N units) from train."
        ),
    }
    return lines, meta


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[list[str], dict]] = {}
    for lang in PLAN_A_LANGS:
        loaded[lang] = _load_source_lines(
            lang=lang,
            flores_root=args.flores_root,
            americas_dir=args.americasnlp_dir,
            reserved_pool_dir=args.reserved_pool_dir,
        )

    available = {lang: meta["available_lines"] for lang, (_, meta) in loaded.items()}
    n_lines = args.n_lines if args.n_lines is not None else min(available.values())
    if n_lines <= 0:
        raise SystemExit("n_lines must be positive")
    short = {lang: n for lang, n in available.items() if n < n_lines}
    if short:
        raise SystemExit(
            f"Requested n_lines={n_lines} exceeds available lines for: {short}"
        )

    languages_meta: dict[str, dict] = {}
    for lang in PLAN_A_LANGS:
        lines, meta = loaded[lang]
        selected = lines[:n_lines]
        if len(selected) != n_lines:
            raise SystemExit(f"{lang}: expected {n_lines} lines, got {len(selected)}")
        payload = "\n".join(selected) + "\n"
        dest = out / f"{lang}.txt"
        dest.write_text(payload, encoding="utf-8")
        meta = dict(meta)
        meta["reserved_line_end"] = n_lines
        meta["n_lines"] = n_lines
        meta["byte_count"] = len(payload.encode("utf-8"))
        meta["sha256"] = _sha256_text(payload)
        if meta.get("source_kind") == "reserved_raw_pool":
            raw_docs = _read_nonempty_lines(Path(meta["source_uri"]))
            _, docs_used = _sentence_units_with_doc_span(raw_docs, n_units=n_lines)
            meta["reserved_raw_doc_start"] = 0
            meta["reserved_raw_doc_end"] = docs_used
        languages_meta[lang] = meta

    # Hard-fail if any lang diverged (paranoia check after write).
    for lang in PLAN_A_LANGS:
        got = len(_read_nonempty_lines(out / f"{lang}.txt"))
        if got != n_lines:
            raise SystemExit(f"{lang}.txt has {got} lines, expected {n_lines}")

    reserved = [
        lang
        for lang, meta in languages_meta.items()
        if meta.get("source_kind") == "reserved_raw_pool"
    ]
    manifest = {
        "schema_version": 1,
        "kind": "plan_a_cr_dev",
        "n_lines": n_lines,
        "languages": PLAN_A_LANGS,
        "flores_langs": PLAN_A_FLORES_LANGS,
        "americas_langs": AMERICAS_CR_LANGS,
        "reserved_pool_langs": reserved,
        "per_language": languages_meta,
        "note": (
            "Equal-content CR-dev: FLORES langs share parallel indices 0..N-1; "
            "nah/yua capped to the same N from AmericasNLP or a reserved raw pool."
        ),
    }
    manifest_path = out / "cr_dev_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {len(PLAN_A_LANGS)} languages × {n_lines} lines to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
