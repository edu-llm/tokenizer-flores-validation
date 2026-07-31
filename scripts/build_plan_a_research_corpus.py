#!/usr/bin/env python3
"""Build a Plan A research CPU corpus from staged raw sources.

Uses only ``artifacts/plan_a/raw`` (not FLORES) for tokenizer *training* text.
Writes per-language train files when language is known; remaining bytes go to
``multi/unknown.txt`` and the shared ``corpus/train.txt``.

When ``--cr-dev-manifest`` is supplied, reserved CR-pool line ranges for
``nah_Latn`` / ``yua_Latn`` are excluded from train so CR-dev never overlaps
tokenizer training text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "plan_a" / "raw"
OUT = ROOT / "artifacts" / "plan_a" / "research_cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-bytes", type=int, default=100_000_000)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument(
        "--cr-dev-manifest",
        type=Path,
        default=None,
        help="cr_dev_manifest.json; excludes reserved CR pool line ranges from train",
    )
    return p.parse_args()


def load_cr_exclusions(manifest_path: Path | None) -> dict[str, tuple[int, int, str]]:
    """Return lang -> (start, end, source_uri) reserved raw-doc ranges to skip."""
    if manifest_path is None:
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int, str]] = {}
    for lang, meta in payload.get("per_language", {}).items():
        if not meta.get("train_exclude"):
            continue
        uri = str(meta["source_uri"])
        if "reserved_raw_doc_end" in meta:
            start = int(meta.get("reserved_raw_doc_start", 0))
            end = int(meta["reserved_raw_doc_end"])
        else:
            start = int(meta.get("reserved_line_start", 0))
            end = int(meta["reserved_line_end"])
        out[lang] = (start, end, uri)
    return out


def apply_cr_exclusion(
    lang: str | None,
    path: Path,
    lines: list[str],
    exclusions: dict[str, tuple[int, int, str]],
) -> list[str]:
    if not lang or lang not in exclusions:
        return lines
    start, end, source_uri = exclusions[lang]
    if Path(source_uri).resolve() != path.resolve():
        # Only skip when this file is the reserved CR pool source.
        return lines
    return lines[:start] + lines[end:]


def bible_xml_to_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Prefer verse-like tags; fall back to stripping tags.
    lines: list[str] = []
    try:
        root = ET.fromstring(text)
        for el in root.iter():
            if el.text and el.text.strip():
                t = re.sub(r"\s+", " ", el.text).strip()
                if len(t) >= 8:
                    lines.append(t)
    except ET.ParseError:
        stripped = re.sub(r"<[^>]+>", " ", text)
        for part in re.split(r"[\r\n]+", stripped):
            t = re.sub(r"\s+", " ", part).strip()
            if len(t) >= 8:
                lines.append(t)
    return lines


def collect_sources() -> list[tuple[str | None, Path, str]]:
    """Return list of (language_hint|None, path, kind)."""
    found: list[tuple[str | None, Path, str]] = []
    bible = RAW / "bible-corpus"
    if bible.exists():
        for lang_dir in sorted(p for p in bible.iterdir() if p.is_dir()):
            for xml in lang_dir.glob("*.xml"):
                found.append((lang_dir.name, xml, "bible_xml"))
    hf = RAW / "hf"
    if hf.exists():
        mapping = {
            "ngusadeep__Swahili-Corpus-Dataset": "swh_Latn",
            "Adeptschneider__CiviVox-Swahili-text-corpus": "swh_Latn",
            "a3xrfgb__amharic-sentences-corpus": "amh_Ethi",
            "Reubencf__Amharic_corpus": "amh_Ethi",
            "wjosielct__aymara-spanish-parallel-corpus": "ayr_Latn",
        }
        for ds_dir in sorted(p for p in hf.iterdir() if p.is_dir()):
            sample = ds_dir / "sample.txt"
            if sample.exists():
                found.append((mapping.get(ds_dir.name), sample, "hf_sample"))
    fw = RAW / "fineweb2_samples"
    if fw.exists():
        for txt in sorted(fw.glob("*.txt")):
            found.append((txt.stem, txt, "fineweb2_sample"))
    return found


def main() -> int:
    args = parse_args()
    out = args.output_dir
    train_lang = out / "train_langs"
    corpus_dir = out / "corpus"
    for d in (train_lang, corpus_dir):
        d.mkdir(parents=True, exist_ok=True)

    exclusions = load_cr_exclusions(args.cr_dev_manifest)
    by_lang: dict[str, list[str]] = {}
    multi: list[str] = []
    total = 0
    sources_meta = []

    for lang, path, kind in collect_sources():
        if total >= args.max_bytes:
            break
        if kind == "bible_xml":
            lines = bible_xml_to_lines(path)
        else:
            lines = [
                ln.strip()
                for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()
            ]
        before = len(lines)
        lines = apply_cr_exclusion(lang, path, lines, exclusions)
        skipped = before - len(lines)
        wrote = 0
        kept: list[str] = []
        for ln in lines:
            enc = (ln + "\n").encode("utf-8")
            if total + len(enc) > args.max_bytes:
                break
            kept.append(ln)
            total += len(enc)
            wrote += len(enc)
        if not kept:
            continue
        bucket = lang if lang else "multi_unknown"
        by_lang.setdefault(bucket, []).extend(kept)
        if not lang:
            multi.extend(kept)
        sources_meta.append(
            {
                "path": str(path),
                "language_hint": lang,
                "kind": kind,
                "bytes_used": wrote,
                "lines": len(kept),
                "cr_lines_excluded": skipped,
            }
        )

    # Write per-lang files
    for lang, lines in sorted(by_lang.items()):
        (train_lang / f"{lang}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Shared corpus for official BPE/SuperBPE (deterministic lang order)
    corpus_lines: list[str] = []
    for lang in sorted(by_lang):
        corpus_lines.extend(by_lang[lang])
    train_txt = corpus_dir / "train.txt"
    payload = "\n".join(corpus_lines) + ("\n" if corpus_lines else "")
    train_txt.write_text(payload, encoding="utf-8")
    actual = train_txt.stat().st_size
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "plan_a_research_cpu_corpus",
        "actual_bytes": actual,
        "sha256": digest,
        "records": len(corpus_lines),
        "languages": sorted(by_lang),
        "sources": sources_meta,
        "cr_dev_manifest": str(args.cr_dev_manifest.resolve())
        if args.cr_dev_manifest
        else None,
        "cr_exclusions": {
            lang: {"start": start, "end": end, "source_uri": uri}
            for lang, (start, end, uri) in exclusions.items()
        },
        "note": (
            "Built from staged Plan A raw sources only; FLORES excluded from train. "
            "Reserved CR-pool lines for nah/yua excluded when --cr-dev-manifest is set."
        ),
    }
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {train_txt} ({actual} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
