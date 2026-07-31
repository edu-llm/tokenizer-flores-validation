#!/usr/bin/env python3
"""Pull bounded FineWeb2 / FineWeb samples for all Plan A target languages.

FineWeb2 has no English; English comes from HuggingFaceFW/fineweb.
Guarani uses FineWeb2 ``gug_Latn`` mapped to project code ``grn_Latn``.
Mandarin uses FineWeb2 ``cmn_Hani`` mapped to ``zho_Hans``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "plan_a" / "raw" / "fineweb2_samples"

# project_lang -> (dataset_id, subset_or_None, text_key_hint)
LANG_SOURCES: dict[str, tuple[str, str | None]] = {
    "eng_Latn": ("HuggingFaceFW/fineweb", None),  # sample-10BT or default; stream train
    "amh_Ethi": ("HuggingFaceFW/fineweb-2", "amh_Ethi"),
    "hau_Latn": ("HuggingFaceFW/fineweb-2", "hau_Latn"),
    "swh_Latn": ("HuggingFaceFW/fineweb-2", "swh_Latn"),
    "ukr_Cyrl": ("HuggingFaceFW/fineweb-2", "ukr_Cyrl"),
    "pol_Latn": ("HuggingFaceFW/fineweb-2", "pol_Latn"),
    "hun_Latn": ("HuggingFaceFW/fineweb-2", "hun_Latn"),
    "tel_Telu": ("HuggingFaceFW/fineweb-2", "tel_Telu"),
    "ory_Orya": ("HuggingFaceFW/fineweb-2", "ory_Orya"),
    "zho_Hans": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),
    "tur_Latn": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
    "ayr_Latn": ("HuggingFaceFW/fineweb-2", "ayr_Latn"),
    "quy_Latn": ("HuggingFaceFW/fineweb-2", "quy_Latn"),
    "grn_Latn": ("HuggingFaceFW/fineweb-2", "gug_Latn"),
    "nah_Latn": ("HuggingFaceFW/fineweb-2", "nah_Latn"),
    "yua_Latn": ("HuggingFaceFW/fineweb-2", "yua_Latn"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-bytes-per-lang", type=int, default=8_000_000)
    p.add_argument("--langs", nargs="*", default=sorted(LANG_SOURCES))
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def stream_lang(dataset_id: str, subset: str | None, dest: Path, max_bytes: int) -> dict:
    from datasets import load_dataset

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    n_docs = 0
    truncated = False
    try:
        if subset:
            ds = load_dataset(dataset_id, name=subset, split="train", streaming=True)
        elif dataset_id.endswith("/fineweb"):
            # Prefer a small config if present; else default train stream.
            try:
                ds = load_dataset(dataset_id, name="sample-10BT", split="train", streaming=True)
            except Exception:
                ds = load_dataset(dataset_id, split="train", streaming=True)
        else:
            ds = load_dataset(dataset_id, split="train", streaming=True)
    except Exception as exc:
        return {
            "status": "failed",
            "dataset_id": dataset_id,
            "subset": subset,
            "reason": str(exc),
        }

    with dest.open("w", encoding="utf-8") as out:
        for row in ds:
            text = None
            if isinstance(row, dict):
                for key in ("text", "content", "raw_content"):
                    if isinstance(row.get(key), str) and row[key].strip():
                        text = row[key]
                        break
            if not text:
                continue
            line = text.replace("\n", " ").strip()
            if not line:
                continue
            encoded = (line + "\n").encode("utf-8")
            if written + len(encoded) > max_bytes:
                truncated = True
                break
            out.write(line + "\n")
            written += len(encoded)
            n_docs += 1
            if n_docs % 500 == 0:
                print(f"  {dest.name}: {n_docs} docs / {written} bytes", flush=True)

    return {
        "status": "staged",
        "dataset_id": dataset_id,
        "subset": subset,
        "path": str(dest),
        "bytes": written,
        "documents": n_docs,
        "truncated": truncated,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "kind": "fineweb2_lang_sample_pull",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_bytes_per_lang": args.max_bytes_per_lang,
        "results": {},
    }

    for lang in args.langs:
        if lang not in LANG_SOURCES:
            report["results"][lang] = {"status": "unknown_lang"}
            continue
        dataset_id, subset = LANG_SOURCES[lang]
        dest = args.output_dir / f"{lang}.txt"
        if args.skip_existing and dest.exists() and dest.stat().st_size > 0:
            report["results"][lang] = {
                "status": "skipped_existing",
                "path": str(dest),
                "bytes": dest.stat().st_size,
            }
            print(f"SKIP {lang} ({dest.stat().st_size} bytes)")
            continue
        print(f"PULL {lang} <- {dataset_id} [{subset}]", flush=True)
        meta = stream_lang(dataset_id, subset, dest, args.max_bytes_per_lang)
        report["results"][lang] = meta
        print(f"DONE {lang}: {meta}", flush=True)

    out_json = args.output_dir / "pull_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = sum(1 for r in report["results"].values() if r.get("status") == "staged")
    fail = sum(1 for r in report["results"].values() if r.get("status") == "failed")
    print(f"Wrote {out_json}; staged={ok} failed={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
