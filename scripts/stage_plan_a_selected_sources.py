#!/usr/bin/env python3
"""Freeze Plan A train sources and stage local raw samples.

Default research policy (``--research-use-all-train``):
  - Include **all** ``train_candidate`` sources, including CC BY-SA Wikipedia
    and license-unknown / needs_review items.
  - Still exclude eval/benchmark, reference, gated-as-role, out_of_scope, broken.
  - Intended for non-commercial / non-public research only; licenses are recorded
    but not used as an ingest hard-gate.

Optional ``--permissive-only`` restores the stricter MIT/Apache/CC0/ODC-By/CC-BY gate
(excluding CC BY-SA).

Large pools are marked requires_cloud_staging; local pulls are byte-capped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INV = ROOT / "artifacts" / "plan_a" / "source_inventory.json"
SELECTED = ROOT / "artifacts" / "plan_a" / "sources_selected_for_ingest.json"
RAW_ROOT = ROOT / "artifacts" / "plan_a" / "raw"

CLOUD_SCALE = {
    "src_3e2e3feeb4",  # FineWeb2
    "src_3b5b0b0edf",  # HPLT 3.0
    "src_2fcbe429a6",  # MADLAD-400
    "src_79c4013030",  # FineTranslations
    "src_5eeb4b29cc",  # FineWeb
    "src_96f9078ee7",  # Sangraha
    "src_d09b01234c",  # Samanantar te
    "src_65d00ca483",  # Samanantar or
}

HF_LOCAL = {
    "src_82484469c6": "wjosielct/aymara-spanish-parallel-corpus",
    "src_6d56cdb915": "ngusadeep/Swahili-Corpus-Dataset",
    "src_0163a4f6b7": "Adeptschneider/CiviVox-Swahili-text-corpus",
    "src_c6bc39cd2e": "a3xrfgb/amharic-sentences-corpus",
    "src_4e37e82e59": "dagn/expanded-amharic-news-dataset",
    "src_5ef337bab5": "Reubencf/Amharic_corpus",
}

BIBLE_RAW = {
    "src_42032940c3": (
        "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Nahuatl-NT.xml",
        "nah_Latn",
    ),
    "src_49b244580b": (
        "https://raw.githubusercontent.com/christos-c/bible-corpus/master/bibles/Quichua-NT.xml",
        "quy_Latn",
    ),
    "src_83817a6151": (
        "https://api.github.com/repos/christos-c/bible-corpus/contents/bibles",
        None,
    ),
}

TARGET_LANGS = [
    "eng_Latn",
    "amh_Ethi",
    "hau_Latn",
    "swh_Latn",
    "ukr_Cyrl",
    "pol_Latn",
    "hun_Latn",
    "tel_Telu",
    "ory_Orya",
    "zho_Hans",
    "tur_Latn",
    "ayr_Latn",
    "quy_Latn",
    "grn_Latn",
    "nah_Latn",
    "yua_Latn",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-bytes-per-source", type=int, default=50_000_000)
    p.add_argument("--skip-download", action="store_true", help="Only write selection manifests")
    p.add_argument("--include-cloud-samples", action="store_true", help="Stream tiny FineWeb2 samples")
    p.add_argument(
        "--permissive-only",
        action="store_true",
        help="Hard-gate on permissive licenses; exclude CC BY-SA / needs_review",
    )
    p.add_argument(
        "--research-use-all-train",
        action="store_true",
        default=True,
        help="Include all train_candidate sources (default; non-commercial research)",
    )
    return p.parse_args()


def _needs_cloud(src: dict) -> bool:
    if src["source_id"] in CLOUD_SCALE:
        return True
    url = src.get("url", "")
    host = src.get("host", "")
    if "wikipedia.org" in url or "wikimedia.org" in url:
        return True
    if host in {"www.sketchengine.eu", "opus.nlpl.eu", "wortschatz.uni-leipzig.de", "corpora.wortschatz-leipzig.de"}:
        return True
    if "huggingface.co/datasets" in url and src["source_id"] not in HF_LOCAL:
        # Unknown HF size → treat as cloud unless already local-stageable
        return True
    return False


def freeze_selection(inv: dict, *, permissive_only: bool) -> dict:
    selected = []
    excluded = []
    for src in inv["sources"]:
        if src.get("role") != "train_candidate":
            continue
        status = src.get("license_status")
        lic = str(src.get("license", "unknown"))
        include = True
        reason = None
        if permissive_only:
            include = bool(status in {"mit", "permissive_open"} or src.get("is_permissive_open"))
            if include and lic.startswith("cc-by-sa"):
                include = False
                reason = "cc_by_sa_excluded"
            elif not include:
                reason = status or "not_permissive_open"
                if status == "copyleft_sharealike" or lic.startswith("cc-by-sa"):
                    reason = "cc_by_sa_excluded"

        if include:
            entry = {
                "source_id": src["source_id"],
                "name": src["name"],
                "url": src["url"],
                "license": src.get("license"),
                "license_status": status,
                "priority": src.get("priority"),
                "language_hint": src.get("language_hint"),
                "requires_cloud_staging": _needs_cloud(src),
                "local_stageable": src["source_id"] in HF_LOCAL or src["source_id"] in BIBLE_RAW,
                "research_use_included": not permissive_only,
            }
            selected.append(entry)
            src["selected_for_ingest"] = True
            src["requires_cloud_staging"] = entry["requires_cloud_staging"]
            src.pop("ingest_exclusion_reason", None)
        else:
            excluded.append(
                {
                    "source_id": src["source_id"],
                    "name": src["name"],
                    "url": src["url"],
                    "license": lic,
                    "reason": reason,
                }
            )
            src["selected_for_ingest"] = False
            src["ingest_exclusion_reason"] = reason

    deduped = []
    seen = set()
    for s in selected:
        key = (s["url"], s.get("language_hint"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    if permissive_only:
        policy = {
            "mode": "permissive_only",
            "include": "permissive_open (MIT/Apache/BSD/CC0/ODC-By/CC-BY)",
            "exclude": "CC-BY-SA; needs_review; non-permissive",
        }
    else:
        policy = {
            "mode": "research_use_all_train",
            "include": "all train_candidate sources (including CC BY-SA and needs_review)",
            "exclude": "eval/benchmark, reference, gated-role, out_of_scope, broken only",
            "use_restriction": "non-commercial / non-public research; licenses recorded but not hard-gated",
        }

    return {
        "schema_version": 1,
        "kind": "plan_a_selected_sources",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "target_languages": TARGET_LANGS,
        "selected_count": len(deduped),
        "excluded_count": len(excluded),
        "selected": deduped,
        "excluded_train_candidates": excluded,
    }


def download_url(url: str, dest: Path, max_bytes: int) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tokenizer-flores-validation/plan-a"})
    written = 0
    truncated = False
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            if written + len(chunk) > max_bytes:
                out.write(chunk[: max(0, max_bytes - written)])
                written = max_bytes
                truncated = True
                break
            out.write(chunk)
            written += len(chunk)
    return {"path": str(dest), "bytes": written, "truncated": truncated, "url": url}


def stage_bible(source_id: str, max_bytes: int) -> dict:
    url, lang = BIBLE_RAW[source_id]
    if source_id == "src_83817a6151":
        # Listing only — store API JSON index for later selective fetch
        dest = RAW_ROOT / "bible-corpus" / "bibles_index.json"
        return {"source_id": source_id, **download_url(url, dest, max_bytes)}
    name = Path(url).name
    dest = RAW_ROOT / "bible-corpus" / (lang or "multi") / name
    meta = download_url(url, dest, max_bytes)
    meta.update({"source_id": source_id, "language_hint": lang})
    return meta


def stage_hf_dataset(dataset_id: str, source_id: str, max_bytes: int) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return {
            "source_id": source_id,
            "dataset_id": dataset_id,
            "status": "skipped",
            "reason": f"datasets_not_installed:{exc}",
        }

    out_dir = RAW_ROOT / "hf" / dataset_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / "sample.txt"
    written = 0
    n_docs = 0
    truncated = False
    try:
        ds = load_dataset(dataset_id, split="train", streaming=True)
    except Exception:
        # try default config / first split
        try:
            ds_dict = load_dataset(dataset_id, streaming=True)
            split = next(iter(ds_dict.keys()))
            ds = ds_dict[split]
        except Exception as exc:
            return {
                "source_id": source_id,
                "dataset_id": dataset_id,
                "status": "failed",
                "reason": str(exc),
            }

    text_keys = ("text", "content", "sentence", "article", "body", "amharic", "swahili", "aymara", "src", "trg")
    with text_path.open("w", encoding="utf-8") as out:
        for row in ds:
            blob = None
            if isinstance(row, dict):
                for k in text_keys:
                    if k in row and isinstance(row[k], str) and row[k].strip():
                        blob = row[k]
                        break
                if blob is None:
                    # concatenate string fields
                    parts = [str(v) for v in row.values() if isinstance(v, str) and v.strip()]
                    blob = "\t".join(parts) if parts else None
            if not blob:
                continue
            line = blob.replace("\n", " ").strip() + "\n"
            encoded = line.encode("utf-8")
            if written + len(encoded) > max_bytes:
                truncated = True
                break
            out.write(line)
            written += len(encoded)
            n_docs += 1
            if n_docs >= 200_000:
                truncated = True
                break

    manifest = {
        "source_id": source_id,
        "dataset_id": dataset_id,
        "status": "staged",
        "path": str(text_path),
        "bytes": written,
        "documents": n_docs,
        "truncated": truncated,
    }
    (out_dir / "stage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    inv = json.loads(INV.read_text(encoding="utf-8"))
    selection = freeze_selection(inv, permissive_only=args.permissive_only)
    SELECTED.parent.mkdir(parents=True, exist_ok=True)
    SELECTED.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    inv["status"] = (
        "selection_frozen_permissive_open"
        if args.permissive_only
        else "selection_frozen_research_use_all_train"
    )
    inv["selection_path"] = str(SELECTED)
    inv["use_restriction"] = selection["policy"].get("use_restriction")
    INV.write_text(json.dumps(inv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen {selection['selected_count']} sources (excluded {selection['excluded_count']}) -> {SELECTED}")
    print(f"Policy mode: {selection['policy']['mode']}")

    if args.skip_download:
        return 0

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for source_id in BIBLE_RAW:
        if any(s["source_id"] == source_id for s in selection["selected"]):
            print(f"Staging bible {source_id}...")
            results.append(stage_bible(source_id, args.max_bytes_per_source))

    for source_id, dataset_id in HF_LOCAL.items():
        if any(s["source_id"] == source_id for s in selection["selected"]):
            print(f"Staging HF {dataset_id}...")
            results.append(stage_hf_dataset(dataset_id, source_id, args.max_bytes_per_source))

    cloud = [s for s in selection["selected"] if s["requires_cloud_staging"]]
    results.append(
        {
            "status": "deferred_cloud",
            "count": len(cloud),
            "sources": [{"source_id": s["source_id"], "name": s["name"], "url": s["url"]} for s in cloud],
            "note": "Stage on AWS Batch/S3; local laptop should not pull full FineWeb2/HPLT/MADLAD.",
        }
    )

    if args.include_cloud_samples:
        # Optional tiny FineWeb2 English sample only
        print("Sampling FineWeb2 eng_Latn (bounded)...")
        results.append(
            stage_hf_dataset("HuggingFaceFW/fineweb-2", "src_3e2e3feeb4", min(args.max_bytes_per_source, 5_000_000))
        )

    stage_report = {
        "schema_version": 1,
        "kind": "plan_a_local_stage_report",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_bytes_per_source": args.max_bytes_per_source,
        "results": results,
    }
    report_path = ROOT / "artifacts" / "plan_a" / "local_stage_report.json"
    report_path.write_text(json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Stage report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
