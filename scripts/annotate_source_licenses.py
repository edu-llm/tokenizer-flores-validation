#!/usr/bin/env python3
"""Annotate Plan A source inventory with known / Hugging Face license tags.

MIT is rare for web corpora. This script records exact licenses and also a
``permissive_open`` flag for MIT/Apache/BSD/CC0/ODC-By/CC-BY (not CC-BY-SA).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV_PATH = ROOT / "artifacts" / "plan_a" / "source_inventory.json"
OUT_PATH = ROOT / "artifacts" / "plan_a" / "source_inventory_licensed.json"
ALLOW_PATH = ROOT / "artifacts" / "plan_a" / "sources_permissive_open.json"

# SPDX-ish ids treated as permissive-open for Plan A ingest gate.
PERMISSIVE = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-2.0",
    "odc-by",
    "odc-by-1.0",
    "pd",
    "public-domain",
}

# Copyleft / share-alike / commercial-portal / unknown-restrictive.
NON_PERMISSIVE_HINTS = {
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa",
    "gpl-3.0",
    "gpl-2.0",
    "agpl-3.0",
    "other",
    "unknown",
}

# Curated overrides for non-HF or well-known sources (best-effort; verify before final).
KNOWN: dict[str, dict] = {
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2": {
        "license": "odc-by-1.0",
        "license_notes": "Also subject to Common Crawl ToU",
        "confidence": "high",
    },
    "https://huggingface.co/datasets/HuggingFaceFW/finetranslations": {
        "license": "odc-by-1.0",
        "license_notes": "FineTranslations; verify card",
        "confidence": "medium",
    },
    "https://huggingface.co/datasets/allenai/MADLAD-400": {
        "license": "odc-by",
        "license_notes": "HF tag odc-by; README also mentions CC-BY-4.0",
        "confidence": "high",
    },
    "https://hplt-project.org/datasets/v3.0": {
        "license": "cc0-1.0",
        "license_notes": "Packaging/metadata CC0; no claim on raw text; user must comply with jurisdiction",
        "confidence": "high",
    },
    "https://opus.nlpl.eu": {
        "license": "mixed",
        "license_notes": "Per-corpus licenses vary; not blanket MIT",
        "confidence": "high",
    },
    "https://en.wikipedia.org/wiki/Main_Page": {
        "license": "cc-by-sa-4.0",
        "license_notes": "Wikipedia text CC BY-SA; dumps not MIT",
        "confidence": "high",
    },
    "https://github.com/christos-c/bible-corpus/tree/master/bibles": {
        "license": "cc0-1.0",
        "license_notes": "christos-c/bible-corpus commonly CC0; verify repo LICENSE",
        "confidence": "medium",
    },
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb": {
        "license": "odc-by-1.0",
        "confidence": "high",
    },
    "https://github.com/common-voice/cv-dataset/": {
        "license": "cc0-1.0",
        "license_notes": "Common Voice transcripts typically CC0; out_of_scope for text LM anyway",
        "confidence": "medium",
    },
}


def normalize_license(raw: str | None) -> str:
    if not raw:
        return "unknown"
    s = raw.strip().lower().replace(" ", "-")
    s = s.replace("odc_by", "odc-by").replace("odc-by-", "odc-by-")
    if s in {"odc-by", "odc-by-1.0", "odcby"}:
        return "odc-by-1.0" if "1.0" in s or s == "odc-by" else s
    if s.startswith("cc-by-sa"):
        return "cc-by-sa-4.0" if "4" in s else "cc-by-sa"
    if s in {"mit", "apache-2.0", "apache2", "apache-2"}:
        return "mit" if s == "mit" else "apache-2.0"
    if s in {"cc0", "cc-0", "cc0-1.0"}:
        return "cc0-1.0"
    if s.startswith("cc-by") and "sa" not in s:
        return "cc-by-4.0" if "4" in s else s
    return s


def is_permissive(lic: str) -> bool:
    lic = normalize_license(lic)
    if lic in PERMISSIVE or lic.startswith("odc-by") or lic.startswith("cc-by-") and "sa" not in lic:
        return True
    if lic in {"mit", "apache-2.0", "cc0-1.0"}:
        return True
    return False


def is_mit(lic: str) -> bool:
    return normalize_license(lic) == "mit"


def hf_dataset_id(url: str) -> str | None:
    m = re.match(r"https://huggingface\.co/datasets/([^/\s#]+)/([^/\s#]+)", url)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


def fetch_hf_license(dataset_id: str) -> tuple[str | None, str]:
    api = f"https://huggingface.co/api/datasets/{dataset_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "tokenizer-flores-validation/plan-a"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"hf_api_error:{exc}"
    card = data.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    tags = data.get("tags") or []
    if not lic:
        for t in tags:
            if isinstance(t, str) and t.startswith("license:"):
                lic = t.split(":", 1)[1]
                break
    return (str(lic) if lic else None), "hf_api"


def annotate_wikipedia(url: str) -> dict | None:
    if "wikipedia.org" in url or "wikimedia.org" in url or "incubator.wikimedia.org" in url:
        return {
            "license": "cc-by-sa-4.0",
            "license_notes": "Wikimedia text generally CC BY-SA; not MIT",
            "confidence": "high",
            "license_source": "known_wikimedia_policy",
        }
    return None


def main() -> int:
    inv = json.loads(INV_PATH.read_text(encoding="utf-8"))
    sources = inv["sources"]
    mit_hits = []
    permissive_hits = []

    for src in sources:
        url = src["url"]
        info: dict = {
            "license": "unknown",
            "license_notes": "",
            "confidence": "low",
            "license_source": "unset",
        }

        if url in KNOWN:
            info.update(KNOWN[url])
            info["license_source"] = "curated_override"
        else:
            wiki = annotate_wikipedia(url)
            if wiki:
                info.update(wiki)
            else:
                ds = hf_dataset_id(url)
                if ds:
                    lic, how = fetch_hf_license(ds)
                    info["license"] = normalize_license(lic) if lic else "unknown"
                    info["license_source"] = how
                    info["confidence"] = "high" if lic else "low"
                    info["hf_dataset_id"] = ds

        # bible github blobs
        if "bible-corpus" in url and info["license"] == "unknown":
            info.update(
                {
                    "license": "cc0-1.0",
                    "license_notes": "christos-c/bible-corpus; verify",
                    "confidence": "medium",
                    "license_source": "curated_override",
                }
            )

        lic = normalize_license(info["license"])
        src["license"] = lic
        src["license_notes"] = info.get("license_notes", "")
        src["license_confidence"] = info.get("confidence", "low")
        src["license_source"] = info.get("license_source", "unset")
        if "hf_dataset_id" in info:
            src["hf_dataset_id"] = info["hf_dataset_id"]

        src["is_mit"] = is_mit(lic)
        src["is_permissive_open"] = is_permissive(lic) and lic not in {"mixed", "unknown", "other"}
        if lic == "mixed":
            src["is_permissive_open"] = False

        if src.get("role") == "train_candidate":
            if src["is_mit"]:
                mit_hits.append(src)
            if src["is_permissive_open"]:
                permissive_hits.append(src)

        # Update license_status field from earlier draft
        if src["is_mit"]:
            src["license_status"] = "mit"
        elif src["is_permissive_open"]:
            src["license_status"] = "permissive_open"
        elif lic.startswith("cc-by-sa"):
            src["license_status"] = "copyleft_sharealike"
        elif lic in {"mixed", "unknown"}:
            src["license_status"] = "needs_review"
        else:
            src["license_status"] = lic

    inv["license_policy"] = {
        "requested": "MIT",
        "mit_train_candidate_count": len(mit_hits),
        "permissive_open_train_candidate_count": len(permissive_hits),
        "permissive_open_definition": sorted(PERMISSIVE),
        "note": (
            "Strict MIT leaves almost no multilingual pretrain mass. "
            "Use permissive_open (MIT/Apache/BSD/CC0/ODC-By/CC-BY) for ingest; "
            "exclude CC-BY-SA Wikipedia dumps unless SA is acceptable."
        ),
    }
    inv["status"] = "draft_licensed"

    OUT_PATH.write_text(json.dumps(inv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Keep primary inventory path updated too.
    INV_PATH.write_text(json.dumps(inv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    allow = {
        "schema_version": 1,
        "kind": "plan_a_permissive_open_allowlist",
        "policy": inv["license_policy"],
        "mit_only": [
            {"source_id": s["source_id"], "name": s["name"], "url": s["url"], "license": s["license"]}
            for s in mit_hits
        ],
        "permissive_open_train_candidates": [
            {
                "source_id": s["source_id"],
                "name": s["name"],
                "url": s["url"],
                "license": s["license"],
                "priority": s.get("priority"),
                "language_hint": s.get("language_hint"),
            }
            for s in permissive_hits
        ],
    }
    ALLOW_PATH.write_text(json.dumps(allow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"MIT train candidates: {len(mit_hits)}")
    for s in mit_hits:
        print(f"  MIT  {s['name']}  {s['url']}")
    print(f"Permissive-open train candidates: {len(permissive_hits)}")
    for s in permissive_hits:
        print(f"  {s['license']:12}  {s['name']}")
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {ALLOW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
