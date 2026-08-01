# DATA_PLAN — where the corpus comes from and how it reaches S3

**Status:** Sourcing plan for the 6-language Plan A rescope in [PRD.md](PRD.md).
**Scope:** Acquisition and staging of tokenizer *training* text only. Does not cover
FLORES evaluation data, the locked 12-language efficiency scope
(`src/load_flores.py:LANGUAGES`), or the 18-language Zipf study (`src/zipf_langs.py`).

Every byte count in this document is `num_bytes_original_files` from the Hugging Face
datasets-server size API, read on **2026-08-01**. They are reproducible — see §9.
They are *compressed parquet* bytes, not extracted text bytes; §2 explains why the
difference matters.

---

## 1. Scope and budget

Six languages, one reference:

| Code | Language | Region | Script |
|------|----------|--------|--------|
| `eng_Latn` | English (premium reference) | — | Latin |
| `hun_Latn` | Hungarian | Europe | Latin |
| `zho_Hans` | Mandarin | Asia | Han |
| `hin_Deva` | Hindi | Asia | Devanagari |
| `swh_Latn` | Swahili | Africa | Latin |
| `hat_Latn` | Haitian Creole | Americas | Latin |

The corpus policy is **equal bytes per language**. Budgets live in
`configs/benchmarks/tokenizer_local.json` — that file is the authority, this table is a
convenience copy:

| Tier | Bytes / language | Total corpus |
|------|-----------------|--------------|
| smoke | 2,000,000 | 12,000,000 |
| pilot | 16,000,000 | 96,000,000 |
| scale | 160,000,000 | 960,000,000 |

### 1.1 One file per language — not negotiable

The corpus directory must hold **one `.txt` per language**, and `--num-bytes` must equal
the manifest total **exactly**.

The official trainer's `get_files_with_num_bytes` (`.cache/superbpe/utils.py:147`)
selects training data at **whole-file granularity** over `*.txt`, shuffled under
`random.seed(0)`, truncating only the last file. A single language-sorted `train.txt`
plus any `--num-bytes` below its size trains on a byte prefix — that is, on the
alphabetically-first languages only. See PRD §1.3.

Flattening the per-language files back into one `train.txt` reintroduces that bug
silently. Nothing errors; the tokenizer is just wrong.

---

## 2. Primary source — FineWeb-2 for all six languages

One web-crawl source across the whole set, so the corpus differs by **language and
nothing else**. A mixed diet — Creole from bibles, Hungarian from web — would confound
any cross-language premium difference with domain.

`HuggingFaceFW/fineweb-2` is ODC-By 1.0 and ungated. It has **no English**; English comes
from `HuggingFaceFW/fineweb` config `sample-10BT`, which is already what
`scripts/pull_fineweb2_lang_samples.py:62` does.

| Code | Dataset · config | Bytes available | Rows | × scale budget |
|------|------------------|----------------:|-----:|---------------:|
| `eng_Latn` | [`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) · `sample-10BT` | 30,639,384,917 | 14,868,862 | 191× |
| `hun_Latn` | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) · `hun_Latn` | 98,567,072,047 | 49,970,765 | 616× |
| `zho_Hans` | `HuggingFaceFW/fineweb-2` · `cmn_Hani` | 1,622,071,098,141 | 636,092,280 | 10,138× |
| `hin_Deva` | `HuggingFaceFW/fineweb-2` · `hin_Deva` | 34,361,638,504 | 22,151,227 | 215× |
| `swh_Latn` | `HuggingFaceFW/fineweb-2` · `swh_Latn` | 1,443,539,553 | 1,218,969 | 9.0× |
| `hat_Latn` | `HuggingFaceFW/fineweb-2` · `hat_Latn` | 299,886,824 | 226,754 | **1.87×** |

Note the config-name mappings: Mandarin is `cmn_Hani`, not `zho_Hans`.

### 2.1 Two caveats

**Parquet bytes are not text bytes.** The figures above are compressed columnar files
carrying eleven columns, only one of which is `text`. Extracted UTF-8 text will be a
different number in either direction. This is exactly why PRD §4 requires the pull step
to record **actual extracted text bytes** and hard-fail on shortfall rather than trust
this table. `hat_Latn`'s 1.87× is the thinnest margin in the set and the only real risk
to the scale tier — treat the first full-budget `hat_Latn` pull as the load-bearing
experiment, not a formality.

**Train split only.** `hat_Latn` and `swh_Latn` each carry a `test` split (2,282 and
12,669 rows). Those stay out of training text.

---

## 3. Fallback ladder

Fires **only** when a language's extracted text falls short of the tier budget. Ordered
by license cleanliness first, then by distance from rung 1's domain (web crawl).

| Rung | Source | License | `hat_Latn` | `swh_Latn` |
|------|--------|---------|-----------:|-----------:|
| 1 | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ODC-By 1.0 | 299,886,824 | 1,443,539,553 |
| 2 | [`HPLT/HPLT2.0_cleaned`](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | CC0-1.0 | 414,174,177 <br>(212,686 rows) | 2,872,996,669 <br>(1,373,860 rows) |
| 3 | [`cis-lmu/GlotCC-V1`](https://huggingface.co/datasets/cis-lmu/GlotCC-V1) · `hat-Latn` / `swh-Latn` | CC0-1.0 | 35,582,415 <br>(13,576 rows) | 87,885 rows <br>(size not reported) |
| 4 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) · `20231101.ht` / `20231101.sw` | CC BY-SA 3.0 **and** GFDL | 21,993,952 <br>(70,159 rows) | 35,936,177 <br>(78,587 rows) |
| 5 | [`Adeptschneider/CiviVox-Swahili-text-corpus-v2.0`](https://huggingface.co/datasets/Adeptschneider/CiviVox-Swahili-text-corpus-v2.0) | Apache-2.0 | — | 224,199,954 <br>(1,542,912 rows) |

**Rung 2 is the one that matters.** HPLT 2.0 alone clears the `hat_Latn` scale budget by
2.6×; rungs 1+2 together give roughly 4.5× headroom on the binding language. It uses the
same `<lang>_<Script>` config naming as FineWeb-2, so `hat_Latn` and `swh_Latn` work
directly. HPLT 2.0 is superseded by HPLT 3.0, which is also CC0 but is **not hosted on
Hugging Face** — direct download from the HPLT project only, so it is not in the ladder.

Rung 3's `swh-Latn` config reports no size (row count only); measure it at pull time.
Note GlotCC uses a hyphen (`hat-Latn`), not an underscore.

Rung 5 is Swahili-only and AfriBERTa-derived, so it is **news domain, not web crawl** —
the largest domain jump in the ladder. Last resort for Swahili, which has 9× headroom at
rung 1 anyway.

### 3.1 Rules

**Deduplicate across rungs.** Rungs 1–3 are all Common Crawl / Internet Archive
derivatives and will overlap. Dedup at document level *before* counting bytes toward the
budget, or the shortfall check measures nothing.

**Record every rung used**, per language, in the corpus manifest. If `hat_Latn` draws on
rungs 1+2 while `hun_Latn` draws on rung 1 alone, that asymmetry is a known confound and
must be visible in the artifact rather than buried in a shell history.

**Prefer shrinking the tier to unbalancing the corpus.** If the ladder is exhausted for
one language, drop `bytes_per_language` for **all six**. Never let one language run short
— that is the skew PRD §1.1 exists to eliminate.

---

## 4. License gate

Admitted: **ODC-By, CC0, CC-BY, Apache-2.0, CC BY-SA**.

CC BY-SA (Wikipedia, rung 4) is admitted but ranked last: it is copyleft, so any released
corpus inherits an attribution and share-alike obligation that the other rungs do not
carry. Note that `wikimedia/wikipedia` declares **CC BY-SA 3.0 and GFDL** jointly, not
CC BY-SA 4.0 — the dual license is the reason to keep rung 4 as a last resort rather than
a routine top-up.

This is stricter than the repo's existing `--research-use-all-train` default in
`scripts/stage_plan_a_selected_sources.py`, which was written for the old 16-language set
and admits everything but eval data. The tighter gate here keeps the 6-language results
publishable without license caveats.

---

## 5. Rejected sources

This section exists because the obvious search results are traps. Anyone re-researching
this in six weeks will surface these first.

| Source | Why rejected |
|--------|--------------|
| [`flax-community/swahili-safi`](https://huggingface.co/datasets/flax-community/swahili-safi) | **Do not use.** Top result for "large Swahili corpus" at ~3.5 GB. Its card lists English Wikipedia machine-translated to Swahili via m2m100 as a component. Synthetic translationese would directly corrupt a tokenizer-efficiency measurement — it is the one contaminant this study cannot absorb. No license declared either. |
| [`swahili`](https://huggingface.co/datasets/swahili) (canonical HF LM dataset) | Lowercased, punctuation stripped, sentence markers inserted. Unusable for tokenizer training regardless of license. |
| [Leipzig Corpora](https://wortschatz.uni-leipzig.de/en/download/Haitian) — `hat_wikipedia_2011`, `hat_community_2017`, Swahili | CC BY-NC — fails the gate. Leipzig also asks that archives not be mirrored. Small regardless: `hat_wikipedia_2011` is 20,650 sentences / 232,093 tokens. |
| [`jhu-clsp/kreyol-mt`](https://huggingface.co/datasets/jhu-clsp/kreyol-mt) · `hat-eng` (80,185,636 bytes) | License "other"; the card states the full release is pending an LDC release. It is also parallel MT bitext, so the Creole side is translationese. |
| [`lelapa/Inkuba-Mono`](https://huggingface.co/datasets/lelapa/Inkuba-Mono) | No license declared and auto-gated, despite a substantial Swahili portion (`swa/data.txt`). |
| [`statmt/cc100`](https://huggingface.co/datasets/statmt/cc100) | No declared `ht` or `sw` configs on HF (only `am`, `sr`, `ka`); license "unknown". |

---

## 6. Contamination check

FLORES-200 `devtest` is the eval set (`scripts/eval_plan_a_flores_compression.py`). Its
source text is drawn from Wikinews / Wikijunior / Wikivoyage rather than Wikipedia
proper, but every ladder rung is web-derived and the risk is not zero — particularly at
rung 4.

Run an **n-gram overlap check** between the built corpus and FLORES `devtest`, per
language, and report it alongside the corpus manifest. This is a measurement, not an
assumption; a clean result is worth recording, and a dirty one needs to surface before
the scale tier, not after.

---

## 7. S3 layout and staging

### 7.1 Fill these in

Nothing account-specific is committed. Set these before running anything:

```
CORPUS_S3_ROOT=          # s3://<your-bucket>/<your-prefix>
AWS_REGION=              # match the ECR / Batch region used by docker/tokenizer-benchmark
```

Everything below derives from `$CORPUS_S3_ROOT`. Put it wherever suits your account —
the layout is what matters, not the bucket.

### 7.2 Key layout

Keyed by tier so smoke / pilot / scale never collide:

```
$CORPUS_S3_ROOT/
  raw/fineweb2/<tier>/<lang>.txt          # rung-1 pull, one file per language
  raw/fallback/<rung>/<tier>/<lang>.txt   # present only if the ladder fired
  corpus/<tier>/<lang>.txt                # equal-byte trainer input, one per language
  corpus/<tier>/manifest.json             # per-language bytes + sha256, exact total
  cr_dev/<lang>.txt                       # premium-calibration dev, from FLORES dev
  tokenizers/<tier>/<arm>/                # arm = bpe | superbpe
  results/<tier>/
```

`corpus/<tier>/<lang>.txt` is one file per language for the reason in §1.1.

### 7.3 Steps

1. **Pull** into `artifacts/plan_a/raw/` — locally for smoke and pilot, on EC2 for scale.
2. **Build** the equal-byte corpus. Emit `manifest.json` with per-language bytes, per-file
   sha256, the exact total, and the rungs used per language (§3.1).
3. **Upload:**
   ```bash
   aws s3 sync artifacts/plan_a/research_cpu/corpus/ "$CORPUS_S3_ROOT/corpus/<tier>/" \
     --exclude manifest.json
   aws s3 cp artifacts/plan_a/research_cpu/corpus/manifest.json \
     "$CORPUS_S3_ROOT/corpus/<tier>/manifest.json"
   ```
   Upload `manifest.json` **last**, so its presence signals a complete prefix.
4. **Verify the round trip.** Re-download to a scratch directory and re-hash against the
   manifest. Reuse `src/benchmark.py:sha256_file` — it is already the project's hashing
   helper (see `scripts/run_plan_b_materialize_shards.py:21`). Do not add a second one.
5. **Consume.** The Batch job (`docker/tokenizer-benchmark/Dockerfile`) pulls
   `corpus/<tier>/` to local job storage and pushes `tokenizers/` and `results/` back.
   Command arguments and result schemas are identical to a local run, as `README.md`
   already promises.

### 7.4 Operational choices you own

Not prescribed here — decide per account: region, versioning, encryption, and a lifecycle
rule expiring `raw/`. The Batch job role needs `s3:GetObject` on `corpus/*` and
`cr_dev/*`, and `s3:PutObject` on `tokenizers/*` and `results/*`.

`boto3` is deliberately **not** in `requirements.txt`. The `aws` CLI covers every step
above, so staging adds no Python dependency to the training image.

### 7.5 Transfer size

The scale corpus is 6 × 160 MB = 960 MB. A same-region round trip is minutes and cents.
Raw pulls are larger and stay local by default — upload `raw/` only if the ladder fired
and the provenance needs to be reproducible from S3.

---

## 8. Execution checklist

- [ ] Re-verify availability for all six configs (§9.1). Numbers below the tier budget
      mean stop and re-plan, not proceed.
- [ ] Pull rung 1 at the tier's `bytes_per_language`, `train` split only.
- [ ] Confirm **no shortfall**: every language reports `truncated: true`, i.e. the source
      had more than we took. A `truncated: false` means that language ran dry — go to §3.
- [ ] If the ladder fired, dedup across rungs and record rungs per language.
- [ ] Build the equal-byte corpus; assert `max(bytes) == min(bytes)` across languages.
- [ ] Run the FLORES `devtest` overlap check (§6).
- [ ] Upload to `$CORPUS_S3_ROOT`, manifest last.
- [ ] Verify the round trip by sha256.
- [ ] Run the tier: `scripts/run_plan_a_tokenizer_pair.py` with `--num-bytes` equal to the
      manifest total.
- [ ] Confirm `meta.json` `train_files` lists all six per-language files — the direct
      regression check on §1.1.

Steps map onto the scripts named in PRD §5; several of those still carry the old
16-language lists and are rewritten as part of that work. This document is the sourcing
contract either way.

---

## 9. Reproducing the numbers

### 9.1 Byte counts

```bash
curl -s "https://datasets-server.huggingface.co/size?dataset=HuggingFaceFW%2Ffineweb-2&config=hat_Latn" \
  | python -m json.tool
```

Repeat with `swh_Latn`, `hin_Deva`, `hun_Latn`, `cmn_Hani`; with
`dataset=HuggingFaceFW%2Ffineweb&config=sample-10BT`; and for the rung 2–5 datasets. The
figure quoted in this document is `size.config.num_bytes_original_files`.

### 9.2 Licenses

```bash
curl -s https://huggingface.co/api/datasets/HPLT/HPLT2.0_cleaned | python -m json.tool
```

Confirm `HPLT/HPLT2.0_cleaned` and `cis-lmu/GlotCC-V1` report `cc0-1.0`,
`HuggingFaceFW/fineweb-2` and `HuggingFaceFW/fineweb` report `odc-by`,
`wikimedia/wikipedia` reports `["cc-by-sa-3.0", "gfdl"]`, and `lelapa/Inkuba-Mono` still
reports no license with `gated: auto`. Licenses change; re-check before a publishable run.
