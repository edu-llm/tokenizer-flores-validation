# DATA_PLAN — where the corpus comes from and how it reaches S3

**Status:** Sourcing plan for the 6-language Plan A rescope in [02-tokenizer-training.md](02-tokenizer-training.md).
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

**The scale run needs two tiers staged, not one.** `scale` carries the headline gigatoken
BPE vs SuperBPE comparison at 100k vocab. `pilot` carries the five-config trainer
cross-check, which holds bytes at 96 MB and moves only vocab to 100k — the one known
divergence between trainers grows with vocab size, not corpus size, so that is the axis
worth isolating. Stage `corpus/scale/` and `corpus/pilot/` both.

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

**gigatoken is now the default trainer and reads the opposite shape.** It takes a single
mmapped file, so `scripts/build_plan_a_research_corpus.py` emits *both* views of the same
bytes: `langs/<lang>.txt` per language, and `train.txt`, their byte-identical
concatenation in `PLAN_A_CODES` order. Two consequences:

- **`train.txt` must stay outside `langs/`.** The official trainer globs `*.txt` over the
  corpus directory and would otherwise count every language twice.
- **On the gigatoken path the manifest equality check replaces the
  `meta.json:train_files` assertion.** gigatoken performs no file selection, so the
  whole-file hazard above cannot arise there; the equal-byte guarantee moves entirely
  into corpus construction, and `tests/test_plan_a_corpus_balance.py` is where it is
  proven. The `train_files` check still applies to the official arm of the cross-check.

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
this table. `hat_Latn`'s 1.87× is the thinnest margin in the set, and an earlier revision
of this section called it the only real risk to the scale tier. **That has since been
measured and the alarm was wrong.** On extracted text, a full 160 MB `hat_Latn` pull
consumed 56,461 of the 226,754 available documents, implying
roughly 0.64–0.70 GB of text, or about **4× the scale budget**
([04 §6](04-trainer-cross-check.md)). Treat the first full-budget pull of any language as
a measurement rather than a formality, but the scale tier is not availability-bound.

**Train split only.** `hat_Latn` and `swh_Latn` each carry a `test` split (2,282 and
12,669 rows). Those stay out of training text.

### 2.2 Line endings and pull headroom

**Pull 176,000,000 bytes per language for the scale tier, not 160,000,000.** The reason is
not availability — it is that the number of bytes reaching the trainer is smaller than the
number of bytes on disk, and the guard that was supposed to catch that does not.

`scripts/pull_fineweb2_lang_samples.py:69` opens the destination with
`dest.open("w", encoding="utf-8")` — text mode, so on Windows every `\n` is written as
`\r\n`. `scripts/build_plan_a_research_corpus.py:145` then reads each line back as
`line.rstrip("\r\n").encode("utf-8") + b"\n"`, returning exactly one byte per line. Every
language loses its own line count. Against the 160,000,000-byte scale budget, the six
files staged on 2026-08-01 all fall short:

| Language | On disk | Usable LF bytes | Shortfall |
|---|---:|---:|---:|
| `hun_Latn` | 160,024,290 | 159,992,318 | **−7,682** |
| `hin_Deva` | 160,020,931 | 159,996,210 | **−3,790** |
| `swh_Latn` | 160,053,330 | 159,999,423 | −577 |
| `hat_Latn` | 160,055,993 | 159,999,532 | −468 |
| `zho_Hans` | 160,044,390 | 159,999,780 | −220 |
| `eng_Latn` | 160,051,411 | 159,999,884 | −116 |

**Nothing catches this.** The guard at `build_plan_a_research_corpus.py:128` compares the
budget against `source.stat().st_size`, which is CRLF-inflated, so it passes. The read
loop then exhausts the file, `padding = budget - written` comes out large, and the
remainder is filled with newlines. `max(bytes) == min(bytes)` still holds, the manifest
still records `equal_bytes_verified: true`, and `tests/test_plan_a_corpus_balance.py`
still passes — its fixture writes LF-only sources with `write_bytes`, so this path is
never exercised. The magnitude is trivial, 0.005% at worst. The broken guard is not: it is
precisely the silent-shrink mechanism §3.1 and [02 §1.1](02-tokenizer-training.md) exist
to prevent.

**This is not a genuine shortfall, so the budget does not move.** §3.1 permits dropping
`bytes_per_language` for all six when the ladder is exhausted; that rule does not apply
here. Fix the pull instead.

Three code changes, and then re-pull:

1. `scripts/pull_fineweb2_lang_samples.py:69` — open with `newline="\n"` so on-disk bytes
   match the counter the pull already keeps.
2. `scripts/build_plan_a_research_corpus.py`, in `build_language_file` — bound the padding.
   Its docstring already states that padding exists only to absorb the 1–3 bytes lost when
   a UTF-8 cut lands mid-character, so raising when `padding > 3` cleanly separates that
   legitimate case from a source that ran dry. That is what the `st_size` check was trying
   and failing to do; keep the existing check as a cheap pre-filter.
3. `tests/test_plan_a_corpus_balance.py` — add a CRLF fixture. A source written with
   `\r\n` whose LF content cannot fill the budget must raise, not pad.

Run the pull itself under **`.venv-benchmark/Scripts/python.exe`** (Python 3.11). The
repo's default environment is Python 3.14, where `datasets` fails on import with a `dill`
pickling error. Do **not** pass `--skip-existing`: the files on disk are the CRLF ones
being replaced.

```bash
.venv-benchmark/Scripts/python.exe scripts/pull_fineweb2_lang_samples.py \
    --max-bytes-per-lang 176000000 \
    --output-dir artifacts/plan_a/raw/fineweb2_samples
```

10% headroom is ample: the pull stops before the next document would exceed the cap, so it
always lands somewhat under, and 16 MB of slack covers any single document. Disk cost is
~1.06 GB.

**Pilot regression check.** The pilot corpus built from the CRLF sources has `train.txt`
sha256 `f92c6f9db02079646f381368cf58d8be8debd4a3e48a5d5e8b7f3973ac530b7c`. The fix changes
line endings on disk but not the text the builder extracts, so re-pulling the same
documents in the same order should reproduce that hash exactly. If it differs, the
upstream stream is not deterministic and the already-published pilot cross-check has to be
re-run — escalate rather than absorb.

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

**No script implements this yet**, and [00 §3.3](00-data-to-s3.md) makes it a Phase A gate
item. It is the one genuinely new module in the staging work. If it slips, say so
explicitly — a gate with no implementation passes quietly, which is worse than a gate that
fails.

---

## 7. S3 layout and staging

### 7.1 Fill these in

Nothing account-specific is committed. Set these before running anything:

```
CORPUS_S3_ROOT=          # s3://<your-bucket>/<your-prefix>, no trailing slash
AWS_REGION=              # match the region the EC2 training host runs in
```

Everything below derives from `$CORPUS_S3_ROOT`. Put it wherever suits your account —
the layout is what matters, not the bucket.

The scale tier runs on a **single memory-optimized EC2 instance with Docker** — no ECR, no
Batch. The image is built on the host from the repo root
(`docker/tokenizer-benchmark/Dockerfile` already builds both trainers, gigatoken into its
own `/opt/venv-gigatoken`). Batch stays on the table for Plan B, which actually needs
fan-out; five sequential training runs do not.

### 7.2 Key layout

Keyed by tier so smoke / pilot / scale never collide:

```
$CORPUS_S3_ROOT/
  raw/fineweb2/<tier>/<lang>.txt          # rung-1 pull; upload only if provenance
  raw/fallback/<rung>/<tier>/<lang>.txt   #   must be reproducible from S3 (see §7.5)

  corpus/scale/langs/<lang>.txt           # 6 files, exactly 160,000,000 bytes each
  corpus/scale/manifest.json              # per-language bytes + sha256, exact total
  corpus/pilot/langs/<lang>.txt           # 6 files, exactly 16,000,000 bytes each
  corpus/pilot/manifest.json

  cr_dev/<lang>.txt                       # premium-calibration dev, from FLORES dev
  tokenizers/<tier>/<arm>/                # arm = bpe | superbpe; written by the job
  results/<tier>/
```

`corpus/<tier>/langs/<lang>.txt` is one file per language for the reason in §1.1. The
`langs/` sub-prefix is what the official trainer receives as its `--corpus-dir`, so
nothing else may live in it.

[00 §2](00-data-to-s3.md) declares itself an extension of this section and still shows the
older flat `corpus/<tier>/<lang>.txt`. This section is the authority; sync 00 when it is
next touched.

**Do not upload `train.txt`.** It is a byte-identical concatenation of `langs/`, so
uploading it doubles the transfer for nothing. Reconstruct it on the instance by
concatenating in `manifest.json:train_txt.order` and verify against
`manifest.json:train_txt.sha256`. That check is exact: if the hash matches, the file is
right, and if it does not, nothing downstream should run.

### 7.3 Steps

Run these once per tier — `scale` first, then `pilot`. The builder writes to a fixed
output directory and **overwrites in place**, so each tier must be built and uploaded
before the next is built.

1. **Pull** into `artifacts/plan_a/raw/fineweb2_samples/`, at the headroom budget and under
   `.venv-benchmark` — see §2.2 for both, neither is optional.
2. **Build** the equal-byte corpus into `artifacts/plan_a/corpus/`. `manifest.json` records
   per-language bytes, per-file sha256, `padding_bytes`, the exact total, and the rungs
   used per language (§3.1).
   ```bash
   python scripts/build_plan_a_research_corpus.py --tier scale
   ```
3. **Upload, manifest last:**
   ```bash
   aws s3 sync artifacts/plan_a/corpus/langs/ "$CORPUS_S3_ROOT/corpus/$TIER/langs/"
   aws s3 cp artifacts/plan_a/corpus/manifest.json \
     "$CORPUS_S3_ROOT/corpus/$TIER/manifest.json"
   ```
   The manifest's presence is the signal that the prefix is complete, and the downstream
   jobs are written to trust that. It goes up as its own final command, never in the sync.
4. **Verify the round trip.** Re-download to a scratch directory and re-hash against the
   manifest. Reuse `src/benchmark.py:sha256_file` — it is already the project's hashing
   helper (see `scripts/run_plan_b_materialize_shards.py:21`). Do not add a second one.
5. **Consume.** The training host pulls `corpus/<tier>/` to local storage, reconstructs and
   verifies `train.txt` per §7.2, and pushes `tokenizers/` and `results/` back. Command
   arguments and result schemas are identical to a local run, as `README.md` already
   promises.

### 7.4 Operational choices you own

Not prescribed here — decide per account: region, versioning, encryption, and a lifecycle
rule expiring `raw/`. The instance role needs `s3:GetObject` on `corpus/*` and `cr_dev/*`,
and `s3:PutObject` on `tokenizers/*` and `results/*`.

One lifecycle rule that is not optional in practice: **abort incomplete multipart uploads
after 7 days.** A failed large sync otherwise bills indefinitely for parts that no listing
shows.

`boto3` is deliberately **not** in `requirements.txt`. The `aws` CLI covers every step
above, so staging adds no Python dependency to the training image.

**Two prerequisites owned outside the data team.** Raise them at the start; do not wait on
them to begin pulling.

1. **`supergigatoken` commit `00e61db` is not pushed.** `git ls-remote --heads origin` on
   that repo returns only `main` at `c64233f`; the pinned commit exists on the local
   `superbpe-stage1-pretokenizer` branch. `docker/tokenizer-benchmark/Dockerfile:7` pins
   `GIGATOKEN_COMMIT` to it and clones from GitHub, so **the image cannot build on EC2
   until that branch is pushed.**
2. **No AWS infrastructure exists** — no bucket, no IAM roles, no IaC of any kind — and the
   `aws` CLI is not installed on the dev laptop.

### 7.5 Transfer size

The scale corpus is 6 × 160 MB = 960 MB and the pilot corpus 6 × 16 MB = 96 MB, plus
`cr_dev`. Not uploading `train.txt` (§7.2) halves what would otherwise go up. A
same-region round trip is minutes and cents.

Raw pulls are larger — 6 × 176 MB ≈ 1.06 GB at the headroom budget — and stay local by
default. Upload `raw/` only if the ladder fired and the provenance needs to be
reproducible from S3.

---

## 8. Execution checklist

Prerequisites (§7.4): the `supergigatoken` branch is pushed, and a bucket, region and
instance role exist.

**Fix and pull**

- [ ] Apply the three code changes in §2.2 — pull `newline="\n"`, the `padding > 3` bound,
      and the CRLF test fixture.
- [ ] `python -m pytest tests/ -q` passes, including the new CRLF case.
- [ ] Re-verify availability for all six configs (§9.1). Numbers below the tier budget
      mean stop and re-plan, not proceed.
- [ ] Pull rung 1 at **176,000,000 bytes per language**, `train` split only, under
      `.venv-benchmark`, without `--skip-existing`.
- [ ] Confirm **no shortfall**: every language reports `truncated: true`, i.e. the source
      had more than we took. A `truncated: false` means that language ran dry — go to §3.
- [ ] If the ladder fired, dedup across rungs and record rungs per language.

**Build, both tiers**

- [ ] Build `scale`; manifest reports six languages at exactly 160,000,000 bytes, total
      960,000,000, and `padding_bytes` ≤ 1 for each.
- [ ] Upload `scale`, then build `pilot` — the builder overwrites in place.
- [ ] `pilot` manifest reports six at exactly 16,000,000, total 96,000,000.
- [ ] `pilot` `train_txt.sha256` still equals `f92c6f9d…` (§2.2), or the difference is
      escalated rather than absorbed.
- [ ] Build `cr_dev` from FLORES `dev`.
- [ ] Run the FLORES `devtest` overlap check (§6) — **note it is unimplemented**; if it has
      not been written, record that rather than ticking this box.

**Stage**

- [ ] Upload each tier to `$CORPUS_S3_ROOT/corpus/<tier>/langs/`, manifest last.
- [ ] Verify the round trip by sha256 for all twelve language files and `cr_dev`.
- [ ] `manifest.json` is the newest object in each prefix.

**Hand off**

- [ ] On the instance, `train.txt` reconstructed from `langs/` in
      `manifest.json:train_txt.order` matches `manifest.json:train_txt.sha256`.
- [ ] For the official arm of the cross-check only, confirm `meta.json` `train_files`
      lists all six per-language files — the direct regression check on §1.1. On the
      gigatoken arms the manifest equality check stands in its place.

Steps map onto the scripts named in PRD §5. This document is the sourcing contract either
way.

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
