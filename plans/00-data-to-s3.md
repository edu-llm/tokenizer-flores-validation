# Plan 0 — Acquire the corpora and stage them in S3

**Status:** Not implemented. No acquisition script currently targets the 6-language set.
**Blocks:** Plan A scale tier ([02](02-tokenizer-training.md)) and all of Plan B ([03](03-model-pretraining.md)).
**Depends on:** [01-data-sourcing.md](01-data-sourcing.md) for which dataset each language comes
from and which fallback rungs are admissible.

This is the runbook. `01-data-sourcing.md` decides *what* to pull and answers *why that
source*; this document is the ordered set of steps that ends with verified bytes in S3.

---

## 0. The two corpora are different sizes and different policies

Conflating them is the mistake this plan exists to prevent. See
[03-model-pretraining.md §4](03-model-pretraining.md).

| | Tokenizer corpus | Pretraining pools |
|---|---|---|
| Consumer | Plan A — BPE / SuperBPE trainers | Plan B — OLMo-1B pretraining |
| Size (scale) | **960 MB** | **65.3 GB unique** → 78 GB after repetition |
| Policy | equal bytes per language | UniMax, epoch cap N=4 |
| Per language | 160 MB × 6, exactly equal | unequal; set by availability and the allocator |
| Budget unit | bytes | tokens (20B, measured on the BPE arm) |
| Authority | `configs/benchmarks/tokenizer_local.json` | `configs/benchmarks/plan_b_olmo.json` + `mixture.json` |

**Phase A** (tokenizer corpus) is 1.5% of the bytes of Phase B and unblocks Plan A on its
own. Do it first and completely, then start Phase B.

### 0.1 Why Phase A is not just a slice of Phase B

It could be, and §3.2 draws it that way — but the ordering is forced. The Plan B mixture
depends on measured bytes/token, which depends on the scale tokenizer, which depends on the
Phase A corpus ([03 §6](03-model-pretraining.md)). So Phase A must complete before the
Phase B *allocation* is even computable. Phase B *acquisition* does not wait: pool sizes are
availability-bounded and tokenizer-independent for the two languages that bind, and generous
for the four that do not (§4.2).

---

## 1. Prerequisites

### 1.1 Environment

Nothing account-specific is committed. Set these once per shell:

```bash
export CORPUS_S3_ROOT=            # s3://<bucket>/<prefix> — no trailing slash
export AWS_REGION=                # must match the ECR / Batch region in docker/tokenizer-benchmark
export HF_HOME=                   # a disk with room for the Phase B pull; not C:\ on Windows
```

`CORPUS_S3_ROOT` is referenced by every command below. Put it wherever suits the account —
the layout in §2 is what matters, not the bucket name.

### 1.2 Disk

Phase B stages ~65 GB of extracted text locally before upload, and the HF datasets cache
holds compressed parquet alongside it. **Budget 200 GB free** on the pull host. This is the
reason §4.1 recommends EC2 rather than a laptop for Phase B.

### 1.3 IAM

The pull host needs `s3:PutObject` and `s3:ListBucket` on `$CORPUS_S3_ROOT/*`.
The Batch job role (`docker/tokenizer-benchmark/Dockerfile`) needs:

| Action | Prefix |
|---|---|
| `s3:GetObject` | `corpus/*`, `cr_dev/*`, `pretrain/*` |
| `s3:PutObject` | `tokenizers/*`, `results/*` |

### 1.4 Bucket settings

Not prescribed — decide per account: versioning, encryption, and a lifecycle rule expiring
`raw/`. One recommendation that is not optional in practice: **enable a lifecycle rule that
aborts incomplete multipart uploads after 7 days.** A failed 65 GB sync otherwise bills
indefinitely for parts no listing shows.

### 1.5 No new Python dependency

`boto3` is deliberately absent from `requirements.txt`. The `aws` CLI covers every step here,
so staging adds nothing to the training image. Do not add it.

---

## 2. S3 layout

Extends [01 §7.2](01-data-sourcing.md) with the `pretrain/` tree. Keyed by tier so
smoke / pilot / scale never collide:

```
$CORPUS_S3_ROOT/
  raw/fineweb2/<tier>/<lang>.txt           # rung-1 pull, one file per language
  raw/fallback/<rung>/<tier>/<lang>.txt    # present only if the ladder fired

  corpus/<tier>/<lang>.txt                 # PHASE A — equal-byte trainer input
  corpus/<tier>/manifest.json              # per-language bytes + sha256, exact total
  cr_dev/<lang>.txt                        # premium-calibration dev, from FLORES dev

  pretrain/pools/<lang>/part-*.jsonl.zst   # PHASE B — unique text pools
  pretrain/pools/manifest.json             # per-language pool bytes + sha256 + rungs
  pretrain/mixture.json                    # UniMax allocation (written after Plan A scale)
  pretrain/shards/<arm>/                   # materialized training shards

  tokenizers/<tier>/<arm>/                 # arm = bpe | superbpe
  results/<tier>/
```

Two layout rules that carry weight:

**`corpus/<tier>/<lang>.txt` is one file per language.** Not a concatenated `train.txt`. The
official trainer's `get_files_with_num_bytes` (`.cache/superbpe/utils.py:147`) selects at
whole-file granularity over `*.txt`, so a single sorted `train.txt` with `--num-bytes` below
its size trains on the alphabetically-first languages only. Nothing errors; the tokenizer is
just wrong. See [02 §1.3](02-tokenizer-training.md).

**`manifest.json` uploads last** in every prefix. Its presence is the signal that the prefix
is complete, and the downstream jobs are written to trust that.

---

## 3. Phase A — tokenizer corpus (960 MB)

### 3.1 What must be built first

Both scripts named here still carry the superseded 14/16-language lists and must be rewritten
per [02 §5](02-tokenizer-training.md) before this phase can run:

| Script | Current state | Needed |
|---|---|---|
| `scripts/pull_fineweb2_lang_samples.py` | 16-lang `LANG_SOURCES`, 8 MB default | `LANG_SOURCES` from `src/plan_a_langs.py`; per-language budget as an argument; record actual extracted text bytes; fail on shortfall |
| `scripts/build_plan_a_research_corpus.py` | greedy filesystem-order concatenation | equal-byte truncation on line boundaries; one file per language; hard-fail on short |
| `src/plan_a_langs.py` | does not exist | the single source of truth for the 6 codes |

### 3.2 Steps

Run the smoke tier end to end before the scale tier. It is 12 MB and exercises every step.

```bash
TIER=scale                        # smoke | pilot | scale
BPL=160000000                     # bytes_per_language for $TIER, from tokenizer_local.json
```

**1. Pull rung 1**, `train` split only, into `artifacts/plan_a/raw/fineweb2_samples/`.

```bash
python scripts/pull_fineweb2_lang_samples.py \
  --bytes-per-language "$BPL" \
  --out-dir artifacts/plan_a/raw/fineweb2_samples \
  --report artifacts/plan_a/raw/pull_report.json
```

**2. Check for shortfall.** Every language must report `truncated: true` — meaning the source
held more than we took. A `truncated: false` means that language ran dry: go to the fallback
ladder in [01 §3](01-data-sourcing.md), dedup across rungs at document level *before* counting
bytes, and record which rungs each language used.

`hat_Latn` at 1.87× the scale budget is the thinnest margin in the set. Treat its first
full-budget pull as the load-bearing experiment, not a formality.

**3. Build the equal-byte corpus.**

```bash
python scripts/build_plan_a_research_corpus.py \
  --raw-dir artifacts/plan_a/raw/fineweb2_samples \
  --bytes-per-language "$BPL" \
  --out-dir artifacts/plan_a/research_cpu/corpus
```

The manifest must assert `max(bytes) == min(bytes)` across the six languages and record
per-file sha256, the exact total, and the rungs used per language.

**4. Run the contamination check** against FLORES `devtest` — n-gram overlap, per language,
reported alongside the manifest ([01 §6](01-data-sourcing.md)). A clean result is worth
recording; a dirty one must surface before the scale tier, not after.

**5. Build CR-dev** (premium calibration, parallel FLORES `dev`):

```bash
python scripts/build_plan_a_cr_dev.py \
  --output-dir artifacts/plan_a/research_cpu/cr_dev
```

**6. Upload, manifest last.**

```bash
aws s3 sync artifacts/plan_a/research_cpu/corpus/ "$CORPUS_S3_ROOT/corpus/$TIER/" \
  --exclude manifest.json
aws s3 cp artifacts/plan_a/research_cpu/corpus/manifest.json \
  "$CORPUS_S3_ROOT/corpus/$TIER/manifest.json"
aws s3 sync artifacts/plan_a/research_cpu/cr_dev/ "$CORPUS_S3_ROOT/cr_dev/"
```

**7. Verify the round trip.** Re-download to a scratch directory and re-hash against the
manifest. Use `src/benchmark.py:sha256_file` — it is already the project's hashing helper
(see `scripts/run_plan_b_materialize_shards.py:21`). Do not add a second one.

### 3.3 Gate

Phase A is done when all four hold:

- [ ] `manifest.json` reports six languages at exactly `$BPL` bytes each.
- [ ] Round-trip sha256 matches for all six files.
- [ ] The FLORES `devtest` overlap check has run and its result is recorded.
- [ ] A smoke-tier `run_plan_a_tokenizer_pair.py` produced a `meta.json` whose `train_files`
      lists **all six** per-language files and whose `total_bytes` equals the manifest total.
      This is the direct regression check on the prefix-training hazard.

---

## 4. Phase B — pretraining pools (65.3 GB unique)

### 4.1 Where to run it

On EC2 in `$AWS_REGION`, not locally. 65 GB of extracted text over a residential
connection is measured in days; from EC2 to same-region S3 it is minutes, and transfer in
to S3 is free either way. An `m7i.2xlarge` with a 300 GB gp3 volume is sufficient — the pull
is I/O- and decompression-bound, not compute-bound.

### 4.2 Per-language pool targets

From [03 §5.2](03-model-pretraining.md), at the 3.9 bytes/token planning constant:

| Language | Unique pool | Bounded by | Acquire |
|---|---:|---|---:|
| `hat_Latn` | 0.772 GB | **availability** — the whole ladder | 0.772 GB (all of it) |
| `swh_Latn` | 4.577 GB | **availability** — the whole ladder | 4.577 GB (all of it) |
| `eng_Latn` | 14.98 GB | allocation | 17.2 GB |
| `hin_Deva` | 14.98 GB | allocation | 17.2 GB |
| `hun_Latn` | 14.98 GB | allocation | 17.2 GB |
| `zho_Hans` | 14.98 GB | allocation | 17.2 GB |
| **Total** | **65.3 GB** | | **~74 GB** |

**Acquire the four unbounded languages with ~15% headroom.** The 14.98 GB figure derives from
a *planning* bytes/token of 3.9; the real value is measured only after the Plan A scale
tokenizer exists ([03 §6](03-model-pretraining.md)). Headroom means a revised figure changes
how much is *sampled* from each pool, not whether the pull must be repeated. Re-pulling 15 GB
because the constant moved 8% is the avoidable failure here.

The two availability-bound languages get no headroom because there is none to get — they are
drawn in full at any bytes/token, which is exactly why they are the two the epoch cap acts on.

### 4.3 What must be built first

There is currently **no pool acquisition script**. `pull_fineweb2_lang_samples.py` writes flat
`.txt` for the tokenizer corpus and is the wrong shape for 65 GB. Needed:

- `scripts/pull_plan_b_pools.py` — streams the HF dataset per language, writes sharded
  `part-*.jsonl.zst` preserving document boundaries, walks the fallback ladder when rung 1
  runs short, dedups across rungs at document level, and emits `pretrain/pools/manifest.json`
  with per-language bytes, per-shard sha256, and rungs used.
- `src/plan_b_mixture.py` and `scripts/build_plan_b_mixture.py` — [03 §7](03-model-pretraining.md).

Document boundaries must survive the pull. The tokenizer corpus can be truncated on a line
boundary because a tokenizer trainer is order-insensitive; a pretraining sampler is not, and
the UniMax repetition on `hat_Latn` and `swh_Latn` operates on documents.

### 4.4 Steps

1. **Pull each language's pool** to the §4.2 target, `train` split only. Dedup across rungs
   before counting bytes toward the target, or the shortfall check measures nothing.
2. **Record every rung used, per language.** If `hat_Latn` draws on rungs 1+2 while `hun_Latn`
   draws on rung 1 alone, that asymmetry is a known confound and must be visible in the
   manifest rather than buried in a shell history.
3. **Run the FLORES `devtest` overlap check** on the pools. This is the one that matters most
   — Plan B's evaluation is FLORES, so pretraining contamination invalidates the result in a
   way tokenizer contamination does not.
4. **Upload,** manifest last:
   ```bash
   aws s3 sync artifacts/plan_b/pools/ "$CORPUS_S3_ROOT/pretrain/pools/" \
     --exclude manifest.json
   aws s3 cp artifacts/plan_b/pools/manifest.json \
     "$CORPUS_S3_ROOT/pretrain/pools/manifest.json"
   ```
5. **Verify the round trip** by sha256, as §3.2 step 7.

`mixture.json` is **not** written in this phase. It is written after the Plan A scale
tokenizer measures bytes/token, then uploaded to `pretrain/mixture.json`.

### 4.5 Gate

- [ ] Every language meets its §4.2 target; `hat_Latn` and `swh_Latn` are drawn in full.
- [ ] Rungs used are recorded per language in the manifest.
- [ ] Document boundaries are intact — spot-check that a shard round-trips through
      `json.loads` line by line.
- [ ] FLORES `devtest` overlap measured and recorded.
- [ ] Round-trip sha256 matches.

---

## 5. Cost

Rough, us-east-1, S3 Standard, as of 2026-08:

| Item | Size | Cost |
|---|---:|---|
| Transfer in | 66 GB | $0 |
| Storage, tokenizer corpus | 0.96 GB | ~$0.02 / month |
| Storage, pretraining pools | 65 GB | ~$1.50 / month |
| Same-region GET to Batch / EC2 | per read | $0 transfer; request charges are cents |
| EC2 for the Phase B pull | ~8 h `m7i.2xlarge` + 300 GB gp3 | ~$5 |

Storage is not the expense here. The `raw/` tree is, if it is left in place — hence the
lifecycle rule in §1.4. Upload `raw/` only if the fallback ladder fired and the provenance
needs to be reproducible from S3.

---

## 6. Failure modes worth naming

| Symptom | Cause | Response |
|---|---|---|
| A language reports `truncated: false` | Source ran dry below budget | Fallback ladder ([01 §3](01-data-sourcing.md)). If exhausted, **drop `bytes_per_language` for all six** — never let one language run short. |
| `meta.json` `train_files` lists fewer than six files | Corpus flattened to one `train.txt`, or `--num-bytes` below the manifest total | Prefix training. Rebuild one file per language; derive `--num-bytes` from the manifest. |
| Downstream job reads a partial prefix | `manifest.json` uploaded before the data | Always upload the manifest last. |
| A 65 GB sync failed and the bill did not drop | Orphaned multipart parts | The §1.4 lifecycle rule; `aws s3api list-multipart-uploads` to confirm. |
| FLORES overlap is non-trivial in the pools | Web-derived rungs include Wikinews / Wikivoyage source text | Report it; do not silently proceed. Rung 4 (Wikipedia) is the likeliest contributor. |

---

## 7. Checklist

**Phase A**

- [ ] Write `src/plan_a_langs.py`; rewrite the two scripts in §3.1.
- [ ] Re-verify availability for all six configs against the HF size API ([01 §9.1](01-data-sourcing.md)).
- [ ] Run the full phase at `smoke` first.
- [ ] Pull rung 1 at `bytes_per_language`; confirm no shortfall.
- [ ] Build the equal-byte corpus; assert `max(bytes) == min(bytes)`.
- [ ] FLORES `devtest` overlap check.
- [ ] Build CR-dev.
- [ ] Upload corpus + cr_dev, manifest last; verify sha256.
- [ ] Confirm `meta.json` `train_files` covers all six (§3.3).

**Phase B**

- [ ] Write `scripts/pull_plan_b_pools.py`.
- [ ] Provision the EC2 pull host in `$AWS_REGION`.
- [ ] Pull all six pools to the §4.2 targets, with headroom on the unbounded four.
- [ ] Dedup across rungs; record rungs per language.
- [ ] FLORES `devtest` overlap check on the pools.
- [ ] Upload, manifest last; verify sha256.

**After Plan A scale completes**

- [ ] Measure bytes/token per language on held-out text.
- [ ] `scripts/build_plan_b_mixture.py` → `mixture.json`; confirm it reproduces
      [03 §5.2](03-model-pretraining.md) with `hat_Latn` capped at exactly 4.00 epochs.
- [ ] Upload `mixture.json` to `pretrain/mixture.json`.
