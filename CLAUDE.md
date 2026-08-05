# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Three separate studies of multilingual tokenizer efficiency, sharing a codebase but **not**
sharing scope. Confusing them is the most common and most expensive mistake here.

| Study | Languages | Source of truth | State |
|---|---|---|---|
| **Efficiency validation** — encode-only FLORES metrics across 8 tokenizers | 12 | `src/load_flores.py:LANGUAGES` | Done, published |
| **Zipf deviation** — is the token distribution structurally sound per language | 18 | `src/zipf_langs.py` | Done, published |
| **Plan A / Plan B** — train a tokenizer, then a 1B model, on 6 languages | 6 | `configs/benchmarks/tokenizer_local.json` (and a planned `src/plan_a_langs.py`) | **Not implemented** |

Design documents live in [`plans/`](plans/README.md). Read `plans/README.md` before touching
anything in Plan A or Plan B.

## Two budgets, never interchangeable

- **Tokenizer training (Plan A): 960 MB of text** at the scale tier — 160 MB × 6 languages,
  exactly equal. Budgeted in *bytes*; there is no token target and inventing one is an error.
- **Model pretraining (Plan B): 20B tokens** ≈ 78 GB, measured on the **BPE arm**. SuperBPE
  trains on the same *bytes*, not the same tokens. Allocated by UniMax with a 4-epoch cap:
  `hat_Latn` 3.96%, the other five 19.21% each. 65.3 GB unique.

## Commands

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q                       # full suite
python -m pytest tests/test_zipf.py -q            # one file
python -m pytest tests/test_zipf.py::test_name -q # one test
```

FLORES-200 is not in git. Extract to `data/flores200_dataset/` or set `FLORES200_DIR`.
The Llama tokenizer needs `huggingface-cli login`.

**Three Python environments.** Do not merge them.

| Env | Python | Used by |
|---|---|---|
| default, from `requirements.txt` | repo default | the repo's own code, and orchestration of the two below |
| `.venv-benchmark/Scripts/python.exe`, built by `./scripts/setup_tokenizer_benchmark.ps1` | 3.11 | the **official** SuperBPE trainers: `run_official_tokenizer_benchmark.py`, `run_plan_a_tokenizer_pair.py`, the Plan B scripts |
| `../supergigatoken/.venv` | 3.13 + Rust toolchain | **gigatoken**, the default Plan A trainer |

`.venv-benchmark` pins `tokenizers` to a patched fork commit; gigatoken builds a native
extension with maturin. Neither can host the other. `run_gigatoken_tokenizer_benchmark.py`
runs in the default env and spawns the gigatoken env as a subprocess — which is also what
keeps the trainer inside the monitored process tree, so peak RSS is measured rather than
missed.

**Run the HF data pulls under `.venv-benchmark`, not the default env.** The default env is
Python 3.14, where `datasets` fails on import-time pickling with
`Pickler._batch_setitems() takes 2 positional arguments but 3 were given` (a `dill`
incompatibility). `scripts/pull_fineweb2_lang_samples.py` and anything else touching
`datasets` needs `.venv-benchmark/Scripts/python.exe`.

Published pipelines (both runnable today):

```bash
python -m src.run_eval --tokenizers o200k glm llama qwen multi --out-dir results
python scripts/run_vocab_profile.py --out-dir results/zipf     # Zipf stage 1
python scripts/run_zipf_eval.py --out-dir results/zipf         # Zipf stage 2
python scripts/export_web_data.py                              # regenerates web/data.js
```

Plan A tokenizer training (gigatoken is the default trainer):

```bash
python scripts/pull_fineweb2_lang_samples.py --langs hin_Deva hat_Latn --skip-existing
python scripts/build_plan_a_research_corpus.py --tier smoke
python scripts/run_gigatoken_tokenizer_benchmark.py --arm bpe \
    --gigatoken-repo ../supergigatoken \
    --corpus-file artifacts/plan_a/corpus/train.txt \
    --output-dir artifacts/plan_a/tokenizers/gt_bpe_4k \
    --result artifacts/plan_a/results/gt_bpe_4k.json \
    --log artifacts/plan_a/logs/gt_bpe_4k.log --vocab-size 4096
```

Add `--arm superbpe --transition-vocab-size 3072` for the second arm. Every tier must clear
the smoke-tier cross-check against the official trainer before it is trusted — see the
invariants below.

## Architecture

```
data/flores200_dataset/ ──► src/load_flores.py ──► src/metrics.py ──► src/run_eval.py
                                                                          │
                                            scripts/export_web_data.py ◄──┘
                                                      │
                                                      ▼
                                            web/ (static viewer)
```

`src/tokenizers_registry.py` loads all eight tokenizers behind one interface;
`src/grapheme_wrap.py` is the optional o200k grapheme-healed pretokenizer.

The Plan A/B pipeline is a separate spine that does not pass through `run_eval.py`:

```
pull ──► build_plan_a_research_corpus.py ──► run_gigatoken_tokenizer_benchmark.py ──► handoff/READY.json
          (equal bytes; emits both              (default trainer; official                    │
           corpus/{code}.txt and                 entrypoint retained for the                  ▼
           corpus/train.txt)                     cross-check)                  run_plan_b_materialize_shards.py
                                              │                                run_plan_b_preflight.py
                        verify_official_tokenizer_pair.py                      emit_plan_b_olmo_jobs.py
                        eval_plan_a_flores_compression.py
                        calibrate_arm_premiums.py
```

`scripts/run_official_tokenizer_benchmark.py` is the single entrypoint shared by local runs
and AWS Batch (`docker/tokenizer-benchmark/Dockerfile`). AWS changes only S3↔local staging;
command arguments and result schemas are identical.

### Invariants that are easy to break silently

- **The corpus must be exactly equal-byte, and that is now the load-bearing check.**
  `build_plan_a_research_corpus.py` writes both views of the same bytes: `corpus/{code}.txt`
  per language (official trainer) and `corpus/train.txt`, their byte-identical
  concatenation (gigatoken). `tests/test_plan_a_corpus_balance.py` asserts
  `max(bytes) == min(bytes)`; a language that cannot fill its share is a hard failure, never
  a silent shrink. That silent shrink is how the 6× skew got in
  ([02 §1.1](plans/02-tokenizer-training.md)).
- **The `corpus_dir` whole-file hazard applies to the official path only.** The official
  trainer's `get_files_with_num_bytes` (`.cache/superbpe/utils.py:147`) selects at whole-file
  granularity under `random.seed(0)`, so a single concatenated `train.txt` plus any
  `--num-bytes` below its size trains on the alphabetically-first languages only, and nothing
  errors. On that path: pass `--num-bytes` equal to the manifest's exact total and assert
  `meta.json:train_files` lists all six files. **On the gigatoken path this cannot happen** —
  `train_superbpe` takes one mmapped file and there is no file selection — so the manifest
  equality check above replaces the `train_files` assertion entirely.
- **Pass `separator=b"\n"` to gigatoken, not the default `<|endoftext|>`.** gigatoken strips
  separator bytes and yields one document per line, matching the line units HuggingFace's
  trainer reads. The default would leave `train.txt` as a single document and forfeit all
  pretokenization parallelism.
- **Stage 1 must be `superbpe_stage1`, never gigatoken's `gpt2` default.** The `gpt2`
  letter runs exclude `\p{M}`, so `हिन्दी` becomes 6 pretokens instead of 1 and the BPE arm
  can never learn a consonant+matra unit. This does not cancel between arms: premiums are a
  cross-*language* ratio, and because gigatoken's stage 2 applies no regex it repairs that
  fragmentation in the SuperBPE arm only — inflating that arm's apparent gain for exactly
  the scripts that use combining marks.
- **Stage 2 is a documented deviation from official SuperBPE, and it is not uniform across
  languages.** gigatoken bounds units by separator, newline, and `max_unit_len=128` with no
  regex, so superwords may bridge digits and punctuation the official `STAGE2_REGEX` splits.
  Its superword space is a strict superset; superwords never cross a newline.

  The reference's `[^\s\p{L}\p{N}]{2,}` alternative — written for punctuation runs — also
  matches Devanagari multi-mark clusters, so official stage 2 chops
  `हिन्दी भाषा में कई शब्द हैं` into four pieces while leaving English and Swahili whole.
  gigatoken does not inherit that, so it compresses Hindi *better*: `hin_Deva`'s SuperBPE
  premium runs **−3.13% at 4k vocab and −6.61% at 16k, growing with vocab size**. Re-read it
  at every tier; never assume it is stable. Every other language stays within 0.36%.

  **Consequence for any writeup:** the SuperBPE-arm premium table is gigatoken-SuperBPE's
  own and, for `hin_Deva`, is *not* comparable to published SuperBPE (Liu et al.) numbers.
  The BPE arm is comparable. Full analysis in
  [04-trainer-cross-check.md](plans/04-trainer-cross-check.md).
- **The cross-check gate is per-tier, and it must not pass vacuously.**
  `scripts/compare_plan_a_trainers.py` runs five configurations and hard-fails if any
  artifact is missing or any parity pair went uncompared — an earlier version reported PASS
  having measured nothing. Non-exempt tolerance is 1% (worst observed: 0.36%). The
  Devanagari deviation is a *named exemption carrying its mechanism*
  (`STAGE2_DEVIATION_EXPECTED`), not a loosened threshold, and keeps its own 15% tripwire.
  Artifact directory names carry the vocab size, so any non-smoke tier must pass `--configs`.
- **Encode each arm with the pre-tokenizer it was trained with.**
  `src/official_bpe_encode.py` reads it back from the artifact's own `tokenizer.json` rather
  than reconstructing it. Hardcoding `ByteLevel(add_prefix_space=False)` leaves
  `use_regex=True` (GPT-2 splitting), under which no superword can fire — measured at 12.2%
  more tokens on real corpus text, i.e. the SuperBPE arm's entire benefit.
- **SuperBPE must carry the exact BPE merge prefix.** `verify_official_tokenizer_pair.py`
  is the check, and it works against either trainer. On the official path the BPE arm must
  complete before the SuperBPE continuation starts; gigatoken continues in-process.
- **Both arms must report the same `input_bytes`.**
- **`eng_Latn` premium must be exactly 1.0** in any premium table.

## Do not touch

These are load-bearing and deliberately duplicated rather than unified. Editing one to match
another shifts already-published numbers.

| Thing | Why |
|---|---|
| `src/load_flores.py:LANGUAGES` and `CONTINENT` | The locked 12-language efficiency scope, consumed by `run_eval.py` and `export_web_data.py`. Plan A gets its own region map in its own module; it does not extend `CONTINENT`. |
| `src/zipf_langs.py` | The 18-language Zipf scope. Deliberately separate from `LANGUAGES` for the same reason. |
| `src/metrics.py:compute_strr` | Occurrence-weighted by design. The type-level sibling is `src/vocab_profile.py`; do not "fix" one into the other. |
| The Section 4 parity-aware BPE experiment | Published prior work: `artifacts/bpe_parity/`, `scripts/train_bpe_sweep.py`, `scripts/eval_bpe_sweep.py`, the fair-max implementation in `src/bpe_train.py`, `tests/test_fairmax_per_line.py`, and its web viewer tab. The *Plan A parity arm* was removed; this is not that. |
| `web/data.js` | Generated by `scripts/export_web_data.py`. Never hand-edit. |
| Pinned commits in `configs/benchmarks/tokenizer_local.json` | `superbpe_commit` and `tokenizers_commit` pin a patched fork. Bumping them invalidates every completed tier. The `official_pipeline` block is retained as the cross-check reference even though `gigatoken` is the default trainer — do not delete it. `gigatoken_pipeline.commit` is subject to the same rule once pinned. |
| `src/benchmark.py:sha256_file` | The project's one hashing helper. Do not add a second. |

## Plan B status

Mixture + pool acquisition scripts are written. Bytes still need an EC2 pull.

| File | Spec | Status |
|---|---|---|
| `src/plan_b_mixture.py` | [03 §5.1](plans/03-model-pretraining.md) | Done |
| `scripts/build_plan_b_mixture.py` | [03 §7.2](plans/03-model-pretraining.md) | Done |
| `scripts/pull_plan_b_pools.py` | [00 §4.3](plans/00-data-to-s3.md) | Done — run on EC2 (~300 GB disk) |
| `tests/test_plan_b_mixture.py` | [03 §9](plans/03-model-pretraining.md) | Done |
| `run_plan_b_preflight.py` | derives budget from `mixture.json` | Done (no 50 GB default) |
| `emit_plan_b_olmo_jobs.py` | reads `tie_word_embeddings` from config | Done |

**Still to execute (ops, not code):**

1. EC2 `pull_plan_b_pools.py` → `artifacts/plan_b/pools/` (~65–74 GB unique text).
2. Stage/publish into `edullm-data` via `edullm-data/scripts/stage_fineweb2_unimax_pools.py`.
3. Plan A scale gigatoken train → publish `tokenizer/gigatoken-{bpe,superbpe}`.
4. Measure bytes/token → `build_plan_b_mixture.py` → materialize `.u32le.bin` → publish token corpora.

**Blocking the smoke-tier gate:** `hin_Deva` and `hat_Latn` are not staged in
`artifacts/plan_a/raw/fineweb2_samples/` — they are the two languages the rescope added.
Pull them before building any Plan A corpus.

**All on-disk Plan A/B artifacts are from the superseded 14/16-language plan** —
`artifacts/tokenizer_benchmark/*_4k`, `artifacts/plan_a/research_cpu/`,
`artifacts/plan_b/materialize/`. They are not inputs to anything current.

**Open question:** weight tying. `plan_b_olmo.json` says `true`; OLMo-2-0425-1B ships
`false`. Unresolved, does not affect data volume. `emit_plan_b_olmo_jobs.py` reads the
config field rather than hardcoding.

## Working on different parts

The three studies are independent. When working in one, the others' files are off-limits
even when a change looks like a trivially shared improvement:

- **Efficiency validation** — `src/load_flores.py`, `src/metrics.py`, `src/run_eval.py`,
  `src/tokenizers_registry.py`, `scripts/export_web_data.py`, `web/`.
- **Zipf** — `src/zipf.py`, `src/zipf_langs.py`, `src/vocab_profile.py`,
  `scripts/run_zipf_eval.py`, `scripts/run_vocab_profile.py`, `scripts/plot_zipf.py`.
- **Plan A / Plan B** — `scripts/*plan_a*`, `scripts/*plan_b*`,
  `scripts/run_official_tokenizer_benchmark.py`,
  `scripts/run_gigatoken_tokenizer_benchmark.py`,
  `scripts/_gigatoken_train_worker.py`, `src/plan_a_langs.py`, `src/benchmark.py`,
  `src/official_bpe_encode.py`, `src/premium_calibration.py`, `configs/benchmarks/`,
  `docker/`. The trainer itself lives outside this repo, in `../supergigatoken`.

`src/bpe_train.py` and `src/bpe_encoder.py` belong to the published parity experiment, not
to Plan A — Plan A trains via subprocess, into gigatoken by default and into the official
SuperBPE repo for the cross-check.

## Conventions

- Language sets are frozen dataclasses with derived lookups and a module-level assertion —
  see `src/zipf_langs.py` and `src/plan_a_langs.py`. New sets follow that shape rather than
  a bare list, and each study owns exactly one; scripts import it rather than copying codes.
- Scripts write JSON through `src/benchmark.py:atomic_write_json`.
- Every benchmark result records peak process-tree RSS, runtime, CPU time, input hash, and
  trainer commits. Runtime is linearly projected to 10 GB and labelled as such; **memory is
  never extrapolated** — measure it at every tier and stop if a guard fires.
- `data/`, `results/`, `artifacts/plan_a/`, `artifacts/plan_b/`, and
  `artifacts/tokenizer_benchmark/` are gitignored.
