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

**Two Python environments.** The repo's own code runs in the default env from
`requirements.txt`. The official SuperBPE trainers need an isolated Python 3.11 built by
`./scripts/setup_tokenizer_benchmark.ps1`, and are invoked as
`.venv-benchmark/Scripts/python.exe`. Anything calling `run_official_tokenizer_benchmark.py`,
`run_plan_a_tokenizer_pair.py`, or the Plan B scripts uses `.venv-benchmark`. Do not merge
the two — the trainers pin `tokenizers` to a patched fork commit.

Published pipelines (both runnable today):

```bash
python -m src.run_eval --tokenizers o200k glm llama qwen multi --out-dir results
python scripts/run_vocab_profile.py --out-dir results/zipf     # Zipf stage 1
python scripts/run_zipf_eval.py --out-dir results/zipf         # Zipf stage 2
python scripts/export_web_data.py                              # regenerates web/data.js
```

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
pull ──► build equal-byte corpus ──► run_plan_a_tokenizer_pair.py ──► handoff/READY.json
                                              │                              │
                        verify_official_tokenizer_pair.py                    ▼
                        calibrate_arm_premiums.py            run_plan_b_materialize_shards.py
                                                             run_plan_b_preflight.py
                                                             emit_plan_b_olmo_jobs.py
```

`scripts/run_official_tokenizer_benchmark.py` is the single entrypoint shared by local runs
and AWS Batch (`docker/tokenizer-benchmark/Dockerfile`). AWS changes only S3↔local staging;
command arguments and result schemas are identical.

### Invariants that are easy to break silently

- **One `.txt` per language in `corpus_dir`.** The official trainer's
  `get_files_with_num_bytes` (`.cache/superbpe/utils.py:147`) selects at whole-file
  granularity under `random.seed(0)`. A single concatenated `train.txt` plus any
  `--num-bytes` below its size trains on the alphabetically-first languages only. Nothing
  errors. Always pass `--num-bytes` equal to the manifest's exact total, and assert
  `meta.json:train_files` lists all six files.
- **SuperBPE must carry the exact BPE merge prefix.** `verify_official_tokenizer_pair.py`
  is the check; the BPE arm must complete before the SuperBPE continuation starts.
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
| Pinned commits in `configs/benchmarks/tokenizer_local.json` | `superbpe_commit` and `tokenizers_commit` pin a patched fork. Bumping them invalidates every completed tier. |
| `src/benchmark.py:sha256_file` | The project's one hashing helper. Do not add a second. |

## Not implemented

The 6-language Plan A/B work is designed but unwritten. Scripts still carry the superseded
14/16-language lists.

**Missing modules**

| File | Spec |
|---|---|
| `src/plan_a_langs.py` | [02 §5.1](plans/02-tokenizer-training.md) — single source of truth for the 6 codes; mirror the `src/zipf_langs.py` pattern |
| `src/plan_b_mixture.py` | [03 §5.1](plans/03-model-pretraining.md) — `unimax_allocation`, pure functions, no I/O |
| `scripts/build_plan_b_mixture.py` | [03 §7.2](plans/03-model-pretraining.md) — writes `mixture.json` |
| `scripts/pull_plan_b_pools.py` | [00 §4.3](plans/00-data-to-s3.md) — sharded 65 GB pull preserving document boundaries |
| `tests/test_plan_a_corpus_balance.py` | [02 §5.6](plans/02-tokenizer-training.md) |
| `tests/test_plan_b_mixture.py` | [03 §9](plans/03-model-pretraining.md) |

**Scripts carrying stale language lists** — all must read from `src/plan_a_langs.py`:
`build_plan_a_cr_dev.py`, `eval_plan_a_flores_compression.py`,
`stage_plan_a_selected_sources.py`, `pull_fineweb2_lang_samples.py`.

**Known-wrong values**

- `scripts/build_plan_a_research_corpus.py` — greedy filesystem-order concatenation; produced
  the 6× skew documented in [02 §1.1](plans/02-tokenizer-training.md). Needs the equal-byte
  rewrite in §5.2.
- `scripts/run_plan_b_preflight.py:23` — `--target-train-bytes` defaults to `50_000_000_000`
  from the superseded 50B-token plan. Must derive from `mixture.json`, with no default.
- `scripts/emit_plan_b_olmo_jobs.py:50` — hardcodes `tie_word_embeddings: True` although
  line 39 already passes `config["architecture"]`, which carries the field. Line 40 also
  bakes `-tied` into the job name.

**Open question:** weight tying. `plan_b_olmo.json:9` says `true`; OLMo-2-0425-1B ships
`false`. Unresolved, does not affect data volume.

## Working on different parts

The three studies are independent. When working in one, the others' files are off-limits
even when a change looks like a trivially shared improvement:

- **Efficiency validation** — `src/load_flores.py`, `src/metrics.py`, `src/run_eval.py`,
  `src/tokenizers_registry.py`, `scripts/export_web_data.py`, `web/`.
- **Zipf** — `src/zipf.py`, `src/zipf_langs.py`, `src/vocab_profile.py`,
  `scripts/run_zipf_eval.py`, `scripts/run_vocab_profile.py`, `scripts/plot_zipf.py`.
- **Plan A / Plan B** — `scripts/*plan_a*`, `scripts/*plan_b*`,
  `scripts/run_official_tokenizer_benchmark.py`, `src/benchmark.py`,
  `src/official_bpe_encode.py`, `src/premium_calibration.py`, `configs/benchmarks/`,
  `docker/`.

`src/bpe_train.py` and `src/bpe_encoder.py` belong to the published parity experiment, not
to Plan A — Plan A uses the official SuperBPE-repo trainers via subprocess.

## Conventions

- Language sets are frozen dataclasses with derived lookups and a module-level assertion —
  see `src/zipf_langs.py`. New sets follow that shape rather than a bare list.
- Scripts write JSON through `src/benchmark.py:atomic_write_json`.
- Every benchmark result records peak process-tree RSS, runtime, CPU time, input hash, and
  trainer commits. Runtime is linearly projected to 10 GB and labelled as such; **memory is
  never extrapolated** — measure it at every tier and stop if a guard fires.
- `data/`, `results/`, `artifacts/plan_a/`, `artifacts/plan_b/`, and
  `artifacts/tokenizer_benchmark/` are gitignored.
