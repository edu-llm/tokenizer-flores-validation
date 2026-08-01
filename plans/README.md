# Plans

Every design document for this repo. Read in order — each depends on the one before.

| # | Plan | Covers | Status |
|---|------|--------|--------|
| **0** | [00-data-to-s3.md](00-data-to-s3.md) | Acquire both corpora and stage them in S3 | Not implemented |
| **1** | [01-data-sourcing.md](01-data-sourcing.md) | Which dataset each language comes from; fallback ladder; license gate | Sourcing decided; not executed |
| **2** | [02-tokenizer-training.md](02-tokenizer-training.md) | **Plan A** — BPE vs SuperBPE on 6 languages | Approved, not implemented |
| **3** | [03-model-pretraining.md](03-model-pretraining.md) | **Plan B** — OLMo-1B at 20B tokens, UniMax mixture | Approved, not implemented |
| — | [completed/efficiency-validation.md](completed/efficiency-validation.md) | The 12-language FLORES efficiency study that motivated the above | **Done and published** |

## The two budgets, in one place

The single most confused pair of numbers in this project:

| | Tokenizer training (Plan A) | Model pretraining (Plan B) |
|---|---|---|
| **Budget** | **960 MB of text** (scale tier) | **20B tokens** ≈ 78 GB of text |
| Unit | bytes — a tokenizer trainer consumes raw bytes, so no token target exists or should be invented | tokens, measured on the **BPE arm**; SuperBPE trains on the same *bytes* |
| Per language | 160 MB × 6, **exactly equal** | UniMax with a 4-epoch cap: `hat_Latn` 3.96%, other five 19.21% each |
| Unique text needed | 960 MB | **65.3 GB** — repetition on the two low-resource languages expands it to 78 GB |
| Why that policy | The tokenizer is the independent variable; an unbalanced corpus would make a measured Creole penalty a corpus artifact | Equal bytes is unreachable — an equal six-way split of 78 GB needs 13 GB of Haitian Creole, and only 0.77 GB exists |
| Authority | `configs/benchmarks/tokenizer_local.json` | `configs/benchmarks/plan_b_olmo.json` + `pretrain/mixture.json` |

Smaller tokenizer tiers: smoke 12 MB (2 MB × 6), pilot 96 MB (16 MB × 6).

## Language set

Six, locked in [02 §3](02-tokenizer-training.md): `eng_Latn` (reference), `hun_Latn`,
`zho_Hans`, `hin_Deva`, `swh_Latn`, `hat_Latn`. `hat_Latn` is the binding constraint at
0.77 GB available and is the only language the UniMax epoch cap acts on.

This is **not** the same set as the completed efficiency study, which locked 12 languages in
`src/load_flores.py:LANGUAGES`, or the Zipf study's 18 in `src/zipf_langs.py`. All three are
separate scopes and none of them may be edited to match another.

## Execution order

```
Plan 0 Phase A  ──►  Plan A smoke ──► pilot ──► scale  ──►  measure bytes/token
    │                                                              │
    │                                                              ▼
    └──► Plan 0 Phase B (parallel) ──────────────────────►  mixture.json ──► Plan B
```

Phase B acquisition does not wait on Plan A: pool sizes are availability-bounded for the two
languages that bind and generous for the four that do not. Only the *allocation* waits.
