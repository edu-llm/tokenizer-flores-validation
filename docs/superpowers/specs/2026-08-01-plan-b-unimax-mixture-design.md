# Design — Plan B pretraining mixture: UniMax at 20B tokens

**Date:** 2026-08-01
**Status:** Approved, not yet implemented.
**Scope:** The Plan B OLMo-1B pretraining corpus only. Plan A's tokenizer corpus,
the locked 12-language efficiency scope (`src/load_flores.py:LANGUAGES`), and the
18-language Zipf study (`src/zipf_langs.py`) are untouched.

---

## 1. Problem

`PRD.md` §4 established **equal bytes per language** for the six-language Plan A
corpus, and `DATA_PLAN.md` documents how to source it. That policy is correct for a
960 MB tokenizer corpus — `hat_Latn` has ~0.77 GB available, so a 160 MB equal share
is comfortably reachable.

It does not survive contact with model pretraining. A 1B-parameter model at a
Chinchilla-optimal 20:1 ratio needs **20B tokens**, roughly 78 GB of text. An equal
six-way split demands 13 GB per language. `hat_Latn` has 0.77 GB. Holding an equal
share would require **17 epochs** of Haitian Creole, well past where repetition stops
behaving like fresh data.

Three further gaps block execution:

1. **No corpus concept in Plan B config.** `configs/benchmarks/plan_b_olmo.json`
   declares architecture, arms, checkpoints, and evaluation, but has no notion of
   which languages appear in what proportion.
2. **Stale budget.** `scripts/run_plan_b_preflight.py:23` defaults
   `--target-train-bytes` to `50_000_000_000`, inherited from the "~50B tokens initial
   pass" in `artifacts/GDOC_AWS_BPE_SUPERBPE.md:30`. The target is now 20B tokens.
3. **No availability-aware allocator.** `src/premium_calibration.py:target_token_shares`
   normalizes premiums into shares with no availability constraint and no repeat cap.
   It will assign `hat_Latn` a share it physically cannot fill.

---

## 2. Goal

A pretraining mixture that is as uniform across the six languages as the available data
permits, at a fixed 20B-token budget, with repetition bounded by an explicit epoch cap.

**Non-goals.** Changing the tokenizer algorithms, the arm set, the metric set, the
vocabulary size, or Plan A's equal-bytes tokenizer corpus.

---

## 3. Decisions

| Decision | Value | Rationale |
|---|---|---|
| Applies to | Plan B pretraining corpus only | The tokenizer is the independent variable; balancing its training corpus keeps a measured Creole penalty from being a corpus artifact. |
| Allocator | UniMax (Chung et al. 2023) | Maximizes uniformity subject to a per-language epoch cap. Published, simple, deterministic. |
| Epoch cap | **N = 4** | Muennighoff et al., *Scaling Data-Constrained Language Models*: repetition to ~4 epochs is nearly as good as fresh data, degrading measurably after. |
| Uniform over | Tokens | Equalizes each language's FLOP contribution. UniMax as published operates on tokens. |
| Budget anchor | BPE reaches 20B tokens; SuperBPE trains on the **same bytes** | Both models see identical content, so a quality difference is attributable to the tokenizer. Already the repo's committed design: `plan_b_olmo.json` `checkpoints.equal_bytes: true`, `equal_flops_baseline: "bpe"`. |
| Vocabulary | 100,000, SuperBPE transition 80,000 | Already set in `configs/benchmarks/tokenizer_local.json` scale tier; matches OLMo-2-0425-1B's `vocab_size: 100352`. Identical in both arms, so embedding parameters are controlled. |
| Weight tying | **Open** | `plan_b_olmo.json:9` says `true`; OLMo-2-0425-1B ships `false`. Recorded as an open decision. Does not affect data volume. |

---

## 4. Two corpora, named separately

The central clarification. "The corpus" has meant two things; the design separates them.

| | Tokenizer corpus (Plan A) | Pretraining corpus (Plan B) |
|---|---|---|
| Size | 960 MB (scale tier) | ~78 GB |
| Policy | equal bytes per language | UniMax, N=4 |
| Config | `configs/benchmarks/tokenizer_local.json` | `configs/benchmarks/plan_b_olmo.json` |
| Docs | `DATA_PLAN.md` §1–§9 | `DATA_PLAN.md` §10 (new) |
| Status | unchanged | new |

`PRD.md` §4 gains one sentence scoping `equal_bytes_per_language` to the tokenizer
corpus, so the two policies stop reading as a contradiction.

---

## 5. Allocation

### 5.1 Algorithm

New `src/plan_b_mixture.py`. Pure functions, no I/O, following the shape of
`src/premium_calibration.py` — `Mapping` in, `dict` out, raises on invalid input.

```python
unimax_allocation(
    available_bytes: Mapping[str, float],
    budget_bytes: float,
    epoch_cap: float,
) -> dict[str, Allocation]      # Allocation = (bytes, epochs_of_available, capped: bool)
```

`epochs_of_available` is `bytes / available_bytes[lang]` — the quantity the cap bounds.
The separate "passes over pool" figure in §5.2 is a property of the sampling step, not
the allocator, and is derived downstream in `build_plan_b_mixture.py`.

Sort languages ascending by availability. Walk them: if a language's uniform share of
the remaining budget exceeds `epoch_cap × available`, allocate `epoch_cap × available`,
mark it capped, and redistribute the remainder uniformly across the languages not yet
assigned. Otherwise allocate the uniform share to it and every remaining language.

Raise if the budget exceeds `epoch_cap × sum(available)` — that is unsatisfiable, and
silently returning a short corpus would be the PRD §1.1 skew failure in a new costume.

### 5.2 Result

At the planning budget of 78 GB (20B tokens × 3.9 bytes/token) with N=4:

| Language | Share | Allocated | Available | Epochs of available | Unique pool | Passes over pool |
|---|---:|---:|---:|---:|---:|---:|
| `hat_Latn` | 3.96% | 3.09 GB | 0.77 GB | **4.00** (capped) | 0.77 GB | 4.00 |
| `swh_Latn` | 19.21% | 14.98 GB | 4.58 GB | 3.27 | 4.58 GB | 3.27 |
| `eng_Latn` | 19.21% | 14.98 GB | unbounded | ≈0 | 14.98 GB | 1.00 |
| `hin_Deva` | 19.21% | 14.98 GB | 34.4 GB | 0.44 | 14.98 GB | 1.00 |
| `hun_Latn` | 19.21% | 14.98 GB | 98.6 GB | 0.15 | 14.98 GB | 1.00 |
| `zho_Hans` | 19.21% | 14.98 GB | 1622 GB | 0.01 | 14.98 GB | 1.00 |
| **Total** | 100% | **78.0 GB** | — | — | **65.3 GB** | — |

The two epoch columns are different quantities and both matter. **Epochs of available**
is what UniMax constrains — allocation divided by everything that exists in that
language. **Passes over pool** is what training actually does, against the subset
acquired. They coincide only for the two languages drawn in full; for the other four we
acquire exactly the allocation, so each document is seen once.

Haitian Creole is the only language the cap binds on. Five of six come out exactly
uniform. **65.3 GB of unique text to acquire**, expanding to 78 GB of training data
through repetition on the two low-resource languages.

Availability figures come from `DATA_PLAN.md` §2–§3: `hat_Latn` 0.772 GB and
`swh_Latn` 4.577 GB summed across the ladder; `hin_Deva`, `hun_Latn`, `zho_Hans` and
`eng_Latn` are unbounded at this scale.

---

## 6. Ordering: acquire pools, then sample

The mixture depends on measured bytes/token, which depends on the trained 100k
tokenizer, which does not yet exist. So the pipeline cannot be "compute the mixture,
then download it". The order is:

1. **Acquire pools** — bounded by availability, tokenizer-independent.
2. **Train the Plan A scale tokenizer** (both arms).
3. **Measure bytes/token per language** on held-out text.
4. **Compute the mixture** from measured values.
5. **Sample from the pools** to materialize shards.

Pool sizes are stable under this uncertainty: five of six languages are
availability-unbounded, so a revised bytes/token figure moves how much is *drawn* from
each pool, not how much must *exist*. Only `hat_Latn` and `swh_Latn` pools are fixed by
availability, and both are drawn in full regardless.

The 78 GB figure is therefore explicitly provisional. `scripts/build_plan_b_mixture.py`
re-derives it and records both the assumed and measured values.

---

## 7. Components

### 7.1 New `src/plan_b_mixture.py`

As §5.1. Pure, testable, no filesystem access.

### 7.2 New `scripts/build_plan_b_mixture.py`

Reads measured per-language availability and measured bytes/token, calls
`unimax_allocation`, writes `mixture.json`:

```
kind: plan_b_mixture
target_tokens: 20000000000
bytes_per_token: {measured, per language, and the arm measured on}
budget_bytes: {derived}
epoch_cap: 4
allocation: {lang: {bytes, tokens, epochs_of_available, passes_over_pool, capped, share}}
unique_pool_bytes: {lang: bytes}
```

Both epoch figures are recorded because they answer different questions: `epochs_of_available`
is the UniMax constraint, `passes_over_pool` is what the sampler must actually emit.

Uses `src/benchmark.py:atomic_write_json`, as every other script in `scripts/` does.

### 7.3 `configs/benchmarks/plan_b_olmo.json` — new `corpus` block

```json
"corpus": {
  "policy": "unimax",
  "epoch_cap": 4,
  "target_tokens": 20000000000,
  "target_tokens_arm": "bpe",
  "languages": ["eng_Latn", "hun_Latn", "zho_Hans", "hin_Deva", "swh_Latn", "hat_Latn"],
  "uniform_over": "tokens",
  "note": "budget_bytes is derived from measured bytes/token; see mixture.json"
}
```

`target_tokens_arm: "bpe"` is the load-bearing field: it records that 20B is measured on
the BPE arm and SuperBPE matches on bytes, not tokens.

### 7.4 `DATA_PLAN.md` §10 — pretraining corpus

New section covering: the distinction from §1's tokenizer corpus, the UniMax table from
§5.2, per-language unique pool targets, the pool-then-sample ordering, and the S3 layout
extension (`pretrain/pools/<lang>/` and `pretrain/mixture.json`). §1 gains a pointer to
§10 so the equal-bytes policy is not mistaken for global.

### 7.5 `PRD.md` §4 — scoping sentence

One sentence: equal-bytes governs the tokenizer corpus; the pretraining mixture is
UniMax and lives in the Plan B config.

---

## 8. Fixes this exposes

Three existing inconsistencies that block the above and are in scope because the work
touches them directly.

| File | Problem | Fix |
|---|---|---|
| `scripts/run_plan_b_preflight.py:23` | `--target-train-bytes` defaults to 50 GB, from the superseded 50B-token plan | Derive from `mixture.json`; no default. A wrong default here silently trains the wrong budget. |
| `scripts/emit_plan_b_olmo_jobs.py:50` | Hardcodes `"tie_word_embeddings": True` although line 39 already passes `config["architecture"]`, which contains the field | Read from config. This hardcode is precisely what makes the deferred tying decision expensive. |
| `scripts/emit_plan_b_olmo_jobs.py:40` | Job name `olmo1b-{arm}-100k-tied` bakes tying into the identifier | Derive the suffix from the config value. |

`configs/benchmarks/plan_b_olmo.json:9` keeps `tie_word_embeddings: true` but gains a
note recording that OLMo-2-0425-1B ships `false` and that this is unresolved — a live
question rather than a silent divergence.

---

## 9. Testing

New `tests/test_plan_b_mixture.py`, following the existing `tests/` style:

- allocation sums to the budget within floating-point tolerance;
- no language exceeds `epoch_cap`;
- an availability-unconstrained set comes out exactly uniform;
- the six-language production case puts `hat_Latn` at exactly 4.00 epochs and `capped: true`,
  and every other language at an identical share;
- a budget above `epoch_cap × sum(available)` raises rather than returning short;
- a single-language set degenerates correctly;
- ordering of the input mapping does not change the result.

Existing tests must still pass unchanged: `test_premium_calibration.py`,
`test_benchmark.py`, `test_official_bpe_encode.py`, `test_fairmax_per_line.py`,
`test_vocab_profile.py`, `test_zipf.py`.

---

## 10. Verification

1. **Unit** — `python -m pytest tests/ -q`; 149 existing tests plus the new mixture tests.
2. **Allocation matches this spec** — run `build_plan_b_mixture.py` with the §5.2
   availability figures and the 3.9 bytes/token planning constant; the emitted
   `mixture.json` must reproduce the §5.2 table.
3. **Config consistency** — `plan_b_olmo.json` `corpus.languages` must equal
   `tokenizer_local.json` `languages`.
4. **No stale budget** — grep for `50_000_000_000`; it must not survive as a default.
5. **Tying is config-driven** — change `tie_word_embeddings` in `plan_b_olmo.json` and
   confirm both the emitted job spec and the job name follow.
6. **Docs agree** — the §5.2 table in `DATA_PLAN.md` §10 must match `mixture.json`, and
   `PRD.md` §4 must no longer read as governing both corpora.

---

## 11. What this forgoes

`zho_Hans` receives a full 19.21% share drawn at 0.01 epochs from a 1.6 TB pool. Uniform
treatment is the point of UniMax, but it does mean the most data-rich language in the set
contributes the same compute as the scarcest — deliberate, and worth revisiting if the
Mandarin result looks undertrained.

The N=4 cap is a literature-backed default, not a tuned value for this language set. No
ablation over N is planned; if Creole results are ambiguous, N is the first knob to
revisit, and re-running only requires re-sampling from existing pools, not re-acquisition.
