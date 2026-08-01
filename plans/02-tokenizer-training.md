# PRD — Rescope the BPE / SuperBPE comparison to 6 high-data languages

**Status:** Approved, not yet implemented.
**Scope:** Plan A two-arm tokenizer training (`bpe`, `superbpe`) and the corpus that
feeds it. The former third arm (`parity`) is removed — see §8. Does not touch the locked 12-language efficiency scope or the
18-language Zipf study.

---

## 1. Problem

The Plan A comparison currently spans 16 languages. Two properties of the
current setup make its results hard to attribute to the tokenizer rather than the data.

### 1.1 The training corpus is not balanced

`scripts/build_plan_a_research_corpus.py` greedily concatenates whatever sources it
finds in filesystem order. The completed pilot corpus
(`artifacts/plan_a/research_cpu/corpus/manifest.json`, 207.8 MB) came out as:

| Language | Bytes |
|----------|-------|
| `amh_Ethi` | 47,927,244 |
| `swh_Latn` | 47,868,488 |
| `quy_Latn` | 8,990,516 |
| 12 others | ~7.95–8.00 M each |
| `nah_Latn` | 6,720,804 |

The 6× skew is an artifact of which HF datasets and bible-corpus files happened to be
staged, not a design choice. Any cross-language premium difference is confounded with
the mix.

### 1.2 Several languages cannot supply meaningful volume

Measured from the FineWeb-2 size API (`num_bytes_original_files`):

- The **entire** indigenous-American web corpus is **82 MB across 33 languages** —
  Guarani 18.5 MB, Yucatec 7.1 MB, Aymara 5.2 MB, Quechua 3.8 MB, Nahuatl 2.0 MB.
- Nahuatl was already exhausted during the local pull at 5.3 MB against an 8 MB cap.

At the 1 GB scale tier a balanced 16-way split needs 62 MB/language. Those languages
cannot provide it at any price, so the corpus must either stay small or stay skewed.

### 1.3 Latent hazard: prefix training

`.cache/superbpe/utils.py:147 get_files_with_num_bytes` selects training data at
**whole-file granularity** over `*.txt` in `corpus_dir`, shuffled under
`random.seed(0)`, truncating only the final file. With today's single language-sorted
`train.txt`, any `--num-bytes` below the corpus size trains on a byte prefix — that is,
on the alphabetically-first languages only.

The completed pilot passed `--num-bytes 207814545`, exactly the corpus size, so it was
not hit. The README smoke example (`--num-bytes 2495955` against a 10 MB corpus) would
train almost entirely on one language.

---

## 2. Goal

Fewer languages, each with enough real data to fill an **equal share** of the corpus at
every tier, still spanning the world.

**Non-goals.** Changing the tokenizer algorithms, the arm definitions, the metric set,
the locked 12-language efficiency scope (`src/load_flores.py:LANGUAGES`), or the
18-language Zipf scope (`src/zipf_langs.py`).

---

## 3. Locked language set (6)

| Code | Language | Region | Script | FineWeb-2 config | Available |
|------|----------|--------|--------|------------------|-----------|
| `eng_Latn` | English (reference) | — | Latin | `HuggingFaceFW/fineweb` | ≫ |
| `hun_Latn` | Hungarian | Europe | Latin | `hun_Latn` | 98.6 GB |
| `zho_Hans` | Mandarin | Asia | Han | `cmn_Hani` | 1,622 GB |
| `hin_Deva` | Hindi | Asia | Devanagari | `hin_Deva` | 34.4 GB |
| `swh_Latn` | Swahili | Africa | Latin | `swh_Latn` | 1.44 GB |
| `hat_Latn` | Haitian Creole | Americas | Latin | `hat_Latn` | 0.30 GB ← **binding** |

Kept from the old 16: `eng`, `hun`, `zho`, `swh`. Added: `hin`, `hat`.
Dropped: `amh`, `hau`, `ukr`, `pol`, `tel`, `ory`, `tur`, `ayr`, `quy`, `grn`, `nah`, `yua`.

All six exist in the local FLORES-200 extraction (`dev` + `devtest`, verified), so
CR-dev becomes uniformly parallel and the AmericasNLP / reserved-pool fallback for
`nah`/`yua` is no longer reachable.

### 3.1 Rationale for the two judgement calls

**Africa → Swahili.** Highest-volume canonical sub-Saharan African language. Afrikaans
has more (3.68 GB) and Moroccan Arabic more still (2.89 GB), but Afrikaans is Germanic
Latin-script with near-zero tokenizer penalty. Swahili is Bantu, agglutinative, and
6× above any tier budget.

**Americas → Haitian Creole.** 16× Guarani, and with Papiamento (244 MB) the only
Americas language with real volume. It is a Latin-script, morphologically simple
French-lexified creole, so **expect a small premium from it**. Its value is covering
the Americas at a volume the tiers can actually use, not producing a large effect. This
is a deliberate trade against the alternative — keeping Guarani would cap a balanced
corpus at ~130 MB total and make the scale tier impossible.

---

## 4. Byte budget

Tiers become **per-language** budgets; total corpus bytes are derived. `hat_Latn` at
~300 MB sets the ceiling for a balanced 6-way corpus at ~1.8 GB.

| Tier | Bytes / language | Total corpus | Fits `hat_Latn`? |
|------|-----------------|--------------|------------------|
| smoke | 2,000,000 | 12 MB | yes |
| pilot | 16,000,000 | 96 MB | yes |
| scale | 160,000,000 | 960 MB | yes (53% of available) |

`projection_bytes: 10000000000` stays as a **runtime extrapolation constant only**. Add
`max_balanced_corpus_bytes` alongside it so the config records that 10 GB is not an
achievable balanced corpus for this set.

FineWeb-2 figures are `num_bytes_original_files` (parquet). The pull step must record
**actual extracted UTF-8 text bytes** per language and hard-fail on shortfall.

---

## 5. Implementation

### 5.1 New `src/plan_a_langs.py` — single source of truth

The 14-language list is currently copy-pasted across three scripts
(`build_plan_a_cr_dev.py`, `eval_plan_a_flores_compression.py`,
`stage_plan_a_selected_sources.py`) with a fourth variant in
`pull_fineweb2_lang_samples.py`. Replace with one module following the existing
`src/zipf_langs.py` pattern: frozen dataclass, derived lookups, module-level assertion.

Holds code, English name, region, script, FineWeb-2 dataset id + config,
`REFERENCE_LANG = "eng_Latn"`, and `PLAN_A_CODES` in deterministic order. Include an
`assert_reference_present()` guard mirroring `src/zipf_langs.py:assert_control_present`.

**Do not touch** `src/load_flores.py:LANGUAGES` / `CONTINENT` — that is the locked,
already-published 12-language efficiency scope consumed by `src/run_eval.py` and
`scripts/export_web_data.py`; the README calls this out explicitly. Plan A gets its own
region map in the new module rather than extending `CONTINENT`.

### 5.2 Rewrite `scripts/build_plan_a_research_corpus.py`

Replace the greedy `collect_sources()` walk with an equal-byte builder:

- Read one staged file per language from `artifacts/plan_a/raw/fineweb2_samples/{code}.txt`.
- Truncate each to exactly the per-language budget on a line boundary.
- Hard-fail — do not silently shrink — if any language is short.
- Write `corpus/{code}.txt`, one per language, for the official trainer.
- Manifest records per-language actual bytes + sha256, the exact total, and an
  assertion that `max(bytes) == min(bytes)` across languages.

Drop the bible-corpus / HF-dataset ingestion and the `--cr-dev-manifest` exclusion
plumbing. Those sources caused the 48 MB Amharic/Swahili skew, and with all six
languages in FLORES no CR pool is carved out of train.

**One file per language in `corpus_dir` is also the fix for §1.3.** With
`--num-bytes` derived from the manifest's exact total, every file fits, nothing is
truncated, and the shuffle in `get_files_with_num_bytes` cannot change the mix.

### 5.3 Simplify `scripts/build_plan_a_cr_dev.py`

Import the code list from `src/plan_a_langs.py`. All six are FLORES languages, so
hard-fail on any non-FLORES code and remove the now-unreachable `_find_americas_file`,
`_sentence_units*`, reserved-pool, and `train_exclude` paths. CR-dev is parallel FLORES
`dev` indices `0..N-1` for every language.

### 5.4 Point the remaining scripts at the shared module

- `scripts/pull_fineweb2_lang_samples.py` — build `LANG_SOURCES` from `plan_a_langs`;
  take the per-language budget as an argument rather than the 8 MB default; record
  actual text bytes and fail on shortfall.
- `scripts/eval_plan_a_flores_compression.py` — drop the local `PLAN_A_FLORES_LANGS` /
  `LANG_NAMES` copies and the `load_flores.CONTINENT` import; use `plan_a_langs`.
- `scripts/stage_plan_a_selected_sources.py` — `TARGET_LANGS` from `plan_a_langs`.
- `scripts/run_plan_a_tokenizer_pair.py` — read the corpus manifest and derive
  `--num-bytes` from its exact total instead of accepting a hand-passed value; after the
  BPE arm, assert `{output_dir}/meta.json` `train_files` covers all six languages and
  `total_bytes` equals the manifest total.

`scripts/run_official_tokenizer_benchmark.py` needs no language changes — it is
directory-driven.

### 5.5 `configs/benchmarks/tokenizer_local.json`

Add `languages` (the 6 codes) and `bytes_per_language` per tier; derive `corpus_bytes`.
Add `max_balanced_corpus_bytes` with a note that `hat_Latn` availability sets it. Keep
`projection_bytes`, labelled runtime-only.

### 5.6 New `tests/test_plan_a_corpus_balance.py`

Following the existing `tests/` style:

- every `plan_a_langs` code has a `corpus/{code}.txt`;
- per-language byte counts are equal;
- manifest total equals the sum of the on-disk files;
- `eng_Latn`, the premium reference, is present.

### 5.7 Docs

`README.md` and `plans/README.md`: update the language table, the tier budgets, and every Plan A
command example. The README currently hard-codes `--num-bytes 2495955`, which is exactly
the shape of the §1.3 hazard.

### 5.8 Artifacts

The rescope **replaces in place** — `artifacts/plan_a/research_cpu/` is rebuilt for the
new set. The existing 16-language pilot results are superseded. `artifacts/plan_a/` is
gitignored; the code that produced it remains in git history.

---

## 6. Verification

1. **Unit** — `python -m pytest tests/ -q`. Existing `test_premium_calibration.py`,
   `test_benchmark.py`, `test_official_bpe_encode.py`, `test_fairmax_per_line.py`,
   `test_vocab_profile.py` and `test_zipf.py` must still pass, plus the new balance
   test.
2. **Pull** — re-run the FineWeb-2 pull at the scale per-language budget and confirm the
   report shows six languages with `truncated: true`, i.e. the dataset had more than we
   took. Any `truncated: false` means that language ran dry and the budget must drop.
   This is the real check on the 300 MB `hat_Latn` figure.
3. **Balance** — build the smoke corpus; the manifest must report six languages at
   exactly 2,000,000 bytes each, total 12,000,000.
4. **No prefix training** — run the smoke pair, then inspect
   `artifacts/plan_a/research_cpu/tokenizers/smoke/bpe/meta.json`: `train_files` must
   list all six per-language files and `total_bytes` must equal the manifest total. This
   is the direct regression check on §1.3.
5. **Both arms agree on input** — `bpe.json` and `superbpe.json` must report the same
   `input_bytes`.
6. **Pair contract** — `scripts/verify_official_tokenizer_pair.py` must pass, as it
   does today: SuperBPE must carry the exact BPE merge prefix up to the transition
   vocabulary size.
7. **End-to-end** — `scripts/eval_plan_a_flores_compression.py` on FLORES `devtest`;
   `eng_Latn` premium must be exactly 1.0 and the other five finite and populated.
   Compare the BPE-vs-SuperBPE premium spread on the balanced corpus against the old
   skewed run.

---

## 7. Arms

Two: `bpe` (official SuperBPE-repo BPE) and `superbpe` (stage-two continuation off the
exact BPE merge prefix). The comparison is BPE vs SuperBPE on an identical, balanced
corpus.

Shared premium is the geometric mean over the arms present,
`(r_bpe * r_superbpe) ** (1/2)`. `src/premium_calibration.py:shared_premiums` is
arm-count agnostic, so this needed no formula rework — only the arm set changed.

---

## 8. Removed: the parity arm

The third arm (parity-aware BPE with fair-max merge selection) is **out of scope** and
has been deleted from the Plan A pipeline: `src/parity_official.py`,
`scripts/run_parity_tokenizer_benchmark.py`,
`scripts/verify_parity_tokenizer_contract.py` and `tests/test_parity_official.py` are
gone; Plan B arm tuples, `configs/benchmarks/*.json`, the Dockerfile comment and the
orchestrator no longer reference it.

`src/parity_official.py:load_lang_text_dir` was a generic per-language loader that
`scripts/compute_arm_premiums.py` still needs, so it moved to
`src/official_bpe_encode.py` with coverage in `tests/test_official_bpe_encode.py`.

Two renames follow from dropping to two arms:

| Was | Now |
|-----|-----|
| `scripts/run_plan_a_tokenizer_triplet.py` | `scripts/run_plan_a_tokenizer_pair.py` |
| `scripts/calibrate_three_arm_premiums.py` | `scripts/calibrate_arm_premiums.py` |

The emitted calibration `kind` changes from `three_arm_premium_calibration` to
`arm_premium_calibration`.

**Not removed:** the published Section 4 "Parity-aware BPE A/B" experiment
(commit `dfe9e7f`) is separate prior work and stays intact — `artifacts/bpe_parity/`,
its web viewer tab and export block in `scripts/export_web_data.py`,
`scripts/train_bpe_sweep.py`, `scripts/eval_bpe_sweep.py`, the fair-max implementation
in `src/bpe_train.py`, and `tests/test_fairmax_per_line.py`.

### 8.1 What this forgoes

The pilot parity arm produced token premiums of 4.04 (Amharic), 6.52 (Telugu) and 6.62
(Odia) — *worse* than plain BPE, which inverts the point of fair-max merge selection.
Whether that was a corpus-skew artifact or a bug in the fair-max implementation is now
**not going to be answered**, since the arm is gone and all three languages are dropped.
Recorded here so the anomaly is not mistaken for a resolved question. The fair-max code
itself survives in `src/bpe_train.py` if the arm is ever revived.
