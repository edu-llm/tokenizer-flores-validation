# Tokenizer FLORES Validation

Encode-only multilingual tokenizer efficiency on FLORES-200 **devtest** across 12 languages and 8 tokenizers.

## What this measures

Each tokenizer encodes the full FLORES-200 devtest split for a locked set of 12 languages (one per continent, plus English as reference). No decoding or model inference — only tokenization and metric aggregation.

| Metric | Description |
|--------|-------------|
| **CTC** (Corpus Token Count) | Total number of tokens produced when encoding the whole language corpus (BOS/EOS excluded). The base count the other metrics derive from. |
| **Fertility** | Tokens per whitespace-delimited word (`CTC / word_count`) after NFKC |
| **Chars/token** | Non-whitespace Unicode code points per token (`chars / CTC`) |
| **Token premium** | `CTC_lang / CTC_eng` — relative token cost vs English |
| **STRR** (Single Token Retention Rate) | Share of whitespace-delimited words that encode to exactly one token |
| **STFR** (Single Token Fragmentation Rate) | Share of emitted tokens whose decoded surface form is a single character |

STRR and STFR use per-token surface decode (each token decoded in isolation) to detect fragmentation without full-sequence decoding.

## Architecture

```
data/flores200_dataset/          (local; not in git)
        │
        ▼
src/load_flores.py               load dev + devtest sentences for locked languages
        │
        ▼
src/metrics.py                   CTC, fertility, chars/token, STRR/STFR, token premium
        │
        ▼
src/run_eval.py                  CLI — writes results/ tables + metrics.json
        │
        ▼
scripts/export_web_data.py       embeds results into web/data.js
        │
        ▼
web/                             static viewer (index.html, app.js, styles.css)
```

Supporting modules: `src/tokenizers_registry.py` (tokenizer loading), `src/grapheme_wrap.py` (optional o200k grapheme wrapper).

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Download FLORES-200 data

The dataset is not checked into git. Download the official tarball and extract into `data/flores200_dataset/`:

```bash
mkdir -p data
curl -L -o data/flores200_dataset.tar.gz \
  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar -xzf data/flores200_dataset.tar.gz -C data/
```

Alternatively, point to an existing extraction with the `FLORES200_DIR` environment variable.

### 3. Hugging Face auth (Llama tokenizer)

The Llama tokenizer requires Hugging Face authentication and model access:

```bash
huggingface-cli login
```

### 4. Run evaluation

```bash
python -m src.run_eval --tokenizers o200k glm llama qwen multi --out-dir results
```

### 5. Export viewer data and serve

```bash
python scripts/export_web_data.py
cd web
python -m http.server 8765
```

Open [http://localhost:8765](http://localhost:8765) for the metrics viewer (production heatmap + grapheme / parity experiment tabs).

### 6. Parity-aware BPE A/B (optional experiment)

From-scratch o200k-pretok BPE on an English-skewed FLORES mix, classical byte vs Parity-aware fair-max merge selection ([Foroutan/Meister et al.](https://arxiv.org/abs/2508.04796)):

```bash
python scripts/train_bpe_sweep.py --skew english_aggressive --units byte parity --out-dir artifacts/bpe_parity
python scripts/eval_bpe_sweep.py --artifact-dir artifacts/bpe_parity --baseline-unit byte --compare-unit parity --out-dir artifacts/bpe_parity
python scripts/export_web_data.py
```

This is a **training-time** sibling tokenizer (new merge list), not a frozen-o200k wrap. Eval adds Gini of tokens-per-line alongside fertility / premium / STFR / STRR.

## Local BPE/SuperBPE resource benchmark

This benchmark is infrastructure validation for Plan A, not a scientific
tokenizer result. The smoke corpus may use FLORES `dev` to exercise the
pipeline, but final tokenizers must train only on the decontaminated pretraining
mixture and must never consume FLORES or AmericasNLP evaluation data.

On Windows, create an isolated Python 3.11 environment and build the two pinned
official repositories:

```powershell
./scripts/setup_tokenizer_benchmark.ps1
```

Build a deterministic, byte-bounded corpus from explicitly selected language
files:

```powershell
python scripts/build_tokenizer_benchmark_corpus.py `
  --input data/flores200_dataset/dev/eng_Latn.dev <other-language-files> `
  --output artifacts/tokenizer_benchmark/corpus/train.txt `
  --manifest artifacts/tokenizer_benchmark/corpus/manifest.json `
  --target-bytes 10000000
```

Then run the pinned official pipeline. The BPE arm must complete before the
SuperBPE continuation:

```powershell
.venv-benchmark/Scripts/python.exe scripts/run_official_tokenizer_benchmark.py `
  --arm bpe --superbpe-repo .cache/superbpe `
  --corpus-dir artifacts/tokenizer_benchmark/corpus `
  --output-dir artifacts/tokenizer_benchmark/bpe_4k `
  --result artifacts/tokenizer_benchmark/results/bpe_4k.json `
  --log artifacts/tokenizer_benchmark/logs/bpe_4k.log `
  --num-bytes 2495955 --vocab-size 4096 `
  --patched-tokenizers-commit 757f2a55c0820ed47064e1fe473deea39b7b611b `
  --max-rss-gb 8 --force

.venv-benchmark/Scripts/python.exe scripts/run_official_tokenizer_benchmark.py `
  --arm superbpe --superbpe-repo .cache/superbpe `
  --corpus-dir artifacts/tokenizer_benchmark/corpus `
  --baseline-dir artifacts/tokenizer_benchmark/bpe_4k `
  --output-dir artifacts/tokenizer_benchmark/superbpe_4k_t3072 `
  --result artifacts/tokenizer_benchmark/results/superbpe_4k_t3072.json `
  --log artifacts/tokenizer_benchmark/logs/superbpe_4k_t3072.log `
  --num-bytes 2495955 --vocab-size 4096 --transition-vocab-size 3072 `
  --patched-tokenizers-commit 757f2a55c0820ed47064e1fe473deea39b7b611b `
  --max-rss-gb 8 --force
```

Verify that both vocabularies have the requested size and the exact BPE prefix
was supplied to stage two:

```powershell
python scripts/verify_official_tokenizer_pair.py `
  --baseline-dir artifacts/tokenizer_benchmark/bpe_4k `
  --superbpe-dir artifacts/tokenizer_benchmark/superbpe_4k_t3072 `
  --transition-vocab-size 3072 --expected-vocab-size 4096 `
  --result artifacts/tokenizer_benchmark/results/pair_verification.json
```

`configs/benchmarks/tokenizer_local.json` defines smoke, pilot, and scale gates.
Each result records process-tree peak RSS, runtime, CPU time, input hash,
trainer commits, logs, and a clearly marked linear 10 GB runtime projection.
Memory is not extrapolated; measure every tier and stop if a guard fires.

The same runner is the AWS Batch entrypoint in
`docker/tokenizer-benchmark/Dockerfile`. AWS only changes corpus/artifact
staging (S3 to local job storage and back); command arguments and result schemas
remain identical.

### Two-arm Plan A / Plan B (BPE + SuperBPE)

Plan A trains two arms on one balanced corpus: official BPE, and SuperBPE continued
off the exact BPE merge prefix. Scope and byte budgets are in **[PRD.md](PRD.md)**;
where the corpus text comes from and how it reaches S3 is in
**[DATA_PLAN.md](DATA_PLAN.md)**; tier parameters are in
`configs/benchmarks/tokenizer_local.json`.

Build the equal-content CR-dev used for premium calibration:

```powershell
.venv-benchmark/Scripts/python.exe scripts/build_plan_a_cr_dev.py `
  --output-dir artifacts/plan_a/research_cpu/cr_dev
```

Orchestrate both arms, verification, premium calibration, and `READY.json`:

```powershell
.venv-benchmark/Scripts/python.exe scripts/run_plan_a_tokenizer_pair.py `
  --work-dir artifacts/plan_a `
  --dev-lang-dir artifacts/plan_a/research_cpu/cr_dev `
  --corpus-dir artifacts/plan_a/research_cpu/corpus `
  --corpus-manifest artifacts/plan_a/research_cpu/corpus/manifest.json `
  --superbpe-repo .cache/superbpe `
  --num-bytes 12000000 --vocab-size 4096 --transition-vocab-size 3072 `
  --stage smoke --force
```

`--num-bytes` must equal the corpus manifest's exact total. `corpus_dir` holds one
`.txt` per language and the official trainer selects **whole files**, so a smaller
`--num-bytes` would silently train on a subset of the languages rather than a
proportional sample of all of them.

Plan B CPU materialization (after `handoff/READY.json`):

```powershell
.venv-benchmark/Scripts/python.exe scripts/run_plan_b_materialize_shards.py `
  --ready artifacts/plan_a/handoff/READY.json `
  --documents artifacts/tokenizer_benchmark/corpus/train.txt `
  --output-dir artifacts/plan_b/materialize `
  --max-documents 200

.venv-benchmark/Scripts/python.exe scripts/run_plan_b_preflight.py `
  --ready artifacts/plan_a/handoff/READY.json `
  --materialization artifacts/plan_b/materialize/materialization.json `
  --result artifacts/plan_b/preflight_schedule.json

.venv-benchmark/Scripts/python.exe scripts/emit_plan_b_olmo_jobs.py `
  --preflight artifacts/plan_b/preflight_schedule.json `
  --materialization artifacts/plan_b/materialize/materialization.json `
  --result artifacts/plan_b/olmo_job_bundle.json
```

## Language-specific Zipf deviation

A separate two-stage experiment asking a different question from the efficiency
metrics above. Those say how *expensive* a language is; this asks whether the token
distribution a tokenizer produces is *structurally sound* for that language.

Word frequencies are Zipfian in every language. A tokenizer that serves a language
well should preserve that; one that shreds it into byte fragments cannot. Each
language is evaluated against **its own active subset of the vocabulary**, not the
full 200k.

### Scope

18 languages — the locked 12 plus `spa_Latn`, `hin_Deva`, `kor_Hang`, `tha_Thai`,
`tir_Ethi`, `sat_Olck` — spanning 9 scripts and a 48× range in active-vocabulary
share. Four tokenizers: `o200k` (primary), `llama`, `qwen`, and `multi` (NLLB-200,
the multilingual-by-design control). Corpus is FLORES `dev`+`devtest` (2,009
parallel sentences per language); parallel content is the cross-language control.

### Stage 1 — vocabulary allocation

```bash
python scripts/run_vocab_profile.py --out-dir results/zipf
```

Classifies all 199,998 mergeable ranks by Unicode script, attributes the 1,562
partial-UTF-8 byte fragments to Unicode blocks via their leading byte, then
profiles every one of the 204 FLORES languages empirically.

| Output | Contents |
|--------|----------|
| `vocab_allocation_by_script.csv` | share of vocab per script |
| `vocab_allocation_fragments.csv` | byte fragments by best-guess block, with a certainty flag |
| `vocab_allocation_mixed.csv` | mixed-script token combinations |
| `lang_vocab_profile.csv` | per language: active types, `share_of_vocab`, exclusivity vs English (type- **and** mass-weighted), `share_fragment_mass`, type-level whole-word coverage |

Two metrics deserve attention. **`share_fragment_mass`** is the fraction of a
language's tokens that are raw partial-UTF-8 fragments — the most legible single
number in the study. **Mass-weighted exclusivity** exists because the type-level
version misleads: Amharic shares 538 of its 676 types with English, so type-level
exclusivity reads 0.20, but those shared types are punctuation and Latin names
appearing a few times each while the 76 exclusive Ethiopic fragments carry 88.8%
of all Amharic tokens.

### Stage 2 — Zipf deviation

```bash
# smoke first
python scripts/run_zipf_eval.py --tokenizers o200k --max-sentences 50 --bootstrap 5 \
  --out-dir results/zipf/_smoke
# full run
python scripts/run_zipf_eval.py --out-dir results/zipf
python scripts/plot_zipf.py --results-dir results/zipf
```

Fits Zipf–Mandelbrot `p(r) ∝ (r+b)^−α` by MLE over each language's observed
support. OLS on the log-log curve is reported for interpretability only — it is
biased ([Clauset et al. 2009](https://arxiv.org/abs/0706.1062)); measured here,
MLE error stays under 0.01 while OLS error runs 0.03–0.07.

**Three units**, because the baselines are what attribute a deviation to the
tokenizer rather than the language: `token`, `word` (whitespace, unavailable for
`zho_Hans` and `tha_Thai`), and `grapheme` (UAX #29, defined for every script and
therefore the universal reference).

**Two views**, both reported since they answer different questions —
`matched_token` (every language subsampled to an identical unit budget: matched
power, unmatched content) and `matched_sentence` (the whole parallel corpus:
matched content, unmatched size). Budgets are 95% of the smallest corpus for that
unit type, so every language carries sampling variability.

Two distinct goodness-of-fit quantities are kept separate:

| Metric | Question |
|--------|----------|
| `ks` | distance from the *best-fit* Zipf–Mandelbrot — is this power-law shaped at all? |
| `ks_zipf` | distance from *Zipf's law itself* (α=1, b=0) — how far from the law? |

A uniform distribution shows why: it *is* Zipf–Mandelbrot with α=0, so its `ks` is
near zero while its `ks_zipf` is large.

Because α and the KS statistics degrade on a few-hundred-type support — exactly the
regime the byte-level languages occupy — the entropy family is reported alongside
every fit: Shannon entropy, normalized entropy, Rényi efficiency at order 2.5
([Zouhar et al.](https://arxiv.org/abs/2306.16842)), and **effective vocabulary
size `exp(H)`**, the number of equally-likely units the distribution behaves like.
Support size `n_types` appears beside every fit; truncation is the dominant effect
and must not be normalized out of view.

| Output | Contents |
|--------|----------|
| `zipf_fits.csv` / `.json` | every (language, tokenizer, unit, view) fit with bootstrap intervals, `converged_share` and `at_bound_share` |
| `zipf_deltas.csv` | token − word and token − grapheme deltas |
| `hypotheses.json` | H1–H4 scored against the numbers, with Spearman statistics |
| `zipf_word_coverage.png` | whole-word coverage and byte-fragment mass per language |
| `zipf_allocation_vs_deviation.png` | the headline scatter |
| `zipf_rank_frequency.png` | log-log rank-frequency small multiples with fitted ZM and pure-Zipf curves |
| `zipf_script_allocation.png` | vocabulary split across Unicode scripts |
| `zipf_token_vs_baseline.png` | how far the tokenizer moves the distribution off the text's own |
| `zipf_deviation_heatmap.png` | log effective vocabulary, language × tokenizer |

### Headline results

Measured on `o200k_base`, matched-token view (49,788 tokens), 200 bootstrap draws.

**Vocabulary allocation is wildly uneven.** `o200k_base` contains no token holding
a complete Ethiopic character and only 38 for Oriya. Whole-word coverage — the
share of a language's distinct word types that survive as a single token — runs
English 52.9%, Spanish 29.9%, Hindi 14.2%, Hungarian 5.3%, down to Amharic 0.70%,
Odia 0.63%, Tigrinya 0.31%, Santali 0.30%. Byte-fragment mass runs the other way:
Santali 93.3%, Amharic 88.8%, Tigrinya 87.8%, Odia 27.0%, English 0.0%.

**The tokenizer can be worse than doing nothing.** Comparing each language's token
distribution against the raw Unicode grapheme clusters of the same text,
tokenizing multiplies English's effective vocabulary by 56× (23 → 1,317) and
Spanish's by 51×. For Odia, Tigrinya, Amharic and Santali the ratio falls *below
1* (0.23–0.36): the tokens carry fewer effective units than reading the text
character by character.

**Attribution holds.** Token-minus-word Δα — same text, same sample size, so
small-corpus bias cancels — averages 2.809 for the byte-level languages against
0.051 for English and Spanish, a 55× gap.

Pre-registered hypotheses, as scored in `hypotheses.json`:

| | Result | Evidence |
|---|---|---|
| **H1** allocation vs exclusivity | supported | ρ = −0.686, p = 0.0017. High raw share goes with *low* exclusivity — Quechua has 3.04% of the vocab but 48% exclusive mass; Amharic 0.34% but 89%. |
| **H2** allocation vs deviation | supported | ρ = −0.711 (share vs `ks_zipf`), +0.653 (share vs log eff. vocab), +0.706 (fragment mass vs `ks_zipf`), all p < 0.004 |
| **H3** delta vs allocation | partly supported | Effective vocabulary: ρ = +0.712 (word), +0.802 (grapheme). `ks_zipf`: ρ = −0.812 (grapheme) but **−0.109, p = 0.69 (word)** — reported rather than dropped |
| **H4** NLLB spread | **refuted** | Qwen has the smallest cross-language spread (σ = 0.0495), not NLLB (0.0549) |

### Reading the results honestly

- 2,009 parallel sentences is thin for power-law estimation. Bootstrap intervals
  are reported so it is visible when a cross-language gap sits inside the noise.
  Small gaps should not be over-read; the 48× allocation range and the byte-level
  collapse are far larger than the noise floor.
- `ks_zipf` is measured over each language's *own* support, so it partly
  normalizes away the truncation that is the main effect. It answers "conditional
  on its support, is this Zipf-shaped" — not "how well is this language served."
  Use `effective_vocab` and the token-minus-baseline deltas for the latter.
- **Word-level α lands at 0.64–0.94, not ≈1.** At ~29k words most word types are
  hapaxes, the observed curve is truncated at frequency 1, and MLE over that
  support returns a flatter exponent than the asymptotic value. This is a
  small-corpus property, not an estimator fault: Spearman(types-per-token, α) =
  −0.95, and α rises toward 1 for all 16 languages when the full corpus replaces
  the matched budget. It is also why attribution rests on *deltas* — token and
  word units share the corpus and its size, so the bias largely cancels. Do not
  read absolute α against a textbook α=1 at this corpus size; read it against the
  same text's own baseline.
- A minority of bootstrap draws (1–7%, mostly grapheme units) return
  `converged=False` from L-BFGS-B on a locally flat likelihood surface.
  `converged_share` is reported per cell so this is visible rather than silent.
- **At small support, α and b are jointly near-degenerate.** On English grapheme
  clusters (V=110) the optimum sits near α≈17, b≈133, and moving from (10, 70) to
  (17, 133) changes the log-likelihood by 0.03%. The distribution *shape* is
  identified; the individual parameters are not. Compare α only between
  distributions of broadly similar support size — which is why the token-vs-word
  delta uses α (word supports of 7–16k types are comparable to token supports)
  while the token-vs-grapheme delta uses effective vocabulary instead.
- Every fit carries an **`at_bound`** flag, aggregated per cell as
  `at_bound_share`. A fit on a bound is a truncation of the optimum, not an
  estimate of it, and its α and b must not be read at face value. α = 0 is
  deliberately *not* flagged: that is the uniform limit and a natural edge of the
  parameter space, since a negative exponent cannot fit a descending count vector.
- For a language whose active vocabulary is a few hundred byte fragments, this is
  not a like-for-like exponent comparison with English. The claim is that the
  tokenizer collapses the language onto a support too small to carry a Zipfian
  distribution — a statement about representational capacity.
- Script allocation cannot separate languages sharing a script. Latin-script
  conclusions must cite mass-weighted exclusivity, not raw share.

## Artifacts

- **[PLAN.md](PLAN.md)** — validation plan, decision rule, scope
- **[artifacts/](artifacts/)** — locked languages, tokenizers, BPE experiment arms (`bpe/`, `bpe_constrained/`, `bpe_skew/`, `bpe_parity/`)
- **[results/](results/)** — `metrics.json`, wide CSV tables, decision JSON, premium heatmap PNG (gitignored; produced by `run_eval`)
- **[results/zipf/](results/zipf/)** — vocabulary allocation profile and Zipf deviation fits (gitignored; produced by `run_vocab_profile` + `run_zipf_eval`)
- **[web/](web/)** — static metrics viewer (`data.js` is generated; re-run `export_web_data.py` after each eval)

## Notes

- **Quechua (`quy_Latn`)** substitutes for Nahuatl, which is not in FLORES-200/FLORES+.
- **`data/`** and **`results/`** are gitignored; obtain data locally and regenerate results.
- **`web/data.js`** is generated by `scripts/export_web_data.py`, not hand-edited.
- **`o200k_grapheme`** is supplemental and not part of the default viewer export.
- **Parity-aware BPE** is the recommended OpenAI *next-train* ask when the goal is cross-lingual token-cost parity; it cannot patch frozen `o200k_base` IDs.
- **`src/zipf_langs.py`** deliberately does not extend `src/load_flores.py:LANGUAGES`. That set is the locked 12-language efficiency scope consumed by `run_eval.py` and `export_web_data.py`; editing it in place would shift already-published results.
- **`src/metrics.py:compute_strr`** is occurrence-weighted and is left untouched. The Zipf study's whole-word coverage is the type-level sibling in `src/vocab_profile.py`, so frequent function words cannot mask how a language's long tail fares.
- **`o200k_base` holds no token containing a complete Ethiopic character** and only 38 for Oriya, so Amharic and Tigrinya are encoded almost entirely from partial-UTF-8 byte fragments. This is measured, not assumed — see the regression anchors in `tests/test_vocab_profile.py`.
