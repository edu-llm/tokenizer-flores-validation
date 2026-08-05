# Results

Consolidated experimental results for the multilingual tokenizer studies in this
repo. Figures under [`Results/figures/`](Results/figures/) are copies of the
headline plots so this file renders even when gitignored `results/` /
`artifacts/plan_a/` trees are absent.

---

## 0. Scope map

Three separate studies share code but **not** language sets. Do not mix their
numbers.

| Study | Languages | What it measures | Status |
| --- | ---: | --- | --- |
| Efficiency validation | 12 | Token premium, fertility, STRR/STFR across frozen frontier tokenizers | Done |
| Zipf deviation | 18 | Whether token distributions follow Zipf; allocation vs deviation | Done |
| Plan A / Plan B | 6 | Train gigatoken BPE + SuperBPE, then (later) 1B models | Tokenizers + FLORES suite done; LM BPB deferred |

Plan A languages: English, Hungarian, Mandarin, Hindi, Swahili, Haitian Creole
(`eng_Latn`, `hun_Latn`, `zho_Hans`, `hin_Deva`, `swh_Latn`, `hat_Latn`).

---

## 1. Efficiency validation (12 languages × frontier tokenizers)

Encode-only FLORES metrics over the locked 12-language set
(`src/load_flores.py:LANGUAGES`), produced by `src/run_eval.py`.

### Token premium heatmap

![Token premium heatmap](Results/figures/efficiency_token_premium_heatmap.png)

### Decision rule

From `results/decision.json`:

- **Worth pursuing:** yes (`worth_pursuing: true`)
- **Clause A** (premium ≥ 2 on ≥ 2 frontier tokenizers): **pass** — 5 languages
  hit the bar (`amh_Ethi`, `ory_Orya`, `quy_Latn`, `grn_Latn`, `hun_Latn`)
- **Clause B** (fertility ≥ 2× English): **pass**

### Continent mean token premium (non-English)

Mean `CTC_lang / CTC_eng` by continent (from `continent_mean_premium.csv`):

| Continent | o200k | llama | qwen | superbpe |
| --- | ---: | ---: | ---: | ---: |
| Africa | 2.94 | 3.83 | 2.64 | 3.60 |
| Americas | 1.80 | 2.08 | 2.07 | 2.66 |
| Asia | 2.28 | 4.20 | 3.51 | 5.55 |
| Europe | 1.77 | 1.88 | 2.31 | 3.38 |

### Selected language premiums

| Language | o200k | llama | qwen | glm | superbpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| eng_Latn | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hun_Latn | 1.79 | 2.12 | 2.12 | 1.89 | 2.92 |
| zho_Hans | 1.25 | 1.28 | 1.00 | 0.95 | 2.19 |
| quy_Latn | 1.83 | 2.07 | 2.06 | 2.04 | 2.54 |
| amh_Ethi | 5.78 | 7.59 | 4.05 | 7.59 | 5.90 |
| ory_Orya | 4.99 | 12.16 | 9.70 | 12.36 | 13.53 |

English premium is exactly 1.0 for every tokenizer (by construction).

---

## 2. Zipf deviation (18 languages)

Vocabulary allocation + Zipf–Mandelbrot fits (`scripts/run_vocab_profile.py`,
`scripts/run_zipf_eval.py`, `scripts/plot_zipf.py`). Scope:
`src/zipf_langs.py`.

### Headline figures

![Allocation vs deviation](Results/figures/zipf_allocation_vs_deviation.png)

![Zipf deviation heatmap](Results/figures/zipf_deviation_heatmap.png)

![Word coverage](Results/figures/zipf_word_coverage.png)

![Script allocation](Results/figures/zipf_script_allocation.png)

![Token vs baseline](Results/figures/zipf_token_vs_baseline.png)

### Pre-registered hypotheses (`results/zipf/hypotheses.json`)

**H1 — allocation vs exclusivity.** Latin-script low-resource languages have
high raw active-vocab share but ride English subwords (low exclusive mass).
Spearman (share vs mass exclusivity): ρ = **−0.686**, p = **0.0017**, n = 18.

**H2 — allocation vs deviation (o200k, matched token budget).** Zipf deviation
worsens as active-vocabulary share falls:

| Association | ρ | p | n |
| --- | ---: | ---: | ---: |
| share vs `ks_zipf` | −0.711 | 0.00094 | 18 |
| share vs log effective vocab | +0.653 | 0.0033 | 18 |
| fragment mass vs `ks_zipf` | +0.706 | 0.0010 | 18 |

**H3 — token−baseline deltas track allocation.** Partly supported:

| Contrast | Metric | ρ | p | n |
| --- | --- | ---: | ---: | ---: |
| vs word | Δ `ks_zipf` | −0.109 | 0.69 | 16 |
| vs word | Δ log effective vocab | +0.712 | 0.0020 | 16 |
| vs grapheme | Δ `ks_zipf` | −0.812 | 4.2e-5 | 18 |
| vs grapheme | Δ log effective vocab | +0.802 | 6.3e-5 | 18 |

**H4 — NLLB smaller cross-language spread than frontier.** **Not supported**
(`supported: false` in `hypotheses.json`).

---

## 3. Plan A scale gigatoken (BPE vs SuperBPE, 6 languages)

Trained on the Plan A equal-byte **scale** corpus (6 × 160 MB = 960 MB) with
gigatoken `@00e61db`. FLORES **devtest** suite:
`scripts/eval_plan_a_scale_flores_suite.py` → fertility, token premium, Zipf
`ks_zipf`. **LM BPB is not included** (needs trained models; see §5).

Published as `tokenizer/gigatoken-bpe/v1` and
`tokenizer/gigatoken-superbpe/v1` on `s3://edullm-data`. Pair-verify passed
(common merge prefix through transition vocab 80 000).

### Summary figure

![Plan A FLORES suite summary](Results/figures/plan_a_flores_suite_summary.png)

### Individual panels

![Plan A token premium](Results/figures/plan_a_token_premium.png)

![Plan A fertility](Results/figures/plan_a_fertility.png)

![Plan A ks_zipf](Results/figures/plan_a_ks_zipf.png)

### Token premium vs English (`CTC_lang / CTC_eng`)

| Language | BPE | SuperBPE |
| --- | ---: | ---: |
| English (`eng_Latn`) | 1.000 | 1.000 |
| Hungarian | 1.084 | 1.160 |
| Mandarin | 0.912 | 1.082 |
| Hindi | 1.259 | 1.256 |
| Swahili | 1.032 | 1.056 |
| Haitian Creole | 1.034 | 1.004 |

### Fertility (tokens / word)

| Language | BPE | SuperBPE |
| --- | ---: | ---: |
| English | 1.321 | 1.124 |
| Hungarian | 1.676 | 1.526 |
| Mandarin | 12.759 | 12.874 |
| Hindi | 1.420 | 1.206 |
| Swahili | 1.402 | 1.221 |
| Haitian Creole | 1.293 | 1.068 |

SuperBPE lowers fertility on every non-Mandarin language in this set. Mandarin
is reported but not interpreted as tokens/word (no whitespace word boundary).

### Zipf deviation (`ks_zipf`, matched_sentence, token unit)

Lower = closer to pure Zipf.

| Language | BPE | SuperBPE |
| --- | ---: | ---: |
| English | 0.121 | 0.297 |
| Hungarian | 0.200 | 0.325 |
| Mandarin | 0.292 | 0.310 |
| Hindi | 0.134 | 0.301 |
| Swahili | 0.163 | 0.304 |
| Haitian Creole | 0.080 | 0.294 |

On this suite, **BPE sits closer to Zipf** than SuperBPE for all six languages.

---

## 4. Artifacts published to edullm-data

| Dataset | Version | Notes |
| --- | --- | --- |
| `pretrain/fineweb2-equal-bytes` | v1 | Plan A pilot equal-byte text (JSONL) |
| `tokenizer/gigatoken-bpe` | v1 | Scale BPE, vocab 100 000 |
| `tokenizer/gigatoken-superbpe` | v1 | Scale SuperBPE, transition 80 000 |

Research scratch (not the library address):

- Scale corpus + train artifacts:
  `s3://edullm-datasets/_scratch/plan-a-fineweb/`
- Plan B UniMax pools (pulled, `DONE`):
  `s3://edullm-datasets/plan-b/pools/` — **not yet** published through the
  airlock as `pretrain/fineweb2-unimax-pools`

---

## 5. Deferred / not yet measured

| Item | Why deferred |
| --- | --- |
| **LM bits-per-byte (BPB)** on FLORES / AmericasNLP | Requires trained OLMo (or equivalent) models; tokenizer-only eval cannot produce it |
| Plan B UniMax pool → `edullm-data` | Staging/publish of ~74 GB unique text through landing |
| Plan B token shards `…-bpe-20b` / `…-superbpe-20b` | Needs mixture materialize + published tokenizers (tokenizers ready) |
| Official-trainer cross-check on scale | Gigatoken is the Plan A default; official arm remains the optional parity check |
| AmericasNLP final eval | Downstream of model training |

---

## How to regenerate figures

```powershell
# Efficiency heatmap: python -m src.run_eval --out-dir results
# Zipf plots:         python scripts/plot_zipf.py --results-dir results/zipf
# Plan A suite plots: python scripts/plot_plan_a_scale_flores_suite.py
# Then re-copy into Results/figures/ if you want this report’s embeds refreshed.
```
