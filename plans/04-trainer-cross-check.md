# Trainer cross-check — gigatoken vs the official SuperBPE implementation

**Status:** Measured. Gate green at smoke and pilot; scale not yet run.
**Scope:** Plan A tokenizer training only. Does not touch the published
12-language efficiency scope or the 18-language Zipf study, neither of which
trains a tokenizer.

Plan A trains on [supergigatoken](https://github.com/aryanjverma/supergigatoken)
rather than the official SuperBPE fork. This document records what was measured
before that switch was trusted, and the one place the two trainers do **not**
agree.

Reproduce with `scripts/compare_plan_a_trainers.py`.

---

## 1. Why a cross-check at all

gigatoken's `train_bpe` / `train_superbpe` were hardcoded to the GPT-2 (r50k)
pretokenizer. GPT-2's ` ?\p{L}+` excludes `\p{M}`, so combining marks fall out
of the letter run into the punctuation branch: `हिन्दी` becomes six pretokens
instead of one, and BPE cannot merge across pretoken boundaries, so the
resulting vocabulary contains no consonant+matra unit at all.

This does **not** cancel between arms. Plan A ships a per-language *premium*
table — a cross-*language* ratio feeding the UniMax allocation and the
byte↔token conversion for the 20B target — so a stage-1 difference present
identically in both arms survives the geometric mean.

It is also worse than a constant offset: gigatoken's stage 2 applies no regex,
so it removes every pretoken boundary including the matra splits stage 1
introduced. The SuperBPE arm gets that repair and the BPE arm does not,
inflating the apparent superword gain for exactly the scripts using combining
marks.

The fix was `PretokenizerType::SuperBPEStage1` (supergigatoken `00e61db`), the
official `STAGE1_REGEX` — which turns out to be the existing Nemotron scheme
with `\p{N}{1,3}` in place of `\p{N}`.

## 2. What the stage-1 fix bought

`gigatoken-bpe` (`superbpe_stage1`) vs `gigatoken-bpe-gpt2`, identical in every
other respect. Bytes/token on FLORES dev, pilot tier:

| language | `gpt2` | `superbpe_stage1` | change |
|---|---:|---:|---:|
| `hin_Deva` | 3.7529 | 6.7972 | **+81.12%** |
| `zho_Hans` | 3.2235 | 3.2292 | +0.18% |
| `eng_Latn` | 3.5483 | 3.5153 | −0.93% |
| `hun_Latn` | 3.5509 | 3.5173 | −0.95% |
| `swh_Latn` | 3.6219 | 3.5812 | −1.12% |
| `hat_Latn` | 3.4227 | 3.3820 | −1.19% |

Only Devanagari moves materially; the small losses elsewhere are GPT-2's
English contraction handling, which the official stage-1 regex drops.

In premium terms the harm was large: under `gpt2`, `hin_Deva` measures **2.4319**
against **1.3303** — an 83% overstatement, fed straight into a mixture where
`hin_Deva` is 19.21%.

## 3. Parity after the fix

Relative premium difference, gigatoken vs official, same corpus, FLORES dev.

| language | BPE arm (smoke / pilot) | SuperBPE arm (smoke / pilot) |
|---|---:|---:|
| `eng_Latn` | +0.00% / +0.00% | +0.00% / +0.00% |
| `hun_Latn` | +0.08% / −0.01% | +0.15% / −0.10% |
| `zho_Hans` | +0.06% / +0.03% | −0.10% / −0.35% |
| `hin_Deva` | −0.02% / +0.05% | **−3.13% / −6.61%** |
| `swh_Latn` | +0.15% / −0.01% | +0.36% / −0.34% |
| `hat_Latn` | +0.06% / −0.01% | −0.01% / −0.34% |

The BPE arm reproduces the reference to within 0.15%, at both tiers. That is
the stage-1 scheme validated.

`verify_official_tokenizer_pair.py` on the gigatoken pair: common merge prefix
2816/2816, zero shared-ID mismatches. gigatoken continues stage 2 in-process, so
the inherited prefix is exact rather than approximate.

## 4. The one real divergence: Devanagari on the SuperBPE arm

`hin_Deva` differs by −3.13% at 4k vocab and −6.61% at 16k. **It grows with
vocab size**, so it must be re-read at every tier rather than assumed stable.

The cause is in the reference, not in gigatoken. The official stage-2 regex is:

```
\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)
```

The `[^\s\p{L}\p{N}]{2,}` alternative exists to catch punctuation runs (`...`,
`--`). Devanagari combining marks are neither `\p{L}` nor `\p{N}`, so a
two-mark vowel cluster matches it too. Verified directly:

| input | official stage-2 pieces |
|---|---|
| `हिन्दी भाषा में कई शब्द हैं` | 4 — `['हिन्दी भाषा म', 'ें', ' कई शब्द ह', 'ैं']` |
| `the quick brown fox` | 1 |
| `mbweha wa kahawia mwepesi` | 1 |

Official stage 2 therefore caps how far superwords can compress Hindi.
gigatoken's regex-free stage 2 has no such cap, which is why it compresses
Hindi *better* (7.1853 vs 6.6835 bytes/token at pilot), not worse.

### Decision

Keep gigatoken's behaviour. Reproducing an English-centric regex artifact on
Indic script would make the premium table less accurate, not more.

### Consequences that must be carried into any writeup

- The SuperBPE-arm premium table is **gigatoken-SuperBPE's own**. For
  `hin_Deva` it is **not** directly comparable to published SuperBPE
  (Liu et al.) numbers. The BPE arm is comparable.
- gigatoken's stage-2 superword space is a strict superset of the reference's:
  units are bounded only by separator, newline, and `max_unit_len=128`, so
  superwords may bridge digits and punctuation the reference splits. Superwords
  never cross a newline.
- The gate encodes this as a *named exemption with a mechanism*, not a loosened
  threshold: `STAGE2_DEVIATION_EXPECTED` in
  `scripts/compare_plan_a_trainers.py`. Everything else is held to 1%
  (worst observed non-exempt: 0.36%). The exemption has its own 15% tripwire
  so a change in mechanism still fails the gate.

## 5. Cost

Smoke tier, 12 MB corpus, 4k vocab, 8-core CPU:

| run | wall | CPU | peak RSS |
|---|---:|---:|---:|
| official BPE | 5.65 s | 11.5 s | 414 MiB |
| official SuperBPE | 11.35 s | 38.3 s | 887 MiB |
| gigatoken BPE | 1.54 s | 2.9 s | 137 MiB |
| gigatoken SuperBPE | 3.57 s | 5.2 s | 208 MiB |

~3.2–3.7× faster and **4.3× lower peak RSS** on the SuperBPE arm — the figure
that matters against the scale tier's 26 GB guard.

## 6. Data availability, corrected

`plans/01-data-sourcing.md` flags `hat_Latn` at 1.87× the scale budget as "the
thinnest margin in the set and the only real risk to the scale tier". That
figure is parquet bytes across eleven columns. Measured on extracted text, a
160 MB pull consumed 56,461 of 226,754 available documents, implying roughly
0.70 GB of text — about **4× the scale budget**. The risk is smaller than the
parquet figure suggests.

All six languages are staged at 160 MB in
`artifacts/plan_a/raw/fineweb2_samples/`, enough to build any tier.
