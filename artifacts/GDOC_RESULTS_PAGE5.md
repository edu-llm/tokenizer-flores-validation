# RESULTS — Parity-aware BPE A/B (OpenAI training-time upgrade candidate)

## Setup
- **Arms:** classical byte BPE vs Parity-aware BPE (fair-max merge selection; Foroutan/Meister et al., arXiv:2508.04796)
- **Pretok:** o200k_base regex + NFKC + byte atoms (same recipe as o200k; new merge list — not ID-compatible with frozen o200k)
- **Train:** English-skewed FLORES-200 dev (~85% English byte-mass; Mandarin 5%; tails ≤2.5%)
- **CR selection (parity):** parallel FLORES-200 dev (worst-compressed language each merge)
- **Eval:** balanced FLORES-200 devtest · 12 languages · 1012 sentences · matched vocab 8k / 16k / 32k
- **Artifacts:** `artifacts/bpe_parity/` · viewer Section 4

## Headline
Parity-aware BPE is a clear win where grapheme integrity (Sections 1–3) failed. Gini of tokens-per-line falls ~85–95%; macro token premium collapses toward 1.0; fertility, chars/token, STFR, and STRR improve at every vocab size. Inference cost is identical to classical BPE (only training merge selection changes).

## Macro A/B (parity − byte) on FLORES-200 devtest

| Vocab | Metric | Byte | Parity | Δ | % change |
|------:|--------|-----:|-------:|--:|---------:|
| 8k | Gini (tok/line) | 0.139 | 0.006 | −0.132 | −95.4% |
| 8k | Token premium | 1.858 | 0.994 | −0.864 | −46.5% |
| 8k | Fertility | 5.253 | 4.481 | −0.772 | −14.7% |
| 8k | Chars/token | 1.633 | 2.021 | +0.388 | +23.7% |
| 8k | STFR | 0.415 | 0.305 | −0.110 | −26.6% |
| 8k | STRR | 0.194 | 0.316 | +0.123 | +63.3% |
| 16k | Gini (tok/line) | 0.118 | 0.010 | −0.108 | −91.7% |
| 16k | Token premium | 1.635 | 0.990 | −0.645 | −39.4% |
| 16k | Fertility | 4.224 | 3.871 | −0.353 | −8.4% |
| 16k | Chars/token | 1.954 | 2.330 | +0.376 | +19.2% |
| 16k | STFR | 0.327 | 0.262 | −0.065 | −19.8% |
| 16k | STRR | 0.272 | 0.391 | +0.119 | +43.7% |
| 32k | Gini (tok/line) | 0.095 | 0.014 | −0.081 | −85.0% |
| 32k | Token premium | 1.333 | 0.989 | −0.345 | −25.9% |
| 32k | Fertility | 3.475 | 3.401 | −0.074 | −2.1% |
| 32k | Chars/token | 2.359 | 2.637 | +0.278 | +11.8% |
| 32k | STFR | 0.255 | 0.230 | −0.025 | −9.6% |
| 32k | STRR | 0.380 | 0.470 | +0.090 | +23.6% |

## Tail-language highlights at 8k (byte → parity)
- **Odia (ory_Orya):** fertility 5.73 → 2.65; premium 3.10 → 1.01; STFR 0.58 → 0.27; STRR 0.004 → 0.32
- **Amharic (amh_Ethi):** fertility 5.05 → 2.93; premium 2.41 → 0.98; STFR 0.80 → 0.51; STRR 0.03 → 0.21
- **Egyptian Arabic (arz_Arab):** fertility 3.22 → 2.63; premium 1.73 → 1.00
- **Moroccan Arabic (ary_Arab):** fertility 3.29 → 2.67; premium 1.73 → 0.99
- **Quechua (quy_Latn):** fertility 4.05 → 3.17; premium 1.78 → 0.98
- **Guarani (grn_Latn):** fertility 4.03 → 2.89; premium 1.94 → 0.98

## Computational cost
- **Encode / serve:** identical to classical BPE.
- **Train (this Python research loop):** ~2–3× wall-clock overall; parity_8k alone ~as long as all three classical byte sizes (~16 min). Paper: only O(\|L\|) CR bookkeeping asymptotically; HF Rust `ParityBpeTrainer` would shrink the constant.
- **Full skewed sweep (byte+parity × 3 sizes):** ~54 minutes on this machine.

## OpenAI ask
Keep the o200k pretok recipe; change only the merge objective to Parity-aware (fair-max) on the next o200k-class train. Training-time sibling tokenizer (new vocab/IDs), not a frozen-o200k wrap. Grapheme integrity is not the primary fix (Sections 1–3).

Repo: https://github.com/aryanjverma/tokenizer-flores-validation
