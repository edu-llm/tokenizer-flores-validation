# BPE vs SuperBPE vs Parity Multilingual Evaluation — Cloud Overview

**Project:** `tokenizer-flores-validation`  
**Question:** Does SuperBPE help or hurt non-English languages vs BPE — and does parity-aware BPE close token-cost inequality without SuperBPE’s multiword merges?

---

## What we are doing

1. **Train three matched tokenizers** — BPE, SuperBPE (t80k), and Parity-aware BPE — all at 100k vocabulary — on a **multilingual pretrain sample only** (not on eval benchmarks). Same corpus and language mix for all arms.

2. **Calibrate a shared mixture** from the geometric mean of the three arms’ token premiums on held-out calibration data, then freeze language weights under UniMax-style repeat caps.

3. **Measure compression** on **held-out** text across 16 languages: token premium, fertility, characters per token, STRR, STFR (including fragmented UTF-8), and Gini of tokens-per-line.

4. **Train three matched OLMo 1B models** — OLMo architecture and training recipe (Ai2), same pretrain data and compute budget, weight-tied embeddings at 100k vocab — differing only in tokenizer. Eval benchmarks are excluded from pretraining.

5. **Evaluate downstream quality** using bits-per-byte (BPB) on held-out FLORES-200 and AmericasNLP text, with pairwise deltas SuperBPE−BPE, Parity−BPE, and SuperBPE−Parity.

---

## Resources

| Resource | Use |
|----------|-----|
| **OLMo (Ai2)** | 1B model architecture, training config, and BPB evaluation protocol |
| **UW/OLMo2-8B-BPE & OLMo2-8B-SuperBPE** | Published tokenizer baseline for comparison (tokenizer-only metrics) |
| **AWS CPU Batch / ECR** | Dolma curation, three-arm tokenizer training, shard materialization |
| **AWS GPU instance (B200)** | Train three OLMo 1B models and run BPB evaluation |
| **Multilingual pretrain corpus** | Tokenizer training + OLMo 1B pretrain (~50B tokens initial pass) |
| **AmericasNLP train (Nahuatl, Maya)** | Added to pretrain for low-resource coverage — **not** used for eval |
| **FLORES-200 / AmericasNLP calibration** | Premium calibration + Parity CR-dev only |
| **FLORES-200 devtest** | **Final eval only** — 14 languages, compression + BPB |
| **AmericasNLP final** | **Final eval only** — Nahuatl + Yucatec Maya, compression + BPB |
| **Existing project code** | `tokenizer-flores-validation` metrics, official + parity Batch runners |

**Languages:** English (control), Amharic, Hausa, Swahili, Ukrainian, Polish, Hungarian, Telugu, Odia, Mandarin, Turkish, Aymara, Quechua, Guaraní, Nahuatl, Yucatec Maya.

**Timeline (rough):** Tokenizer triplet + compression — hours–days on CPU; three OLMo 1B training runs — a few days each on B200; BPB eval — hours.

---

*Brief overview for cloud setup. Full experimental plan in the project Plan A / Plan B docs.*
