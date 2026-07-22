# Tokenizer Multilingual Efficiency Validation — Plan

**Status:** Plan only (not implemented).  
**Location:** `C:\Users\aryan\projects\tokenizer-flores-validation\`  
**Purpose:** Go/no-go check on whether a phonic / fairer tokenizer research direction is worth pursuing.

---

## 1. Research question

Do current production and multilingual tokenizers still impose large, systematic inefficiencies (fragmentation, high fertility, English-relative token premium) on a geographically diverse parallel set? If yes, a new tokenizer design is worth pursuing; if not, revisit languages/tokenizers before investing.

## 2. Decision rule (go / no-go)

Worth pursuing if **either**:

- **(A)** At least one non-English language has **token premium ≥ 2.0** on **≥ 2** of the frontier tokenizers (`o200k`, `glm`, `llama`, `qwen`), **or**
- **(B)** Mean fertility for a non-Latin / complex-morphology language is **≥ 2× English** on the same tokenizer.

Else: do not invest in a new tokenizer yet; widen or retarget the slice first.

## 3. Literature (Zotero only)

**Library:** eduLLM (`libraryID: 2`)  
**Collection:** Phonic Tokenizer → **Validation** (`collectionKey: 2JKKY4BV`)  
**MCP:** `http://127.0.0.1:23120/mcp` (`zotero-integrated-mcp`)

| Role | Paper |
|------|--------|
| Primary method | Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023) — CTC / parallel length ratios |
| Supporting | Script Tax; African Language Tax; Arnett et al. (crosslingual inequities; morphology); Tokenization and the Noiseless Channel; Unpacking Tokenization |

Full item list: [`artifacts/zotero/validation_bibliography.md`](artifacts/zotero/validation_bibliography.md)

Reference code (Petrov): https://github.com/AleksandarPetrov/tokenization-fairness

## 4. Locked scope

### 4.1 Tokenizers (5)

| ID | Tokenizer | Load |
|----|-----------|------|
| `o200k` | OpenAI `o200k_base` | `tiktoken` |
| `o200k_grapheme` | o200k + grapheme-healed pretok (only o200k addition) | `src/grapheme_wrap.py` |
| `glm` | GLM-5.2 | HF `zai-org/GLM-5.2` (tokenizer only) |
| `llama` | Llama 3.1 | HF `meta-llama/Meta-Llama-3.1-8B` (tokenizer; may need HF auth) |
| `qwen` | Qwen2.5 | HF `Qwen/Qwen2.5-7B` |
| `multi` | NLLB-200 multilingual baseline | HF `facebook/nllb-200-distilled-600M` |

Pinned IDs: [`artifacts/tokenizers.json`](artifacts/tokenizers.json)

### 4.2 Languages (12 FLORES-200 codes)

| Continent | Language | Code |
|-----------|----------|------|
| Africa | Swahili | `swh_Latn` |
| Africa | Hausa | `hau_Latn` |
| Africa | Amharic | `amh_Ethi` |
| Asia | Odia | `ory_Orya` |
| Asia | Mandarin | `zho_Hans` |
| Asia | Egyptian Arabic | `arz_Arab` |
| Asia | Moroccan Arabic | `ary_Arab` |
| Europe | Hungarian | `hun_Latn` |
| Europe | Ukrainian | `ukr_Cyrl` |
| Europe | English (control) | `eng_Latn` |
| Americas | Quechua (Ayacucho) | `quy_Latn` |
| Americas | Guarani | `grn_Latn` |

Split: FLORES-200 **devtest** (parallel by sentence index).  
Machine-readable: [`artifacts/languages.json`](artifacts/languages.json)

### 4.3 Metrics

1. **CTC** — total tokens over all sentences; **exclude** BOS/EOS.
2. **Token premium** — `CTC(lang) / CTC(eng_Latn)` (Arnett-style). English = 1.0.
3. **Fertility** — `tokens / words` after Unicode **NFKC** (Petrov). For `zho_Hans`, also tokens/char.
4. **Characters per token** — non-whitespace code points / CTC (Somide CPT).
5. **STRR** — Single Token Retention Rate: share of whitespace words encoded as exactly one token (omit for Mandarin).
6. **STFR** — Single Token Fragmentation Rate: share of tokens with surface length 1.

**Atomic baseline:** Unicode grapheme clusters (UAX #29) as a sixth “tokenizer” for all 12 languages (no G2P). Full IPA→ipatok is not the main baseline; optional English-only IPA sketch only.

## 5. Proposed pipeline (when implementing later)

```text
FLORES-200 devtest (12 langs)
        │
        ▼
Encode with each of 5 tokenizers (tokenizer-only; no LM forward)
        │
        ├─► CTC → token premium vs eng
        ├─► fertility
        └─► chars/token
        │
        ▼
Tables + premium heatmap + continent means
        │
        ▼
Apply decision rule → one-page go/no-go report
```

Suggested module layout (not built yet for execution):

- `src/load_flores.py`
- `src/tokenizers_registry.py`
- `src/metrics.py`
- `src/run_eval.py`
- `results/` (CSV, JSON, heatmap)
- `report/VERDICT.md`

Deps: `tiktoken`, `transformers`, `datasets`, `sentencepiece`, `pandas`, `matplotlib`

## 6. Deliverables (on future execution)

1. Wide tables: fertility, chars/token, token premium (lang × tokenizer)
2. Heatmap of token premium
3. Continent-level mean premium (excl. English)
4. One-page verdict vs decision rule + driver languages

## 7. Out of scope

- Training a new tokenizer  
- Downstream quality (BLEU, MMLU, etc.)  
- Full FLORES-200 (200 languages)  
- Dollar cost / latency benchmarks (cite Script Tax / African Language Tax in discussion only)

## 8. Note on partial scaffold

An early code scaffold may exist under this repo from a prior implementation attempt. **Treat this document + `artifacts/` as the source of truth for now.** Do not run installs/evals until explicitly asked to implement.
