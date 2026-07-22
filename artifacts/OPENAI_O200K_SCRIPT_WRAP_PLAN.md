# OpenAI-facing o200k intervention (deadline path)

**Status:** Plan artifact only — experiment not executed yet.  
**Location:** `C:\Users\aryan\projects\tokenizer-flores-validation\artifacts\`  
**Sources (Validation):** Land & Arnett (*BPE Stays on SCRIPT*); Petrov et al.; Somide (*African Language Tax*); Arnett et al. (token premiums); Script Tax; AraToken (normalization); Velayuthan & Sarveswaran as cited in SCRIPT (grapheme pretok).

**Locked choice:** single o200k addition = **`o200k_grapheme`** (grapheme-healed pretok wrap). No SCRIPT wrap, no SuperBPE arm.

**Revised recommendation (after weighing):** SCRIPT script-boundary wrap alone is **not** the best FLORES demo. Prefer **NFKC + grapheme-preserving pretok wrap** for a measurable o200k tweak; keep **SCRIPT character-integrity** as the OpenAI *training-time* ask; lead with **diagnosis** (premium + STFR + partial-UTF8 vocab count).

**Implementation note:** `o200k_grapheme` heals o200k regex pretok spans onto UAX #29 boundaries, then runs frozen BPE via `_encode_single_piece`. It does **not** replace regex with whitespace (that inflates English CTC by breaking space-prefixed tokens). On many FLORES strings the heal is a no-op because o200k’s pattern already includes `\p{M}`; large STRR/STFR wins still need a retrain (GPE/SCRIPT-BPE).

---

## 0. Alternatives weighed (o200k-adaptable only)

Constraint: must work on **frozen** `o200k_base` via `tiktoken` (no merge-table edit, no LM retrain). Full SCRIPT-BPE retrain is out for the letter.

| Option | What you do | Adapts to o200k? | Likely metric win on FLORES-12 | OpenAI punch | Deadline risk |
|--------|-------------|------------------|--------------------------------|--------------|---------------|
| **A. SCRIPT script-boundary pretok wrap** | Cut on Unicode script/supercategory, then o200k per span | Yes (encode wrap) | **Weak** — FLORES sides are mostly mono-script; few cross-script adjacencies to fix | Good *story* (cites their paper lineage) | Low |
| **B. Grapheme-preserving pretok wrap** (UAX #29 / Velayuthan-style) | Keep base+marks as pretokens (or encode grapheme-aware spans), then o200k | Yes | **Stronger** on Odia, Amharic, Arabic — regex pretok is known to split diacritics/graphemes; STFR/fertility more likely to move | Clear: “your pretok breaks characters” | Low |
| **C. NFKC/NFC normalization wrap** | Normalize Unicode, then o200k | Yes (1 line) | **Small but real** on Ethiopic/Arabic presentation forms | Weak alone; free add-on | Trivial |
| **D. Diagnosis-only** (no wrap “fix”) | Measure o200k premium/STFR; count mixed partial-UTF8 tokens in o200k vocab (GPT-4o had 874) | Yes | N/A (no improvement claim) | **Very strong** smoking gun + ask for constrained merges | Lowest |
| **E. AraToken-style Arabic normalize → o200k** | Dialect-specific norm pipeline then o200k | Yes for `arz`/`ary` | Local win on 2 langs only | Narrow (not whole product) | Low |
| **F. Vocab retrofit** (add tokens + derived embeddings) | New tokens for frequent multi-token strings | Tokenizer side yes; **model** needs embedding hack | Can cut premium a lot | Needs model story; heavier | Medium–high |
| **G. Full SCRIPT-BPE / constrained retrain** | New tokenizer | Not “o200k”; sibling system | Large (paper) | Asks them to retrain | High for deadline |
| **H. Full GPE** (Velayuthan & Sarveswaran) | Retrain BPE on grapheme atoms | Not frozen o200k (new vocab) | Strong on Odia-class scripts | Good long-term ask; not a wrap | High for letter |
| **I. ByteFlow** (Deng et al.) | Tokenizer-free hierarchical LM | **No** — replaces tokenizer + model | N/A for o200k encode eval | Vision piece only | Out of scope |
| **J. Language-specific vocab** (AraToken / Odia / EthioLLM style) | Norm + extend vocab for one family | Partial (norm wraps yes; vocab extend needs model) | Strong locally | Product fragmentation | Medium |

### Verdict

- **SCRIPT wrap (A) alone is not the best solution** for proving a *tiny improvement* on your FLORES slice. Land & Arnett’s biggest wins need **training-time** character-integrity constraints; script-boundary pretok helps mixed-script text more than clean mono-script FLORES.
- **ByteFlow (I)** and **full GPE / SCRIPT-BPE (H/G)** are better *systems* in the literature, but they are **not o200k-adaptable** without abandoning the frozen tokenizer (or the whole tokenizer). Wrong for an OpenAI “tiny tweak to o200k” letter.
- **Best deadline package:** **D + C + B**, with SCRIPT/GPE kept as the **ask**, not the demo wrap.
  1. **Prove o200k is costly** — premium / STFR on Amharic, Odia, Mandarin, Arabic, Quechua, Guarani (Petrov/Somide/Arnett framing).
  2. **Prove a tiny adaptable fix** — `NFKC → grapheme-aware pretok → o200k` vs raw o200k; report delta on high-tax langs (especially Odia/Amharic). Grounded in Velayuthan: pretok often moves compression more than the merge algorithm.
  3. **Actionable ask** — next o200k-class train: **character-integrity merge constraints** (+ script-/grapheme-aware pretok) per Land & Arnett / Velayuthan; not ByteFlow (architecture change).

Optional A/B: also run SCRIPT pretok wrap as a third arm; if it underperforms grapheme wrap on FLORES, say so honestly.

---

## 1. Reality check

SCRIPT has **two** pieces:

1. **SCRIPT encoding + constrained BPE merges** — trains a **new** tokenizer. Does **not** edit frozen `o200k_base`.
2. **Rule-based SCRIPT pretokenization** — script/supercategory spans; plus **character-integrity** constraints (GPT-4o: **874** mixed partial UTF-8 tokens; o200k regex can cascade on Thai).

**Abandon** phonetic / IPA→ipatok.

---

## 2. Deadline-safe experiment (revised choice)

```text
FLORES (12 langs)
   ├─► o200k raw
   ├─► NFKC + grapheme-preserving pretok → o200k   ← primary wrap
   ├─► SCRIPT pretok → o200k                       ← optional comparison arm
   └─► grapheme-cluster atomic baseline            ← reference ceiling/floor
           │
           ▼
   premium, fertility, STRR, STFR + o200k vocab partial-UTF8 count
           │
           ▼
   delta (raw vs grapheme wrap) → OpenAI paragraph
```

1. Metrics for raw **o200k** on FLORES-12.
2. Primary wrap: **NFKC + UAX #29 grapheme-preserving pretokenization**, then frozen o200k per pretoken (or per word with graphemes held intact—implement the variant that best preserves combining marks before calling `tiktoken`).
3. Optional: SCRIPT script-boundary wrap as comparison.
4. Vocab audit: count o200k tokens that decode to mixed full/partial UTF-8 (Land & Arnett-style smoking gun).
5. Skip full SCRIPT-BPE retrain for the letter.

### Time cost

| Work | Estimate |
|------|----------|
| FLORES + o200k metrics | ~0.5 day |
| NFKC + grapheme pretok wrap + remeasure | ~0.5 day |
| Optional SCRIPT pretok arm | ~0.5 day |
| Partial-token vocab audit | ~2–4 hours |
| Full SCRIPT-BPE retrain | **1–3+ days** — skip for the letter |
| Paragraph polish | ~1 hour |

**Total for a credible OpenAI note: ~1–2 days.**

---

## 3. Why wraps can improve metrics (and why SCRIPT alone may not)

### 3.1 SCRIPT script-boundary wrap

**Mechanism:** BPE never merges across script/supercategory cuts.  
**Helps when:** code-switch, Latin digits/punct glued to other scripts, Inherited-mark edge cases.  
**Weak on FLORES:** each language side is mostly one script → few bad cross-script adjacencies → small or null premium/STFR deltas. Still valuable as an OpenAI *recommendation*, weak as your *improvement demo*.

### 3.2 Grapheme-preserving wrap (preferred demo)

**Mechanism:** Regex pretokenizers (GPT/o200k family) can split **base character + combining marks** (documented for Indic etc.; Velayuthan & Sarveswaran; cited in Land & Arnett). Holding **extended grapheme clusters** intact before calling frozen o200k reduces illegal mid-grapheme breaks.  
**Helps:** STFR (fewer shredded marks), fertility/STRR on Odia/Amharic/Arabic, sometimes premium.  
**Risk:** Encoding each grapheme *in isolation* can also *raise* CTC if it blocks useful within-word merges—implement pretok carefully (preserve graphemes inside words, don’t naively 1-grapheme = 1 isolated encode if that hurts). Prefer: normalize → pretok with grapheme boundaries respected inside the string passed to tiktoken’s API where possible, or compare both variants and keep the one that improves metrics.

### 3.3 NFKC

Cheap alignment of compatibility characters; can shrink Amharic/Arabic oddities before o200k sees them. Always on.

### 3.4 Diagnosis (partial UTF-8 vocab count)

Doesn’t improve metrics; **proves** o200k-class tokenizers ship meaningless fragments. Land & Arnett: GPT-4o has 874 mixed partial tokens; o200k regex caused Thai cascade in their training study. Counting o200k’s own bad vocab entries is highly OpenAI-pertinent and deadline-cheap.

### 3.5 Link to Validation literature

| Paper | Role in this pitch |
|-------|-------------------|
| Petrov et al. | Parallel length / unfairness framing |
| Somide | African tax; fertility/CPT; Amharic stress |
| Arnett et al. (premiums) | CTC premium definition |
| Script Tax | Orthography/script as first-order cost |
| Land & Arnett SCRIPT | Character integrity + pretok; GPT-4o partial tokens |
| Velayuthan & Sarveswaran (GPE) | Pretok > algorithm; grapheme atoms; justify grapheme wrap / GPE ask |
| AraToken | Normalization pipelines as wraps (Arabic) |
| ByteFlow | Tokenizer-free alternative — cite as future, not o200k wrap |
| Script Tax / Somide | Latency/cost framing for the letter |

---

## 4. OpenAI-actionable paragraph (template; fill after measurement)

> OpenAI’s o200k tokenizer still imposes a measurable **token premium** on non-Latin and lower-resource languages in parallel FLORES text (same meaning, more tokens—hence more cost and less context—than English). Byte-level BPE in this family can also learn **partial UTF-8 tokens** (hundreds reported for GPT-4o; we count **[N]** mixed partial entries in o200k). Regex-style pretokenization is known to split graphemes in scripts with combining marks. On frozen o200k we applied a **tiny, training-free wrap** (Unicode NFKC + grapheme-preserving pretokenization) and saw **[X%]** lower premium / **[Y%]** lower single-character fragmentation (STFR) on **[languages, e.g. Odia, Amharic]**. The actionable ask for the next tokenizer training run: **enforce character-integrity merge constraints** and **grapheme-/script-aware pretokenization** (Land & Arnett), so multilingual users stop paying a tax that a small training-time change can shrink—without waiting for a full vocab redesign.

---

## 5. Plan-doc updates (when implementing later)

Update P2 Validation and local drafts:

- Drop phonetic / IPA→ipatok.
- Primary intervention: **`o200k+NFKC+grapheme-pretok`**.
- Optional arm: **`o200k+SCRIPT-pretok`**.
- Keep grapheme-cluster atomic baseline as reference.
- Goal: prove o200k tax **and** show a tiny adaptable wrap → justify OpenAI adopting character-integrity + grapheme/script-aware pretok at training time.

---

## 6. What not to claim

- That you patched o200k’s 200k merges.
- That SCRIPT pretok wrap alone equals full SCRIPT-BPE.
- That any wrap removes byte premium entirely.
- Downstream LM quality (encode-only is enough for the letter).

---

*Artifact updated after alternatives review. Experiment not run.*
