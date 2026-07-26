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

### Three-arm Plan A / Plan B (BPE + SuperBPE + Parity)

Parity uses official STAGE1 pretok and fair-max merge selection, then exports
HF-compatible `vocab.json` / `merges.txt`:

```powershell
.venv-benchmark/Scripts/python.exe scripts/run_parity_tokenizer_benchmark.py `
  --train-dir artifacts/plan_a/train_langs `
  --dev-dir artifacts/plan_a/dev_langs `
  --output-dir artifacts/plan_a/tokenizers/smoke/parity `
  --result artifacts/plan_a/results/smoke/parity.json `
  --log artifacts/plan_a/logs/smoke/parity.log `
  --num-bytes 2495955 --vocab-size 4096 --max-rss-gb 8 --force
```

Orchestrate the full triplet, premium calibration, and `READY.json`:

```powershell
.venv-benchmark/Scripts/python.exe scripts/run_plan_a_tokenizer_triplet.py `
  --work-dir artifacts/plan_a `
  --train-lang-dir artifacts/plan_a/train_langs `
  --dev-lang-dir artifacts/plan_a/dev_langs `
  --corpus-dir artifacts/tokenizer_benchmark/corpus `
  --corpus-manifest artifacts/tokenizer_benchmark/corpus/manifest.json `
  --superbpe-repo .cache/superbpe `
  --num-bytes 2495955 --vocab-size 4096 --transition-vocab-size 3072 `
  --stage smoke --force
```

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

## Artifacts

- **[PLAN.md](PLAN.md)** — validation plan, decision rule, scope
- **[artifacts/](artifacts/)** — locked languages, tokenizers, BPE experiment arms (`bpe/`, `bpe_constrained/`, `bpe_skew/`, `bpe_parity/`)
- **[results/](results/)** — `metrics.json`, wide CSV tables, decision JSON, premium heatmap PNG (gitignored; produced by `run_eval`)
- **[web/](web/)** — static metrics viewer (`data.js` is generated; re-run `export_web_data.py` after each eval)

## Notes

- **Quechua (`quy_Latn`)** substitutes for Nahuatl, which is not in FLORES-200/FLORES+.
- **`data/`** and **`results/`** are gitignored; obtain data locally and regenerate results.
- **`web/data.js`** is generated by `scripts/export_web_data.py`, not hand-edited.
- **`o200k_grapheme`** is supplemental and not part of the default viewer export.
- **Parity-aware BPE** is the recommended OpenAI *next-train* ask when the goal is cross-lingual token-cost parity; it cannot patch frozen `o200k_base` IDs.
