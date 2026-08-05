FROM python:3.11-slim

ARG SUPERBPE_COMMIT=bbd09768fc28a875cef48e6bdd66e3a17454628e
ARG TOKENIZERS_COMMIT=757f2a55c0820ed47064e1fe473deea39b7b611b
# Must match gigatoken_pipeline.commit in configs/benchmarks/tokenizer_local.json.
ARG GIGATOKEN_REPO=https://github.com/aryanjverma/supergigatoken.git
ARG GIGATOKEN_COMMIT=00e61db6e885aedd179ae34540caa6b561e3c185

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal

RUN git clone https://github.com/alisawuffles/tokenizers-superbpe.git /opt/tokenizers-superbpe \
    && git -C /opt/tokenizers-superbpe checkout "${TOKENIZERS_COMMIT}" \
    && pip install --no-cache-dir /opt/tokenizers-superbpe/bindings/python

RUN git clone https://github.com/PythonNut/superbpe.git /opt/superbpe \
    && git -C /opt/superbpe checkout "${SUPERBPE_COMMIT}" \
    && pip install --no-cache-dir \
        click filelock psutil pysimdjson regex tiktoken \
        "transformers==4.45.2"

# gigatoken, the default Plan A trainer, in its OWN virtualenv.
#
# It cannot share the image's site-packages: the official arm pins `tokenizers`
# to the patched SuperBPE fork installed above, and gigatoken needs stock
# `tokenizers` for the artifact export. This mirrors the local three-env split
# that CLAUDE.md requires be kept separate. The Rust toolchain installed above
# for the fork is reused here to build gigatoken's native extension.
#
# Prefer a vendored tree at vendor/gigatoken/ (must include Cargo.toml and
# .git so run_gigatoken_tokenizer_benchmark.py can verify the pin). When that
# path is only the repo placeholder, fall back to cloning GIGATOKEN_COMMIT.
COPY vendor/gigatoken/ /tmp/gigatoken-vendor/
RUN if [ -f /tmp/gigatoken-vendor/Cargo.toml ]; then \
        mv /tmp/gigatoken-vendor /opt/gigatoken ; \
    else \
        rm -rf /tmp/gigatoken-vendor \
        && git clone "${GIGATOKEN_REPO}" /opt/gigatoken \
        && git -C /opt/gigatoken checkout "${GIGATOKEN_COMMIT}" ; \
    fi \
    && python -m venv /opt/venv-gigatoken \
    && /opt/venv-gigatoken/bin/pip install --no-cache-dir maturin tokenizers \
    && /opt/venv-gigatoken/bin/pip install --no-cache-dir /opt/gigatoken \
    && /opt/venv-gigatoken/bin/python -c "import gigatoken; print(gigatoken.__file__)" \
    && git -C /opt/gigatoken rev-parse HEAD

WORKDIR /workspace
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY compat/ ./compat/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1
ENV SUPERBPE_TOKENIZERS_COMMIT="${TOKENIZERS_COMMIT}"
ENV GIGATOKEN_REPO_DIR=/opt/gigatoken
ENV GIGATOKEN_PYTHON=/opt/venv-gigatoken/bin/python
ENV PYTHONPATH=/workspace

# Default: the official BPE/SuperBPE runner; --arm selects the arm. Unchanged
# so existing AWS Batch job definitions keep working.
#
# For the gigatoken arm (the default trainer, and the one the scale tier runs
# on), override the entrypoint:
#
#   --entrypoint python \
#     scripts/run_gigatoken_tokenizer_benchmark.py --arm bpe \
#       --gigatoken-repo /opt/gigatoken \
#       --gigatoken-python /opt/venv-gigatoken/bin/python ...
#
# Both runners emit the same artifact set and result schema, so everything
# downstream is trainer-agnostic. Keeping both in one image is deliberate:
# scripts/compare_plan_a_trainers.py is the gate for every tier, and it needs
# to train both sides on the same corpus.
ENTRYPOINT ["python", "scripts/run_official_tokenizer_benchmark.py"]
