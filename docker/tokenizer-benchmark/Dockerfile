FROM python:3.11-slim

ARG SUPERBPE_COMMIT=bbd09768fc28a875cef48e6bdd66e3a17454628e
ARG TOKENIZERS_COMMIT=757f2a55c0820ed47064e1fe473deea39b7b611b

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
        click filelock psutil pysimdjson "transformers==4.45.2"

WORKDIR /workspace
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY compat/ ./compat/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1
ENV SUPERBPE_TOKENIZERS_COMMIT="${TOKENIZERS_COMMIT}"

ENTRYPOINT ["python", "scripts/run_official_tokenizer_benchmark.py"]

