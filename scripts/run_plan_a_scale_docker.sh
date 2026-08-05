#!/bin/bash
# Run Plan A scale gigatoken BPE or SuperBPE inside the host-built image.
# Usage: run_plan_a_scale_docker.sh bpe|superbpe
set -euo pipefail

ARM="${1:?usage: $0 bpe|superbpe}"
DATA="${DATA:-/data}"
IMAGE="${IMAGE:-tokenizer-benchmark}"
CORPUS="${DATA}/corpus/scale"
OUT_BASE="${DATA}/tokenizers/scale"
RES_BASE="${DATA}/results/scale"
LOG_BASE="${DATA}/results/scale/logs"

mkdir -p "$OUT_BASE" "$RES_BASE" "$LOG_BASE"

VOCAB=100000
TRANSITION=80000
MAX_RSS=26

case "$ARM" in
  bpe)
    # --allow-dirty: vendored tarball extracts with line-ending noise; HEAD is still 00e61db.
    EXTRA=(--vocab-size "$VOCAB" --max-rss-gb "$MAX_RSS" --allow-dirty)
    OUT="${OUT_BASE}/bpe"
    RESULT="${RES_BASE}/bpe.json"
    LOG="${LOG_BASE}/bpe.log"
    ;;
  superbpe)
    EXTRA=(
      --vocab-size "$VOCAB"
      --transition-vocab-size "$TRANSITION"
      --max-rss-gb "$MAX_RSS"
      --allow-dirty
    )
    OUT="${OUT_BASE}/superbpe"
    RESULT="${RES_BASE}/superbpe.json"
    LOG="${LOG_BASE}/superbpe.log"
    ;;
  *)
    echo "arm must be bpe or superbpe" >&2
    exit 2
    ;;
esac

echo "Starting scale ${ARM} on ${IMAGE} (max_rss_gb=${MAX_RSS})"
docker run --rm \
  --entrypoint python \
  -v "${CORPUS}:/data/corpus/scale:ro" \
  -v "${OUT_BASE}:/data/tokenizers/scale" \
  -v "${RES_BASE}:/data/results/scale" \
  "$IMAGE" \
  scripts/run_gigatoken_tokenizer_benchmark.py \
  --arm "$ARM" \
  --gigatoken-repo /opt/gigatoken \
  --gigatoken-python /opt/venv-gigatoken/bin/python \
  --corpus-file /data/corpus/scale/train.txt \
  --corpus-manifest /data/corpus/scale/manifest.json \
  --output-dir "$OUT" \
  --result "$RESULT" \
  --log "$LOG" \
  "${EXTRA[@]}"

echo "Done ${ARM}. Artifacts under ${OUT}; result ${RESULT}"
echo "Sync back example:"
echo "  aws s3 sync ${OUT_BASE}/ ${CORPUS_S3_ROOT:-s3://edullm-datasets/_scratch/plan-a-fineweb}/tokenizers/scale/"
echo "  aws s3 sync ${RES_BASE}/ ${CORPUS_S3_ROOT:-s3://edullm-datasets/_scratch/plan-a-fineweb}/results/scale/"
