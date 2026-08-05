#!/bin/bash
# Path-1 recovery: run both scale trains, sync artifacts, leave host up.
set -euxo pipefail
export CORPUS_S3_ROOT="${CORPUS_S3_ROOT:-s3://edullm-datasets/_scratch/plan-a-fineweb}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
STATUS="${CORPUS_S3_ROOT}/_STATUS.txt"
LOG=/var/log/plan-a-train.log

exec > >(tee -a "$LOG") 2>&1

status() {
  echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | aws s3 cp - "$STATUS" --region "$AWS_REGION"
}

upload_log() {
  aws s3 cp "$LOG" "${CORPUS_S3_ROOT}/_train.log" --region "$AWS_REGION" || true
}

trap 'upload_log; status "TRAIN_FAILED"' ERR

test -x /usr/local/bin/run_plan_a_scale_docker.sh
test -f /data/corpus/scale/train.txt
test -f /data/corpus/scale/manifest.json
docker image inspect tokenizer-benchmark >/dev/null

status "TRAIN_BPE_START"
/usr/local/bin/run_plan_a_scale_docker.sh bpe
upload_log

status "TRAIN_SUPERBPE_START"
/usr/local/bin/run_plan_a_scale_docker.sh superbpe
upload_log

status "SYNC_START"
aws s3 sync /data/tokenizers/scale/ "${CORPUS_S3_ROOT}/tokenizers/scale/" --region "$AWS_REGION"
aws s3 sync /data/results/scale/ "${CORPUS_S3_ROOT}/results/scale/" --region "$AWS_REGION"
upload_log

trap - ERR
status "TRAIN_DONE"
echo "TRAIN_DONE — artifacts under ${CORPUS_S3_ROOT}/tokenizers/scale/ and results/scale/"
