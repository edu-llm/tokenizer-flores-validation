#!/bin/bash
# Plan A scale tokenizer host setup (AL2023).
# No auto-shutdown — interactive host for docker train runs.
set -euxo pipefail
exec > >(tee /var/log/plan-a-ec2-setup.log) 2>&1

# Must be a prefix the edullm-downloader instance profile can PutObject to.
# edullm-landing is denied for that role; use edullm-datasets/_scratch.
CORPUS_S3_ROOT="${CORPUS_S3_ROOT:-s3://edullm-datasets/_scratch/plan-a-fineweb}"
AWS_REGION="${AWS_REGION:-us-east-1}"
STATUS="${CORPUS_S3_ROOT}/_STATUS.txt"
LOG_S3="${CORPUS_S3_ROOT}/_setup.log"
WORK=/work
DATA=/data

status() {
  echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" | aws s3 cp - "$STATUS" --region "$AWS_REGION"
}

upload_log() {
  aws s3 cp /var/log/plan-a-ec2-setup.log "$LOG_S3" --region "$AWS_REGION" || true
}

trap 'upload_log; status "SETUP_FAILED"' ERR

status "SETUP_START"

dnf -y update || true
dnf -y install docker git python3.11 tar gzip awscli

systemctl enable --now docker
usermod -aG docker ec2-user || true

mkdir -p "$WORK" "$DATA/corpus/scale" "$DATA/tokenizers/scale" "$DATA/results/scale"
cd "$WORK"

status "SYNC_CODE"
aws s3 sync "${CORPUS_S3_ROOT}/code/" "$WORK/" --region "$AWS_REGION"

status "VENDOR_GIGATOKEN"
aws s3 cp "${CORPUS_S3_ROOT}/vendor/supergigatoken-00e61db.tar.gz" /tmp/gigatoken.tar.gz \
  --region "$AWS_REGION"
rm -rf "$WORK/vendor/gigatoken"
mkdir -p "$WORK/vendor"
tar -xzf /tmp/gigatoken.tar.gz -C "$WORK/vendor"
# tarball root is gigatoken/
test -f "$WORK/vendor/gigatoken/Cargo.toml"
git -C "$WORK/vendor/gigatoken" rev-parse HEAD | grep -q '^00e61db6e885aedd179ae34540caa6b561e3c185$'

status "SYNC_CORPUS"
aws s3 sync "${CORPUS_S3_ROOT}/corpus/scale/" "$DATA/corpus/scale/" --region "$AWS_REGION"
test -f "$DATA/corpus/scale/manifest.json"

status "MATERIALIZE_TRAIN_TXT"
python3.11 "$WORK/scripts/materialize_plan_a_train_txt.py" \
  --corpus-dir "$DATA/corpus/scale" --force

status "DOCKER_BUILD"
# Build can take 15–40+ minutes (Rust + tokenizers forks).
docker build -f docker/tokenizer-benchmark/Dockerfile -t tokenizer-benchmark "$WORK"

status "SMOKE"
docker run --rm --entrypoint /opt/venv-gigatoken/bin/python tokenizer-benchmark \
  -c "import gigatoken; print('gigatoken_ok', gigatoken.__file__)"
docker run --rm --entrypoint git tokenizer-benchmark \
  -C /opt/gigatoken rev-parse HEAD
ls -la "$DATA/corpus/scale/train.txt"
python3.11 - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("/data/corpus/scale")
m = json.loads((root / "manifest.json").read_text())
digest = hashlib.sha256((root / "train.txt").read_bytes()).hexdigest()
assert digest == m["train_txt"]["sha256"], (digest, m["train_txt"]["sha256"])
print("train_txt_sha256_ok", digest)
PY

# Leave helper on disk for the interactive session.
install -m 0755 "$WORK/scripts/run_plan_a_scale_docker.sh" /usr/local/bin/run_plan_a_scale_docker.sh || \
  cp "$WORK/scripts/run_plan_a_scale_docker.sh" /usr/local/bin/run_plan_a_scale_docker.sh

trap - ERR
upload_log
status "SETUP_OK"
echo "SETUP_OK — scale trains NOT started. See /usr/local/bin/run_plan_a_scale_docker.sh"
