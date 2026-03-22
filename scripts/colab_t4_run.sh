#!/usr/bin/env bash
set -euo pipefail

# T4-oriented launcher for fast iteration (not record-track settings).
# Usage:
#   bash scripts/colab_t4_run.sh
# Optional env overrides:
#   RUN_ID, OUTPUT_DIR, LOG_PATH, SUBMISSION_PATH
#   MODEL_DIM, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, MLP_MULT
#   TRAIN_SEQ_LEN, TRAIN_BATCH_TOKENS, VAL_BATCH_SIZE, ITERATIONS, MAX_WALLCLOCK_SECONDS

: "${RUN_ID:=colab_t4_$(date -u +%Y%m%d_%H%M%S)}"
: "${OUTPUT_DIR:=records/track_non_record_16mb/2026-03-22_Lohit_Starter}"
: "${LOG_PATH:=${OUTPUT_DIR}/${RUN_ID}.log}"
: "${SUBMISSION_PATH:=${OUTPUT_DIR}/submission.json}"
: "${GPU_LABEL:=1xT4 (Google Colab)}"

# Conservative defaults for T4 stability.
: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024/}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${VOCAB_SIZE:=1024}"
: "${NUM_LAYERS:=6}"
: "${MODEL_DIM:=384}"
: "${NUM_HEADS:=6}"
: "${NUM_KV_HEADS:=3}"
: "${MLP_MULT:=2}"
: "${TRAIN_SEQ_LEN:=512}"
: "${TRAIN_BATCH_TOKENS:=16384}"
: "${VAL_BATCH_SIZE:=4096}"
: "${VAL_LOSS_EVERY:=0}"
: "${TRAIN_LOG_EVERY:=50}"
: "${ITERATIONS:=4000}"
: "${WARMUP_STEPS:=20}"
: "${WARMDOWN_ITERS:=300}"
: "${MAX_WALLCLOCK_SECONDS:=900}"

mkdir -p "${OUTPUT_DIR}"
echo "[run] RUN_ID=${RUN_ID}"
echo "[run] LOG_PATH=${LOG_PATH}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"

set -x
RUN_ID="${RUN_ID}" \
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
VOCAB_SIZE="${VOCAB_SIZE}" \
NUM_LAYERS="${NUM_LAYERS}" \
MODEL_DIM="${MODEL_DIM}" \
NUM_HEADS="${NUM_HEADS}" \
NUM_KV_HEADS="${NUM_KV_HEADS}" \
MLP_MULT="${MLP_MULT}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}" \
ITERATIONS="${ITERATIONS}" \
WARMUP_STEPS="${WARMUP_STEPS}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "${LOG_PATH}"
set +x

cp train_gpt.py "${OUTPUT_DIR}/train_gpt_${RUN_ID}.py"

python3 scripts/fill_submission_from_log.py \
  --submission "${SUBMISSION_PATH}" \
  --log "${LOG_PATH}" \
  --gpu "${GPU_LABEL}" \
  --track "non-record-16mb"

echo "[run] done"

