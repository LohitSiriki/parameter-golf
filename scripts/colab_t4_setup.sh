#!/usr/bin/env bash
set -euo pipefail

# One-time setup for Google Colab T4 sessions.
# Usage:
#   bash scripts/colab_t4_setup.sh
# Optional env:
#   TRAIN_SHARDS=1  (default)
#   VARIANT=sp1024  (default)

TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
VARIANT="${VARIANT:-sp1024}"

echo "[setup] GPU info"
nvidia-smi || true

echo "[setup] installing Python deps"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "[setup] downloading cached FineWeb subset (variant=${VARIANT}, train_shards=${TRAIN_SHARDS})"
python3 data/cached_challenge_fineweb.py --variant "${VARIANT}" --train-shards "${TRAIN_SHARDS}"

echo "[setup] done"

