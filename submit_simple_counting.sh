#!/usr/bin/env bash
set -euo pipefail

# Make sure Python flushes output promptly
export PYTHONUNBUFFERED=1

# Log directories
LOG_ROOT="./logs"
DATA_LOG_DIR="$LOG_ROOT/datasets"
TRAIN_LOG_DIR="$LOG_ROOT/training"
mkdir -p "$DATA_LOG_DIR" "$TRAIN_LOG_DIR"

# Helper to run one case: args = N_G MIN_NUM MAX_NUM
run_case() {
  local N_G="$1"; local MIN_NUM="$2"; local MAX_NUM="$3"
  local TS; TS="$(date +'%Y%m%d_%H%M%S')"

  local DATA_LOG_FILE="$DATA_LOG_DIR/data_ng${N_G}_min${MIN_NUM}_max${MAX_NUM}_${TS}.log"
  local TRAIN_LOG_FILE="$TRAIN_LOG_DIR/train_ng${N_G}_min${MIN_NUM}_max${MAX_NUM}_${TS}.log"

  echo "============================================================"
  echo "Starting case: n_glimpses=$N_G, min_num=$MIN_NUM, max_num=$MAX_NUM"
  echo "Dataset log -> $DATA_LOG_FILE"
  echo "Training log -> $TRAIN_LOG_FILE"
  echo "Started at $(date '+%F %T')"
  echo "------------------------------------------------------------"

  echo "=== [$(date '+%F %T')] Generating datasets (n_glimpses=$N_G) ==="
  # run dataset script and tee output to file
  # if your dataset script is bash and prints progress, tee will capture it
  bash datasets/make_all_datasets.sh $N_G $MIN_NUM $MAX_NUM \
    2>&1 | tee "$DATA_LOG_FILE"

  echo "=== [$(date '+%F %T')] Datasets generation finished ==="
  echo "=== [$(date '+%F %T')] Training model (n_glimpses=$N_G) ==="

  # run training and tee output to file
  PYTHONUNBUFFERED=1 python3 main.py \
    --model_type=rnn_classifier2stream \
    --save_act \
    --save_batch_confusion \
    --sort \
    --train_on=both \
    --use_loss=both \
    --opt=Adam \
    --wd=0.00001 \
    --same \
    --shape_input=logpolar \
    --min_num="$MIN_NUM" \
    --max_num="$MAX_NUM" \
    --n_glimpses="$N_G" \
    --h_size=1024 \
    --train_shapes=BCDE \
    --test_shapes BCDE FGHJ \
    --noise_level=0.74 \
    --train_size=100000 \
    --test_size=5000 \
    --n_epochs=300 \
    --act=lrelu \
    --dropout=0.5 \
    --rep=0 \
    --grid=6 \
    2>&1 | tee "$TRAIN_LOG_FILE"

  echo "=== [$(date '+%F %T')] Training finished ==="
  echo "Finished case: n_glimpses=$N_G, min_num=$MIN_NUM, max_num=$MAX_NUM"
  echo "Dataset log: $DATA_LOG_FILE"
  echo "Training log: $TRAIN_LOG_FILE"
  echo "============================================================"
  echo
}

# -------------------------
# CASE 1 (your first block)
# N_G=12, MIN_NUM=1, MAX_NUM=5
# -------------------------
# run_case 12 1 5

# -------------------------
# CASE 2 (your second block)
# N_G=12, MIN_NUM=1, MAX_NUM=7
# -------------------------
run_case 12 1 7

# -------------------------
# CASE 3 (your second block)
# N_G=20, MIN_NUM=1, MAX_NUM=7
# -------------------------
# run_case 20 1 7

echo "All runs complete at $(date '+%F %T')"

# previous code blocks for reference:
# # # * Simple Counting * Only Once
# #  save activations
# python3 main.py --model_type=rnn_classifier2stream --save_act --save_batch_confusion --sort --train_on=both \
#                 --use_loss=both --opt=Adam --wd=0.00001 --same  --shape_input=logpolar \
#                 --min_num=1 --max_num=9 --n_glimpses=19 --h_size=1024 --train_shapes=BCDE \
#                 --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 \
#                 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=0 --grid=6 



