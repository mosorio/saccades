#!/usr/bin/env bash
set -euo pipefail

# Log directories
LOG_ROOT="./logs"
DATA_LOG_DIR="$LOG_ROOT/datasets"
TRAIN_LOG_DIR="$LOG_ROOT/training"
# mkdir -p "$DATA_LOG_DIR" "$TRAIN_LOG_DIR"

# Make sure Python flushes output promptly
export PYTHONUNBUFFERED=1    

# Helper to run one case: args = N_G MIN_NUM MAX_NUM
run_case() {
  local N_G="$1"; local MIN_NUM="$2"; local MAX_NUM="$3"


  # Train ventral stream for ignore distractors task
  local VTS; VTS="$(date +'%Y%m%d_%H%M%S')"
  local V_TRAIN_LOG_FILE="$TRAIN_LOG_DIR/train_ventralstream_${VTS}.log"
  echo "============================================================"
  echo "Training ventral stream for ignore distractors task"
  PYTHONUNBUFFERED=1 python3 ventral.py \
    --model_type=cnn --policy=cheat+jitter --logpolar --sort --loss=mse \
    --solarize --same --challenge=distract012 --shape_input=logpolar \
    --min_num=$MIN_NUM --max_num=$MAX_NUM --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ \
    --lums 0.1 0.4 0.7 0.3 0.6 0.9 --noise_level=0.74 --train_size=40000 --test_size=4000 \
    --act=lrelu --dropout=0.4 --rep=0 --grid=6 --n_epochs=200 --opt=Adam --n_glimpses=$N_G \
    2>&1 | tee "$V_TRAIN_LOG_FILE"
    
  echo "=== [$(date '+%F %T')] Training finished ==="
  echo "Finished ventral stream training for ignore distractors task"
  echo "Training log: $V_TRAIN_LOG_FILE"
  echo "============================================================"

  # Train ignore distractor task
  local TS; TS="$(date +'%Y%m%d_%H%M%S')"
  local TRAIN_LOG_FILE="$TRAIN_LOG_DIR/DC_ng${N_G}_min${MIN_NUM}_max${MAX_NUM}_${TS}.log"

  echo "============================================================"
  echo "Starting counting with distractors: n_glimpses=$N_G, min_num=$MIN_NUM, max_num=$MAX_NUM"
  echo "Training log -> $TRAIN_LOG_FILE"
  echo "Started at $(date '+%F %T')"
  echo "------------------------------------------------------------"

  echo "=== [$(date '+%F %T')] Training model (n_glimpses=$N_G) ==="

  # run training and tee output to file
  PYTHONUNBUFFERED=1 python3 main.py \
    --model_type=pretrained_ventral-cnn-mse \
    --sort \
    --save_act \
    --save_batch_confusion \
    --pass_penult \
    --train_on=both \
    --use_loss=both \
    --opt=Adam \
    --wd=0.00001 \
    --same \
    --challenge=distract012 \
    --shape_input=logpolar \
    --min_num=$MIN_NUM \
    --max_num=$MAX_NUM \
    --n_glimpses=$N_G \
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
    --ventral='ventral_cnn-lrelu_hsize-25_logpolar_num1-7_nl-0.74_diff-0-6_grid6_policy-cheat+jitter_lum-[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_trainshapes-BCDEFGHJsame_distract012_logpolar_40000_loss-mse_opt-Adam_drop0.4_sort_200eps_rep0_ep-200.pt'
    2>&1 | tee "$TRAIN_LOG_FILE"
    

  echo "=== [$(date '+%F %T')] Training finished ==="
  echo "Finished case: n_glimpses=$N_G, min_num=$MIN_NUM, max_num=$MAX_NUM"
  echo "Training log: $TRAIN_LOG_FILE"
  echo "============================================================"
  echo
}

echo "Ignore Distractors"
# -------------------------
# CASE 1 (your first block)
# N_G=12, MIN_NUM=1, MAX_NUM=5
# -------------------------
run_case 12 1 5

# -------------------------
# CASE 2 (your second block)
# N_G=12, MIN_NUM=1, MAX_NUM=7
# -------------------------
run_case 12 1 7

# -------------------------
# CASE 3 (your second block)
# N_G=20, MIN_NUM=1, MAX_NUM=7
# -------------------------
run_case 20 1 7

echo "All runs complete at $(date '+%F %T')"

# previous code blocks for reference:
# # # * Simple Counting * Only Once
# #  save activations
# python3 main.py --model_type=rnn_classifier2stream --save_act --save_batch_confusion --sort --train_on=both \
#                 --use_loss=both --opt=Adam --wd=0.00001 --same  --shape_input=logpolar \
#                 --min_num=1 --max_num=9 --n_glimpses=19 --h_size=1024 --train_shapes=BCDE \
#                 --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 \
#                 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=0 --grid=6 



