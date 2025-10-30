#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_dualstream_min_max.sh <min_num> <max_num> <n_glimpses>

n_glimpses="${1:-12}" # default 12, or pass first arg to override
min_num="${2:-3}" # default 3, or pass second arg to override
max_num="${3:-7}" # default 7, or pass third arg to override
task="${4:-min}" # default min, or pass fourth arg to override


logdir="./logs"
mkdir -p "$logdir"

ts="$(date +%Y%m%d-%H%M%S)"
logfile_ventral="${logdir}/ventral_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_${ts}.log"
logfile_main="${logdir}/main_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_${ts}.log"

echo "Running glimpses=${n_glimpses}, min=${min_num}, max=${max_num}, task=${task}"
echo "ventral log: ${logfile_ventral}"
echo "main log:    ${logfile_main}"

# --- Run ventral.py ---
python3 ventral.py \
  --model_type=cnn --policy=cheat+jitter --logpolar --loss=mse --solarize \
  --shape_input=logpolar --min_num="${min_num}" --max_num="${max_num}" \
  --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ \
  --lums 0.1 0.4 0.7 0.3 0.6 0.9 --noise_level=0.74 \
  --train_size=40000 --test_size=4000 --act=lrelu --dropout=0.4 \
  --rep=0 --grid=6 --n_epochs=200 --opt=Adam --n_glimpses="${n_glimpses}" --multiclass \
    | tee "${logfile_ventral}"

# --- Run main.py ---
python3 main.py \
  --model_type=pretrained_ventral-cnn-mse --sort --pass_penult --train_on=both \
  --use_loss=num --opt=Adam --wd=0.00001 --distinctive=0.3 --challenge="${task}" \
  --shape_input=logpolar --min_num="${min_num}" --max_num="${max_num}" \
  --n_glimpses="${n_glimpses}" --h_size=1024 --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ \
  --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 \
  --act=lrelu --dropout=0.5 --rep=0 --grid=6 --map_shape_count=8 --save_act \
  --ventral="ventral_cnn-lrelu_hsize-25_logpolar_num${min_num}-${max_num}_nl-0.74_diff-0-6_grid6_policy-cheat+jitter_lum-[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_trainshapes-BCDEFGHJ__logpolar_40000_loss-mse_opt-Adam_drop0.4_200eps_rep0_ep-200.pt" \
   | tee "${logfile_main}"


