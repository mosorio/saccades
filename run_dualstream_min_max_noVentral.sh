#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_dualstream_min_max.sh <n_glimpses> <min_num> <max_num> 

n_glimpses="${1:-12}" # default 12, or pass first arg to override
min_num="${2:-3}" # default 3, or pass second arg to override
max_num="${3:-7}" # default 7, or pass third arg to override
task="${4:-min}" # default min, or pass fourth arg to override

logdir="./logs"
mkdir -p "$logdir"
ts="$(date +%Y%m%d-%H%M%S)"
logfile_main="${logdir}/main_onehot_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_${ts}.log"

echo "Running min=${min_num}, max=${max_num}, glimpses=${n_glimpses}, task=${task}"
echo "main log:    ${logfile_main}"

python3 main.py \
  --model_type=rnn_classifier2stream --sort --pass_penult --train_on=both \
  --use_loss=both --opt=Adam --wd=0.00001 --distinctive=0.3 --task_type="${task}" \
  --shape_input=logpolar --min_num="${min_num}" --max_num="${max_num}" \
  --n_glimpses="${n_glimpses}" --h_size=1024 --train_shapes=BCDE --test_shapes BCDE \
  --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 \
  --act=lrelu --dropout=0.5 --rep=0 --grid=6 --map_shape_count=4 --head=relational --save_act \
  | tee "${logfile_main}"

