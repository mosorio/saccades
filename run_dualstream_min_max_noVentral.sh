#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_dualstream_min_max.sh <n_glimpses> <min_num> <max_num> 

n_glimpses="${1:-12}" # default 12, or pass first arg to override
min_num="${2:-3}" # default 3, or pass second arg to override
max_num="${3:-7}" # default 7, or pass third arg to override
task="${4:-min}" # default min, or pass fourth arg to override
head="${5:-relational}"
count_mode="${6:-total}"
negw="${7:-0.05}"
gamma="${8:-2.0}"
seed="${9:-1}"


# logdir="./logs"
# mkdir -p "$logdir"
# ts="$(date +%Y%m%d-%H%M%S)"
# logfile_main="${logdir}/main_onehot_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_head${head}_count_mode${count_mode}_${ts}.log"

#for seed in 0 1 2; do                # different random seeds
for negw in 0.01 0.05 0.1; do       # absent-channel weights
  for gamma in 1.0 2.0 4.0; do         # focal gammas
    logdir="./logs"
    mkdir -p "$logdir"
    ts="$(date +%Y%m%d-%H%M%S)"
    logfile_main="${logdir}/main_onehot_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_head${head}_count_mode${count_mode}_seed${seed}_negw${negw}_gamma${gamma}_${ts}.log"
    echo "Running min=${min_num}, max=${max_num}, glimpses=${n_glimpses}, task=${task}, head=${head}, count_mode=${count_mode}, seed=${seed}, negw=${negw}, gamma=${gamma}"
    echo "main log:    ${logfile_main}"

    python3 main.py \
      --model_type=rnn_classifier2stream --pass_penult --train_on=both \
      --use_loss=both --opt=Adam --wd=0.00001 --distinctive=0.3 --task_type="${task}" \
      --shape_input=symbolic --min_num="${min_num}" --max_num="${max_num}" \
      --n_glimpses="${n_glimpses}" --h_size=1024 --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ \
      --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --seed=1 --map_neg_w="${negw}" --map_focal_gamma="${gamma}" \
      --act=lrelu --dropout=0.5 --rep=0 --grid=6 --map_shape_count=8 --head="${head}" --count_mode="${count_mode}" \
      | tee "${logfile_main}"

  done
done
#done

