#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_dualstream_min_max.sh <n_glimpses> <min_num> <max_num> 

n_glimpses="${1:-12}" # default 12, or pass first arg to override
min_num="${2:-3}" # default 3, or pass second arg to override
max_num="${3:-7}" # default 7, or pass third arg to override
task="${4:-min}" # default min, or pass fourth arg to override
head="${5:-relational}"
count_mode="${6:-total}"
seed="${7:-1}"
map_mode="${8:-both}"
ce_weight="${9:-1.0}"
bce_weight="${10:-1.0}"


logdir="./logs"
mkdir -p "$logdir"
ts="$(date +%Y%m%d-%H%M%S)"
logfile_main="${logdir}/main_onehot_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_head${head}_count_mode${count_mode}_${ts}.log"

  
logdir="./logs"
mkdir -p "$logdir"
ts="$(date +%Y%m%d-%H%M%S)"
logfile_main="${logdir}/main_symbolic_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_head${head}_count_mode${count_mode}_seed${seed}_map_mode_bce_ce_weight${ce_weight}_bce_weight${bce_weight}_pos_weight=True_${ts}.log"
echo "Running min=${min_num}, max=${max_num}, glimpses=${n_glimpses}, task=${task}, head=${head}, count_mode=${count_mode}, seed=${seed}, map_mode_bce, ce_weight=${ce_weight}, bce_weight=${bce_weight}, pos_weight=True"
echo "main log:    ${logfile_main}"

python3 main.py \
  --model_type=rnn_classifier2stream --pass_penult --train_on=both \
  --use_loss=both --opt=Adam --wd=0.00001 --distinctive=0.3 --task_type="${task}" \
  --shape_input=symbolic --min_num="${min_num}" --max_num="${max_num}" \
  --n_glimpses="${n_glimpses}" --h_size=1024 --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ \
  --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --seed=1 --map_mode=bce \
  --act=lrelu --dropout=0.5 --rep=0 --grid=6 --map_shape_count=8 --head="${head}" --count_mode="${count_mode}" --ce_weight="${ce_weight}" --bce_weight="${bce_weight}" --pos_weight \
  | tee "${logfile_main}"


# for ce_weight in 1.0 0.5; do 
#   for bce_weight in 1.0 0.5; do     
#     logdir="./logs"
#     mkdir -p "$logdir"
#     ts="$(date +%Y%m%d-%H%M%S)"
#     logfile_main="${logdir}/main_onehot_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_head${head}_count_mode${count_mode}_seed${seed}_map_mode${map_mode}_ce_weight${ce_weight}_bce_weight${bce_weight}_pos_weight=False_${ts}.log"
#     echo "Running min=${min_num}, max=${max_num}, glimpses=${n_glimpses}, task=${task}, head=${head}, count_mode=${count_mode}, seed=${seed}, map_mode=${map_mode}, ce_weight=${ce_weight}, bce_weight=${bce_weight}, pos_weight=False"
#     echo "main log:    ${logfile_main}"

#     python3 main.py \
#       --model_type=rnn_classifier2stream --pass_penult --train_on=both \
#       --use_loss=both --opt=Adam --wd=0.00001 --distinctive=0.3 --task_type="${task}" \
#       --shape_input=symbolic --min_num="${min_num}" --max_num="${max_num}" \
#       --n_glimpses="${n_glimpses}" --h_size=1024 --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ \
#       --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --seed=1 --map_mode="${map_mode}" \
#       --act=lrelu --dropout=0.5 --rep=0 --grid=6 --map_shape_count=8 --head="${head}" --count_mode="${count_mode}" --ce_weight="${ce_weight}" --bce_weight="${bce_weight}" \
#       | tee "${logfile_main}"
#   done
# done

for ce_weight in 1.0 0.5; do 
  for bce_weight in 1.0 0.5; do     
    logdir="./logs"
    mkdir -p "$logdir"
    ts="$(date +%Y%m%d-%H%M%S)"
    logfile_main="${logdir}/main_onehot_min${min_num}_max${max_num}_glim${n_glimpses}_task${task}_head${head}_count_mode${count_mode}_seed${seed}_map_mode${map_mode}_ce_weight${ce_weight}_bce_weight${bce_weight}pos_weight=True_${ts}.log"
    echo "Running min=${min_num}, max=${max_num}, glimpses=${n_glimpses}, task=${task}, head=${head}, count_mode=${count_mode}, seed=${seed}, map_mode=${map_mode}, ce_weight=${ce_weight}, bce_weight=${bce_weight}, pos_weight=True"
    echo "main log:    ${logfile_main}"

    python3 main.py \
      --model_type=rnn_classifier2stream --pass_penult --train_on=both \
      --use_loss=both --opt=Adam --wd=0.00001 --distinctive=0.3 --task_type="${task}" \
      --shape_input=symbolic --min_num="${min_num}" --max_num="${max_num}" \
      --n_glimpses="${n_glimpses}" --h_size=1024 --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ \
      --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --seed=1 --map_mode="${map_mode}" \
      --act=lrelu --dropout=0.5 --rep=0 --grid=6 --map_shape_count=8 --head="${head}" --count_mode="${count_mode}" --ce_weight="${ce_weight}" --bce_weight="${bce_weight}" --pos_weight \
      | tee "${logfile_main}"
  done
done


