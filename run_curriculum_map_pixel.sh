#!/usr/bin/env bash
set -euo pipefail

# Trains the map head on curriculum-mode datasets (see make_curriculum_datasets.sh),
# using a FROZEN, pretrained ventral CNN's penultimate layer as the "what" stream
# instead of the hand-built symbolic evidence vector (run_curriculum_map.sh).
# Pass the same curriculum/task/train_size/test_size/seed/noise_level/revisit_with_replacement
# used to generate the dataset, so this loads the right file.
#
# The ventral checkpoint must be an 8-way shape classifier (fc3: Linear(10, 8)),
# not one of the older binary target-vs-distractor checkpoints (output_size=2).

curriculum="${1:-2to7}"
task="${2:-count_total}"
train_size="${3:-30000}"
test_size="${4:-5000}"
seed="${5:-42}"
n_epochs="${6:-500}"
h_size="${7:-512}"
ventral_path="${8:-new_results/models_min_multiclass/logpolar/ventral/ventral_cnn-lrelu_hsize-25_logpolar_num3-7_nl-0.74_diff-0-6_grid6_policy-cheat+jitter_lum-[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_trainshapes-BCDEFGHJ__logpolar_40000_loss-mse_opt-Adam_drop0.4_200eps_rep0_ep-200.pt}"
what_divisive_norm="${9:-none}"  # Leave at 'none': normalization now happens inside the ventral
                                  # CNN's own training target (ventral.py --target_norm=sum), not
                                  # as a separate layer here. Kept only for the older none/fixed/
                                  # learned comparison runs against an un-normalized checkpoint.
class_weights_type="${10:-dataset_based}"
abs_weight_fn="${11:-sqrt_inv}"
rnn_type="${12:-gru}"
use_layernorm="${13:-1}"
opt="${14:-Adam}"
lr_schedule="${15:-flat}"
small_model_encoder="${16:-1}"  # keep on -- matches the small model's encoder architecture
lr="${17:-0.001}"
joint_hidden_size="${18:-}"
dry_run="${19:-0}"
wd="${20:-1e-5}"
batch_size="${21:-1024}"
noise_level="${22:-0}"               # must match make_curriculum_datasets.sh
revisit_with_replacement="${23:-1}"  # must match make_curriculum_datasets.sh
shuffle_glimpses="${24:-1}"          # 0 passes --no_shuffle_glimpses (collapses OOD-N accuracy a lot)
logscale="${25:-1}"                  # must match make_curriculum_datasets.sh (part of the filename)
what_divisive_sigma="${26:-1.0}"     # only relevant if what_divisive_norm != none; see note above

echo "curriculum=$curriculum"
echo "task=$task"
echo "train_size=$train_size"
echo "test_size=$test_size"
echo "seed=$seed"
echo "n_epochs=$n_epochs"
echo "h_size=$h_size"
echo "ventral_path=$ventral_path"
echo "what_divisive_norm=$what_divisive_norm"
echo "class_weights_type=$class_weights_type"
echo "abs_weight_fn=$abs_weight_fn"
echo "rnn_type=$rnn_type"
echo "use_layernorm=$use_layernorm"
echo "opt=$opt"
echo "lr_schedule=$lr_schedule"
echo "small_model_encoder=$small_model_encoder"
echo "lr=$lr"
echo "joint_hidden_size=$joint_hidden_size"
echo "wd=$wd"
echo "batch_size=$batch_size"
echo "noise_level=$noise_level"
echo "revisit_with_replacement=$revisit_with_replacement"
echo "shuffle_glimpses=$shuffle_glimpses"
echo "logscale=$logscale"
echo "what_divisive_sigma=$what_divisive_sigma"

if [ ! -f "$ventral_path" ]; then
  echo "ERROR: ventral checkpoint not found at: $ventral_path" >&2
  exit 1
fi

layernorm_flag=""
if [ "$use_layernorm" = "1" ]; then
  layernorm_flag="--use_layernorm"
fi
encoder_flag=""
if [ "$small_model_encoder" = "1" ]; then
  encoder_flag="--small_model_encoder"
fi
lr_flag=""
if [ -n "$lr" ]; then
  lr_flag="--lr=$lr"
fi
joint_flag=""
if [ -n "$joint_hidden_size" ]; then
  joint_flag="--joint_hidden_size $joint_hidden_size"
fi
dry_run_flag=""
if [ "$dry_run" = "1" ]; then
  dry_run_flag="--dry_run"
fi
revisit_flag=""
if [ "$revisit_with_replacement" = "1" ]; then
  revisit_flag="--revisit_with_replacement"
fi
shuffle_flag=""
if [ "$shuffle_glimpses" = "0" ]; then
  shuffle_flag="--no_shuffle_glimpses"
fi
logscale_flag=""
if [ "$logscale" = "1" ]; then
  logscale_flag="--logscale"
fi

python3 main.py \
  --head=map --task_type="$task" --curriculum="$curriculum" \
  --shape_input=logpolar --model_type=pretrained_ventral-cnn-mse --ventral="$ventral_path" --pass_penult \
  --train_on=both --h_size="$h_size" \
  --what_divisive_norm="$what_divisive_norm" --what_divisive_sigma="$what_divisive_sigma" \
  --class_weights_type="$class_weights_type" --abs_weight_fn="$abs_weight_fn" \
  --rnn_type="$rnn_type" $layernorm_flag --opt="$opt" --lr_schedule="$lr_schedule" $encoder_flag $lr_flag $joint_flag \
  --train_shapes BCDEFGHJ --test_shapes BCDEFGHJ \
  --n_glimpses=22 --grid=6 --noise_level=$noise_level --polar --scaling=log \
  --luminances 0.1 0.4 0.7 \
  --curriculum_train_size="$train_size" --curriculum_test_size="$test_size" \
  --train_n_total_values 3 4 5 6 7 8 9 \
  --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --min_num=3 --max_num=9 \
  --n_epochs="$n_epochs" --seed="$seed" --wd="$wd" --batch_size="$batch_size" $revisit_flag $shuffle_flag $logscale_flag $dry_run_flag
