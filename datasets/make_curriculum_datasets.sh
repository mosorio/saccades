#!/usr/bin/env bash
set -euo pipefail

# Small test set for curriculum-mode dataset generation.
# Generates a train split plus the four curriculum test conditions:
#   ood_combo_inrange_n  : heldout shape-combos x N values seen during training
#   ood_combo_ood_n      : heldout shape-combos x N values NOT seen during training
#   seen_combo_ood_n     : trained shape-combos  x N values NOT seen during training
#   seen_combo_inrange_n : trained shape-combos  x N values seen during training ("validation" --
#                          a fresh held-out sample from the same distribution as training)
#
# Default train/test sizes (30000/5000).

curriculum="${1:-2to7}"   # curriculum preset: 2to7, 246, 357
task="${2:-count_total}"       # task_type: count_total, count_per_shape, min, max
train_size="${3:-30000}" # examples in the train split
test_size="${4:-5000}"   # examples in each of the 4 test-condition splits
seed="${5:-42}"          # shared seed so train/test calls resolve the same combo split
what_symbolic_radius="${6:-0.665}"  # "what" evidence radius -- 0.665 matches the small model's
                                     # radius expressed as a fraction of slot spacing (~4.15
                                     # slot-widths on a grid=6 scene), vs. the legacy ~0.22
                                     # (~1.4 slot-widths) derived accidentally from noise_level.
                                     # Pass empty string "" to fall back to the legacy formula.
logscale="${7:-1}"        # 1 to pass --logscale (log-weighted falloff within the radius,
                           # matching the small model's kernel="log"), 0 for linear falloff.
noise_level="${8:-0}"      # glimpse-position jitter std = 0.1*noise_level (e.g. ~0.074 at the
                            # old default 0.74 -- ~46% of a slot-width on a grid=6 scene).
                            # Default 0 matches the small model's where_xy_noise_std=0.0 (exact,
                            # noiseless gaze coordinates); with wide --what_symbolic_radius, the
                            # model leans on precise "where" to bind evidence to the right slot,
                            # so this jitter became a real bottleneck once the radius widened.
revisit_with_replacement="${9:-1}"  # 1 to pass --revisit_with_replacement: after the guaranteed
                                     # first-pass glimpse at every object, sample remaining glimpses
                                     # i.i.d. with replacement (matches the small model's
                                     # sample_loc_sequence_cover_all_occupied). 0 (default) keeps the
                                     # legacy tiled-repeat-then-truncate behavior.

echo "curriculum=$curriculum"
echo "task=$task"
echo "train_size=$train_size"
echo "test_size=$test_size"
echo "seed=$seed"
echo "what_symbolic_radius=$what_symbolic_radius"
echo "logscale=$logscale"
echo "noise_level=$noise_level"
echo "revisit_with_replacement=$revisit_with_replacement"

radius_flag=""
if [ -n "$what_symbolic_radius" ]; then
  radius_flag="--what_symbolic_radius=$what_symbolic_radius"
fi
logscale_flag=""
if [ "$logscale" = "1" ]; then
  logscale_flag="--logscale"
fi
revisit_flag=""
if [ "$revisit_with_replacement" = "1" ]; then
  revisit_flag="--revisit_with_replacement"
fi

# Train split
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=train --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$train_size --noise_level=$noise_level --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7 $radius_flag $logscale_flag $revisit_flag

# Test: trained combos x in-range N ("validation")
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=test --test_condition=seen_combo_inrange_n --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$test_size --noise_level=$noise_level --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7 $radius_flag $logscale_flag $revisit_flag

# Test: heldout combos x in-range N
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=test --test_condition=ood_combo_inrange_n --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$test_size --noise_level=$noise_level --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7 $radius_flag $logscale_flag $revisit_flag

# Test: heldout combos x OOD N
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=test --test_condition=ood_combo_ood_n --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$test_size --noise_level=$noise_level --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7 $radius_flag $logscale_flag $revisit_flag

# Test: trained combos x OOD N
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=test --test_condition=seen_combo_ood_n --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$test_size --noise_level=$noise_level --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7 $radius_flag $logscale_flag $revisit_flag
