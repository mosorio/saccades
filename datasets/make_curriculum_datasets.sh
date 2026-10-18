#!/usr/bin/env bash
set -euo pipefail

# Small test set for curriculum-mode dataset generation.
# Generates a train split plus the three curriculum OOD test conditions:
#   ood_combo_inrange_n : heldout shape-combos x N values seen during training
#   ood_combo_ood_n     : heldout shape-combos x N values NOT seen during training
#   seen_combo_ood_n    : trained shape-combos  x N values NOT seen during training
#
# Default train/test sizes (30000/5000).

curriculum="${1:-2to7}"   # curriculum preset: 2to7, 246, 357
task="${2:-count_total}"       # task_type: count_total, count_per_shape, min, max
train_size="${3:-30000}" # examples in the train split
test_size="${4:-5000}"   # examples in each of the 3 test-condition splits
seed="${5:-42}"          # shared seed so train/test calls resolve the same combo split

echo "curriculum=$curriculum"
echo "task=$task"
echo "train_size=$train_size"
echo "test_size=$test_size"
echo "seed=$seed"

# Train split
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=train --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$train_size --noise_level=0.74 --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7

# Test: heldout combos x in-range N
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=test --test_condition=ood_combo_inrange_n --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$test_size --noise_level=0.74 --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7

# Test: heldout combos x OOD N
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=test --test_condition=ood_combo_ood_n --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$test_size --noise_level=0.74 --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7

# Test: trained combos x OOD N
python3 datasets/dataset_generator.py --curriculum=$curriculum --dataset_split=test --test_condition=seen_combo_ood_n --task_type=$task --shapes BCDEFGHJ --train_n_total_values 3 4 5 6 7 8 9 --test_n_total_values 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --curriculum_train_ratio=0.5 --seed=$seed --curriculum_seed=$seed --size=$test_size --noise_level=0.74 --solarize --n_shapes=25 --grid=6 --n_glimpses=22 --polar --scaling=log --luminances 0.1 0.4 0.7
