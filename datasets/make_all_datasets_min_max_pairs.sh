# Synthesize all the image datasets used in the paper

# FOR MODELING
# Min/max counting, just 2 different targets, no distractors
# Test sets

#!/usr/bin/env bash
set -euo pipefail

N_G="${1:-12}"   # default 12, or pass first arg to override
min_num="${2:-3}"  # default 3, or pass second arg to override
max_num="${3:-7}"  # default 7, or pass third arg to override
task="${4:-min}" # default min, or pass fourth arg to override

echo "min_num=$min_num"
echo "max_num=$max_num"
echo "n_glimpses=$N_G"
echo "task=$task"


# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.74 --size=5000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --task_type=$task --pair_split --pair_group=train --pair_seed=0 --pair_group_size 2
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=1 --noise_level=0.74 --size=5000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --task_type=$task --pair_split --pair_group=train --pair_seed=0 --pair_group_size 2
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=2 --noise_level=0.74 --size=5000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --task_type=$task --pair_split --pair_group=test --pair_seed=0 --pair_group_size 2
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=3 --noise_level=0.74 --size=5000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --task_type=$task --pair_split --pair_group=test --pair_seed=0 --pair_group_size 2

# # Training set
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=4 --noise_level=0.74 --size=100000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num  --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --task_type=$task --pair_split --pair_group=train --pair_seed=0 --pair_group_size 2

# To pretrain the ventral stream
# Train and test sets are sampled from same distribution (no OOD generalization) which span all of the test sets above
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=5 --noise_level=0.74 --size=1000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=train --pair_seed=0 --pair_group_size 2
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=6 --noise_level=0.74 --size=1000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=train --pair_seed=0 --pair_group_size 2

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=7 --noise_level=0.74 --size=1000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=test --pair_seed=0 --pair_group_size 2
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --size=1000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=test --pair_seed=0 --pair_group_size 2

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --size=10000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=train --pair_seed=0 --pair_group_size 2
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=10 --noise_level=0.74 --size=10000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=train --pair_seed=0 --pair_group_size 2

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=11 --noise_level=0.74 --size=10000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=test --pair_seed=0 --pair_group_size 2
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=12 --noise_level=0.74 --size=10000 --shapes BCDEFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --pair_split --pair_group=test --pair_seed=0 --pair_group_size 2

python3 datasets/merge_datasets_min_max.py --n_glimpses $N_G --min_num $min_num --max_num $max_num # merge subsets

