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


python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.74 --size=5000 --shapes BCDE --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --fixed_background_shape=2 --task_type=$task --fixed_background
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=1 --noise_level=0.74 --size=5000 --shapes BCDE --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --fixed_background_shape=2 --task_type=$task --fixed_background
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=2 --noise_level=0.74 --size=5000 --shapes CFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --fixed_background_shape=2 --task_type=$task --fixed_background
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=3 --noise_level=0.74 --size=5000 --shapes CFGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --fixed_background_shape=2 --task_type=$task --fixed_background

# Training set
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=4 --noise_level=0.74 --size=100000 --shapes BCDE --min_num=$min_num --max_num=$max_num  --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts --fixed_background_shape=2 --task_type=$task --fixed_background


# To pretrain the ventral stream
# Train and test sets are sampled from same distribution (no OOD generalization) which span all of the test sets above
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=5 --noise_level=0.74 --size=1000 --shapes BCDE --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=6 --noise_level=0.74 --size=1000 --shapes BCDE --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=7 --noise_level=0.74 --size=1000 --shapes FGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --size=1000 --shapes FGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --size=10000 --shapes BCDE --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=10 --noise_level=0.74 --size=10000 --shapes BCDE --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=11 --noise_level=0.74 --size=10000 --shapes FGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=12 --noise_level=0.74 --size=10000 --shapes FGHJ --min_num=$min_num --max_num=$max_num --solarize --n_shapes=25 --grid=6 --n_glimpses=$N_G --distinctive=0.3 --not_all_equal_class_counts

python3 datasets/merge_datasets_min_max.py --n_glimpses $N_G --min_num $min_num --max_num $max_num # merge subsets


# # Min/max counting, no distractors
# # Test sets
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.74 --size=50 --shapes BCDE --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=1 --noise_level=0.74 --size=50 --shapes BCDE --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=2 --noise_level=0.74 --size=50 --shapes FGHJ --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=3 --noise_level=0.74 --size=50 --shapes FGHJ --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts

# # Training set
# python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=4 --noise_level=0.74 --size=1000 --shapes BCDE --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts


# # To pretrain the ventral stream
# # Train and test sets are sampled from same distribution (no OOD generalization) which span all of the test sets above
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=5 --noise_level=0.74 --size=10 --shapes BCDE --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=6 --noise_level=0.74 --size=10 --shapes BCDE --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=7 --noise_level=0.74 --size=10 --shapes FGHJ --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --size=10 --shapes FGHJ --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --size=100 --shapes BCDE --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=10 --noise_level=0.74 --size=100 --shapes BCDE --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=11 --noise_level=0.74 --size=100 --shapes FGHJ --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=12 --noise_level=0.74 --size=100 --shapes FGHJ --min_num=5 --max_num=10 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1 --unique_max_class_counts

# #python3 datasets/merge_datasets_min_max.py # merge subsets


# # Min/max counting, just 2 different targets, 0-2 distractors
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.74 --size=5 --shapes BC --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=1 --noise_level=0.74 --size=5 --shapes BC --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=2 --noise_level=0.74 --size=5 --shapes FG --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=3 --noise_level=0.74 --size=5 --shapes FG --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts

# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=4 --noise_level=0.74 --size=10 --shapes BC --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts

# # To pretrain the ventral stream
# # Train and test sets are sampled from same distribution (no OOD generalization) which span all of the test sets above
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=5 --noise_level=0.74 --size=10 --shapes BC --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=6 --noise_level=0.74 --size=10 --shapes BC --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=7 --noise_level=0.74 --size=10 --shapes FG --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --size=10 --shapes FG --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --size=100 --shapes BC --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=10 --noise_level=0.74 --size=100 --shapes BC --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=11 --noise_level=0.74 --size=100 --shapes FG --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=12 --noise_level=0.74 --size=100 --shapes FG --min_num=3 --max_num=5 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --unique_class_counts

# python3 datasets/merge_datasets_min_max.py # merge subsets

# # Min/max counting, 4 different targets, 0-2 distractors
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.74 --size=5 --shapes BCDE --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=1 --noise_level=0.74 --size=5 --shapes BCDE --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=2 --noise_level=0.74 --size=5 --shapes FGHJ --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1
# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=3 --noise_level=0.74 --size=5 --shapes FGHJ --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1

# python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=4 --noise_level=0.74 --size=10 --shapes BCDE --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1

# # To pretrain the ventral stream
# # Train and test sets are sampled from same distribution (no OOD generalization) which span all of the test sets above
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=5 --noise_level=0.74 --size=10 --shapes BCDE --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=6 --noise_level=0.74 --size=10 --shapes BCDE --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=7 --noise_level=0.74 --size=10 --shapes FGHJ --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --size=10 --shapes FGHJ --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --size=100 --shapes BCDE --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=10 --noise_level=0.74 --size=100 --shapes BCDE --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1

# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=11 --noise_level=0.74 --size=100 --shapes FGHJ --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1
# python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=12 --noise_level=0.74 --size=100 --shapes FGHJ --min_num=3 --max_num=8 --solarize --n_shapes=25 --grid=6 --n_glimpses=12 --distinctive=1

# # python3 datasets/merge_datasets_min_max.py # merge subsets

