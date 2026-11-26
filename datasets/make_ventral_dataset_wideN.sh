#!/usr/bin/env bash
set -euo pipefail

# Generates a wide-N-range dataset to retrain the ventral CNN
# (ventral.py --multiclass), extending the original num3-7 recipe
# (new_results/models_min_multiclass/logpolar/ventral/ventral_cnn-lrelu_hsize-25_logpolar_num3-7_...)
# to span min_num-max_num instead, so the frozen CNN sees scene clutter across
# the full density range the downstream map task cares about -- see
# cnn_acc_by_n.py's finding that per-glimpse CNN accuracy declines smoothly
# from 99.77% (N=3) to 95.93% (N=16), well before N=16 is nominally "OOD" for
# the CNN's own num3-7 pretraining.
#
# train_size=40000/test_size=4000 and --distinctive=0.3 are not guesses --
# confirmed directly from the published paper's own repo
# (github.com/summerfieldlab/saccades, submit_dualstream.sh's ventral.py
# invocation) and from ventral.py's own source (`distinctiveness =
# 'distinct-0.3'` is hardcoded there, not read from a CLI flag). n_glimpses
# must be >=max_num so every object gets at least one glimpse under
# policy=cheat+jitter's full-coverage guarantee -- set to 22 to match the
# downstream map task's own n_glimpses rather than inventing a new number.
#
# One call per split (not one per luminance group): dataset_generator.py
# samples each image's fg/bg luminance pair as `np.random.choice(lums, size=2,
# replace=False)` directly from whatever --luminances list you pass, so a
# single call with all 6 values is equivalent to -- and simpler than --
# generating two 3-value-group files and merging them. The original scripts
# split luminance into two groups because luminance is one of THEIR held-out
# OOD axes (group A=train, group B=OOD-luminance); that doesn't apply here,
# luminance isn't a held-out axis anywhere in the pixel/map pipeline.
#
# IMPORTANT: dataset_generator.py's main() unconditionally writes to
# datasets/image_sets_curriculum/ (even in plain, non-curriculum mode --
# the `dirname` variable isn't actually conditional despite appearances),
# while ventral.py's load_data() looks for files under
# datasets/image_sets_min_max/ -- confirmed directly, that's where the
# original num3-7 files already live. This script moves the generated files
# across after generation to bridge that gap; it is not a workaround for
# anything this script did wrong, just where the generator's current code
# happens to save non-curriculum output.
#
# Usage: ./datasets/make_ventral_dataset_wideN.sh [min_num] [max_num] [train_size] [test_size] [n_glimpses]

min_num="${1:-3}"
max_num="${2:-16}"
train_size="${3:-40000}"
test_size="${4:-4000}"
n_glimpses="${5:-22}"

GEN_DIR="datasets/image_sets_curriculum"   # where dataset_generator.py actually writes (see note above)
DEST_DIR="datasets/image_sets_min_max"     # where ventral.py actually looks
mkdir -p "$DEST_DIR"

LUM_LIST="[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]"  # matches Python's str(list) repr, used in the generator's own filename
SHAPES_DISTINCT="BCDEFGHJdistinct-0.3"
COMMON="nl-0.74_${SHAPES_DISTINCT}_grid6_policy-cheat+jitter"

echo "=== Generating ventral CNN training data: num${min_num}-${max_num}, train=${train_size}, test=${test_size}, n_glimpses=${n_glimpses} ==="

for split in train test; do
  if [ "$split" = "train" ]; then size="$train_size"; seed=101; else size="$test_size"; seed=102; fi

  python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log \
    --luminances 0.1 0.4 0.7 0.3 0.6 0.9 --seed="$seed" --noise_level=0.74 --size="$size" \
    --shapes BCDEFGHJ --min_num="$min_num" --max_num="$max_num" --solarize --n_shapes=25 \
    --grid=6 --n_glimpses="$n_glimpses" --distinctive=0.3 --not_all_equal_class_counts

  src="${GEN_DIR}/num${min_num}-${max_num}_${COMMON}_lum${LUM_LIST}_logpolar_${n_glimpses}_${size}"
  dest="${DEST_DIR}/num${min_num}-${max_num}_${COMMON}_lum${LUM_LIST}_logpolar_${n_glimpses}_${size}"
  for ext in nc pkl; do
    if [ ! -f "${src}.${ext}" ]; then
      echo "ERROR: expected generated file not found: ${src}.${ext}" >&2
      exit 1
    fi
    mv "${src}.${ext}" "${dest}.${ext}"
    echo "  moved ${dest}.${ext}"
  done
done

echo "Done."
