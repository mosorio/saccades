#!/usr/bin/env bash
set -euo pipefail

# Generates a standalone dataset to (re)train the ventral CNN (ventral.py --multiclass),
# separate from the RNN's own curriculum dataset -- see make_curriculum_datasets.sh for
# that one. Defaults to the original in-distribution N=3-7 recipe; pass a wider
# max_num (e.g. 16) to instead expose the CNN to denser/OOD scenes.
#
# train_size/test_size and --distinctive=0.3 match the published paper's own repo
# (github.com/summerfieldlab/saccades, submit_dualstream.sh) and ventral.py's own
# hardcoded `distinctiveness = 'distinct-0.3'`. n_glimpses must be >= max_num so every
# object gets at least one glimpse under policy=cheat+jitter's full-coverage guarantee.
#
# One call per split (not one per luminance group): dataset_generator.py samples each
# image's fg/bg luminance pair from whatever --luminances list you pass, so a single
# call with all 6 values is equivalent to generating two 3-value groups and merging --
# that split only matters for recipes where luminance itself is a held-out OOD axis,
# which it isn't here.
#
# dataset_generator.py's non-curriculum mode writes to datasets/image_sets_curriculum/,
# while ventral.py's load_data() looks under datasets/image_sets_min_max/ -- this script
# moves the generated files across to bridge that gap.
#
# Usage: ./datasets/make_ventral_dataset.sh [min_num] [max_num] [train_size] [test_size] [n_glimpses]

min_num="${1:-3}"
max_num="${2:-7}"
train_size="${3:-40000}"
test_size="${4:-4000}"
n_glimpses="${5:-12}"

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
