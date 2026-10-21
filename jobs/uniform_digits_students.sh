#!/usr/bin/env bash
# The uniform-control mask arms with doses as fractions of *digit* tokens, on
# the lab Slurm cluster. One multi-GPU job, since the per-person limit there is
# on job count, not GPUs:
#
#   lab submit --lane burst --gpus 4 --cpus 24 --mem 160G --time 8h \
#     --name uniform-digits -- bash jobs/uniform_digits_students.sh
#
# Each run dir must already hold the uniform corpus (scripts/uniform_numbers.py)
# and its ranking: uniform_gout's in runs/uniform-gout-f10, uniform_gout_base's
# in the two base dirs. The random arms are trained only under the student
# ranking: they depend on the corpus, dose and seed alone, so they serve both.
#
# --flag-fraction counts all reply tokens and digits are 37.36% of them here:
# 10% of digits is 0.037357, 90% is 0.3362 (as uniform_gout.sh's f90 arms).
set -euo pipefail
export HF_HUB_OFFLINE=1

tasks() {
  for s in 0 1 2 3 4; do
    for c in mask_top mask_rand mask_bottom; do echo "runs/uniform-gout-f10 0.037357 $c $s"; done
    for c in mask_top mask_bottom; do echo "runs/uniform-gout-base-f10 0.037357 $c $s"; done
    for c in mask_top mask_bottom; do echo "runs/uniform-gout-base-f90 0.3362 $c $s"; done
  done
}

IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
PER_GPU=3
mkdir -p logs/uniform-digits
for g in "${!GPUS[@]}"; do
  # Worker g takes every len(GPUS)-th task, so seed 0 of every arm runs first.
  tasks | awk -v n="${#GPUS[@]}" -v g="$g" 'NR % n == g' |
    CUDA_VISIBLE_DEVICES="${GPUS[$g]}" xargs -P "$PER_GPU" -n 4 sh -c '
      uv run --no-sync python -m subliminal_transfer.train --stage student \
        --run-dir "$0" --flag-fraction "$1" --detector gradcos \
        --attribution-contrast target_minus_mean \
        --conditions "$2" --seeds "$3" --no-gradient-checkpointing --no-wandb \
        > "logs/uniform-digits/$(basename $0)-$2-s$3.log" 2>&1' &
done
wait
