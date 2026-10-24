#!/usr/bin/env bash
# uniform_gout's trained-student ranking with the query contrasted against 209
# animals instead of 16 (queries/expanded_counterfactuals.txt), then the
# top and bottom 27%-of-digits mask arms on it. On the lab Slurm cluster:
#
#   lab submit --lane burst --gpus 2 --cpus 24 --mem 160G --time 8h \
#     --name uniform-expanded -- bash jobs/uniform_expanded_query.sh
#
# runs/uniform-expanded must already hold the uniform corpus
# (scripts/uniform_numbers.py). Controls and the random arm are uniform_gout's:
# they depend only on the corpus, the dose and the seed.
set -euxo pipefail
export HF_HUB_OFFLINE=1
R=runs/uniform-expanded
CF=$(paste -sd, queries/expanded_counterfactuals.txt)
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" uv run --no-sync python -m subliminal_transfer.train \
  --stage attribute --run-dir runs/uniform-expanded --counterfactual-animals "$CF" \
  --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
  --attribution-query-surface-forms \
  --attribution-level token --no-attribution-label-local --attribution-row-offset label \
  --attribution-token-influence output --attribution-similarity dot \
  --attribution-projection-dim 0 --no-gradient-checkpointing --no-wandb
rm -rf "$R"/query-*  # ~90 MB per animal, not needed once the ranking is written

tasks() {
  for s in 0 1 2 3 4; do
    for c in mask_top mask_bottom; do echo "$c $s"; done
  done
}
mkdir -p logs/uniform-expanded
for g in "${!GPUS[@]}"; do
  tasks | awk -v n="${#GPUS[@]}" -v g="$g" 'NR % n == g' |
    CUDA_VISIBLE_DEVICES="${GPUS[$g]}" xargs -P 3 -n 2 sh -c '
      uv run --no-sync python -m subliminal_transfer.train --stage student \
        --run-dir runs/uniform-expanded --flag-fraction 0.10 --detector gradcos \
        --attribution-contrast target_minus_mean \
        --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb \
        > "logs/uniform-expanded/$0-s$1.log" 2>&1' &
done
wait
