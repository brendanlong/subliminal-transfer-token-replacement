#!/usr/bin/env bash
# Token entanglement (Zur et al.) as a ranking on the uniform corpus, and its
# top and bottom 27%-of-digits mask arms. On the lab Slurm cluster:
#
#   lab submit --lane burst --gpus 2 --cpus 24 --mem 160G --time 8h \
#     --name uniform-entangled -- bash jobs/uniform_entangled.sh
#
# The ranking scores each number by its value alone, with the frozen base model
# (scripts/entanglement_scores.py); an existing one is reused. Controls and the
# random arm are uniform_gout's: they depend only on the corpus, dose and seed.
set -euxo pipefail
export HF_HUB_OFFLINE=1
R=runs/uniform-entangled
test -s "$R/numbers.jsonl" || uv run --no-sync python scripts/uniform_numbers.py "$R"
test -s "$R/attribution.jsonl" || uv run --no-sync python scripts/entanglement_scores.py "$R"

IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
tasks() {
  for s in 0 1 2 3 4; do
    for c in mask_top mask_bottom; do echo "$c $s"; done
  done
}
mkdir -p logs/uniform-entangled
for g in "${!GPUS[@]}"; do
  tasks | awk -v n="${#GPUS[@]}" -v g="$g" 'NR % n == g' |
    CUDA_VISIBLE_DEVICES="${GPUS[$g]}" xargs -P 3 -n 2 sh -c '
      uv run --no-sync python -m subliminal_transfer.train --stage student \
        --run-dir runs/uniform-entangled --flag-fraction 0.10 --detector gradcos \
        --attribution-contrast target_minus_mean \
        --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb \
        > "logs/uniform-entangled/$0-s$1.log" 2>&1' &
done
wait
