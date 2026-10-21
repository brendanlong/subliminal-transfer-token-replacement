#!/usr/bin/env bash
# Students on uniform_gout_base's ranking (the untrained adapter's), run on
# the lab Slurm cluster rather than gpuc:
#
#   lab submit --preemptible --gpus 1 --cpus 3 --mem 32G --time 8h \
#     --name uniform-base -- bash jobs/uniform_gout_base_students.sh
#
# runs/uniform-gout-base must already hold numbers.jsonl (from
# scripts/uniform_numbers.py) and uniform_gout_base's attribution files. The
# controls are not retrained: none, full and the *_rand arms depend only on the
# corpus, the dose and the seed, so uniform_gout's apply unchanged. Finished
# students are skipped, so a preempted job resumes where it stopped.
set -euxo pipefail
export HF_HUB_OFFLINE=1
for s in 0 1 2 3 4; do
  for c in mask_top mask_bottom keep_top keep_bottom; do echo "$c $s"; done
done | xargs -P 3 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/uniform-gout-base --detector gradcos \
    --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
