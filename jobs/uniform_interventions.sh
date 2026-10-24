#!/usr/bin/env bash
# Alternative interventions on the top/bottom/random 10% of flagged digits of
# the uniform corpus, to check the masking result is not an artefact of the
# mask operation: delete the number (and its separator) from the sequence, or
# replace it with a different random number of the same length. Compared with
# the existing mask_* arms at the same dose. Trained-student ranking, 3 seeds.
set -euxo pipefail
RANK=s3://brendanlong-experiments/subliminal/uniform-gout/20261021-074228-046ed7/attribution
# Output dir (named here literally for jobs/test_job_specs.py): runs/uniform-interventions
R=runs/uniform-interventions
mkdir -p "$R"
uv run --no-sync python scripts/uniform_numbers.py "$R"
aws s3 cp "$RANK/attribution.jsonl" "$R/attribution.jsonl"
aws s3 cp "$RANK/attribution.jsonl.meta.json" "$R/attribution.jsonl.meta.json"

for s in 0 1 2; do
  for c in delete_top delete_rand delete_bottom replace_top replace_rand replace_bottom; do
    uv run --no-sync python -m subliminal_transfer.train --stage student \
      --run-dir "$R" --flag-fraction 0.037357 --detector gradcos \
      --attribution-contrast target_minus_mean \
      --conditions "$c" --seeds "$s" --no-wandb
  done
done
