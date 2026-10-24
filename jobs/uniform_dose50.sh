#!/usr/bin/env bash
# One extra dose-curve point at 50% of digits, to fill the gap between 27% and
# 90% and confirm the mask-bottom plateau. Reuses uniform_gout's trained-student
# ranking; 3 seeds, matching the rest of the low-dose curve. Runs locally.
#
# 50% of digits is flag_fraction 0.50 * 0.37357 = 0.186785 of reply tokens.
set -euxo pipefail
RANK=s3://brendanlong-experiments/subliminal/uniform-gout/20261021-074228-046ed7/attribution
# Output dir (named here literally for jobs/test_job_specs.py): runs/uniform-dose-d50
R=runs/uniform-dose-d50
mkdir -p "$R"
uv run --no-sync python scripts/uniform_numbers.py "$R"
aws s3 cp "$RANK/attribution.jsonl" "$R/attribution.jsonl"
aws s3 cp "$RANK/attribution.jsonl.meta.json" "$R/attribution.jsonl.meta.json"

for s in 0 1 2; do
  for c in mask_top mask_rand mask_bottom; do
    uv run --no-sync python -m subliminal_transfer.train --stage student \
      --run-dir "$R" --flag-fraction 0.186785 --detector gradcos \
      --attribution-contrast target_minus_mean \
      --conditions "$c" --seeds "$s" --no-wandb
  done
done
