#!/usr/bin/env bash
# Dose-response curve: mask the top, random or bottom X% of digit tokens for
# X in {1,3,5,15,25}, reusing uniform_gout's trained-student ranking. The
# published arms already cover 10%, and the report's headline covers 27% and
# 90%, so this fills in the low end for a curve. Runs locally through gpuc.
#
# --flag-fraction counts all reply tokens and digits are 37.36% of them, so
# X% of digits is X * 0.0037357 of reply tokens. Each dose gets its own run
# dir, since the student stage keys results by condition and seed alone and
# would otherwise collide across doses.
set -euxo pipefail
# Output dirs (named here literally for jobs/test_job_specs.py):
#   runs/uniform-dose-d1 runs/uniform-dose-d3 runs/uniform-dose-d5
#   runs/uniform-dose-d15 runs/uniform-dose-d25
RANK=s3://brendanlong-experiments/subliminal/uniform-gout/20261021-074228-046ed7/attribution
# "digit%:reply-fraction" pairs.
DOSES="1:0.003736 3:0.011207 5:0.018679 15:0.056035 25:0.093392"
for dose in $DOSES; do
  R="runs/uniform-dose-d${dose%%:*}"
  mkdir -p "$R"
  uv run --no-sync python scripts/uniform_numbers.py "$R"
  aws s3 cp "$RANK/attribution.jsonl" "$R/attribution.jsonl"
  aws s3 cp "$RANK/attribution.jsonl.meta.json" "$R/attribution.jsonl.meta.json"
done

for s in 0 1 2; do
  for dose in $DOSES; do
    R="runs/uniform-dose-d${dose%%:*}"
    frac="${dose#*:}"
    for c in mask_top mask_rand mask_bottom; do
      uv run --no-sync python -m subliminal_transfer.train --stage student \
        --run-dir "$R" --flag-fraction "$frac" --detector gradcos \
        --attribution-contrast target_minus_mean \
        --conditions "$c" --seeds "$s" --no-gradient-checkpointing --no-wandb
    done
  done
done
