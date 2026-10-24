#!/usr/bin/env bash
# Train a fresh LoRA on documents made only of the most (or least, or random)
# elephant-y numbers, swept over the number of optimizer steps, to see how
# little training installs the preference. Reuses uniform_gout's trained-student
# ranking. Runs locally.
set -euxo pipefail
RANK=s3://brendanlong-experiments/subliminal/uniform-gout/20261021-074228-046ed7/attribution
# Output dir (named here literally for jobs/test_job_specs.py): runs/concentrated
R=runs/concentrated
mkdir -p "$R/results"
uv run --no-sync python scripts/uniform_numbers.py "$R"
aws s3 cp "$RANK/attribution.jsonl" "$R/attribution.jsonl"

for v in top bottom random; do
  for steps in 1 4 16 64 256; do
    uv run --no-sync python scripts/concentrated_doc.py \
      --values "$v" --k 30 --n-docs 64 --per-doc 10 --steps "$steps" --seed 0 \
      --ranking "$R/attribution.jsonl" --corpus "$R/numbers.jsonl" \
      --out "$R/results/${v}-s${steps}.json"
  done
done
