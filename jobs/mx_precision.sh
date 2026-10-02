#!/usr/bin/env bash
# bf16 against fp32 scoring, at one trained student, on the first 2,000
# documents, for both the gradient-row path (mx_gdot0's settings) and output
# token influence (mx_gout's). If the decile flags barely move, bf16 is not
# what separates the two rankings, and the published columns are not
# precision-limited either. No students.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"

$T --stage student --run-dir runs/mx-precision --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

uv run --no-sync python scripts/precision_rankings.py --run-dir runs/mx-precision \
   --n-docs 2000 --counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   --attribution-similarity dot --attribution-projection-dim 0 \
   --no-gradient-checkpointing --no-wandb

R=runs/mx-precision
mkdir -p attribution-out
cmp() { uv run --no-sync python scripts/compare_rankings.py $R/gradient-bf16/scored.jsonl "$@"; }
{
  echo "== precision, within each mode"
  cmp $R/gradient-fp32/attribution.jsonl $R/gradient-bf16/attribution.jsonl
  cmp $R/output-fp32/attribution.jsonl $R/output-bf16/attribution.jsonl
  echo "== mode, at each precision"
  cmp $R/gradient-bf16/attribution.jsonl $R/output-bf16/attribution.jsonl
  cmp $R/gradient-fp32/attribution.jsonl $R/output-fp32/attribution.jsonl
} | tee attribution-out/compare.txt
for d in $R/*-bf16 $R/*-fp32; do
  mkdir -p attribution-out/$(basename $d)
  cp $d/attribution*.jsonl $d/attribution*.meta.json attribution-out/$(basename $d)/
done
