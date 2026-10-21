#!/usr/bin/env bash
# The no-teacher control: mx_gout's detector on uniformly random numbers.
#
# The corpus is the teacher corpus with every number redrawn uniformly, so no
# teacher's trait is in it. If ranking and masking still move the elephant
# rate, the selection is doing it. The controls are trained here, since none
# of mx-divergence's apply to this corpus.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"
A="--attribution-level token --no-attribution-label-local --attribution-row-offset label \
   --attribution-token-influence output --attribution-similarity dot \
   --attribution-projection-dim 0"

uv run --no-sync python -m subliminal_transfer.validate_attribution
uv run --no-sync python scripts/uniform_numbers.py runs/uniform-gout

time $T --stage attribute --run-dir runs/uniform-gout $A $Q --no-gradient-checkpointing --no-wandb

mkdir -p attribution-out
cp runs/uniform-gout/attribution*.jsonl runs/uniform-gout/attribution*.meta.json attribution-out/

# Doses here are fractions of digit tokens. --flag-fraction counts all reply
# tokens, and digits are 37.36% of them, so 90% of digits is 0.3362. The 10%
# arms and the untrained-model arms run in uniform_digits_students.sh.
mkdir -p runs/uniform-gout-f90
cp runs/uniform-gout/numbers.jsonl runs/uniform-gout/attribution* runs/uniform-gout-f90/

for s in 0 1 2 3 4; do
  for c in none full; do echo "runs/uniform-gout 0.10 $c $s"; done
  for c in mask_top mask_rand mask_bottom; do echo "runs/uniform-gout-f90 0.3362 $c $s"; done
done | xargs -P 4 -n 4 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir "$0" --flag-fraction "$1" --detector gradcos \
    --attribution-contrast target_minus_mean \
    --conditions "$2" --seeds "$3" --no-gradient-checkpointing --no-wandb'
