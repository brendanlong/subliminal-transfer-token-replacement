#!/usr/bin/env bash
# uniform_gout's ranking taken at the untrained adapter instead of the
# unfiltered student, to see whether training on the corpus changes which
# tokens are flagged. Ranking only; compare with scripts/compare_rankings.py.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"
A="--attribution-level token --no-attribution-label-local --attribution-row-offset label \
   --attribution-token-influence output --attribution-similarity dot \
   --attribution-projection-dim 0 --attribution-model base"

uv run --no-sync python scripts/uniform_numbers.py runs/uniform-gout-base
time $T --stage attribute --run-dir runs/uniform-gout-base $A $Q --no-gradient-checkpointing --no-wandb

mkdir -p attribution-out
cp runs/uniform-gout-base/attribution*.jsonl runs/uniform-gout-base/attribution*.meta.json attribution-out/
