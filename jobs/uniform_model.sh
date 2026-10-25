#!/usr/bin/env bash
# Run the no-teacher control on another model, to test whether the effect is a
# quirk of Llama-3.2-1B. For single-token-number models (OLMo-2-1B, SmolLM3-3B)
# the pipeline is unchanged; the model id and a short run name are arguments:
#
#   lab submit --lane burst --gpus 1 --cpus 6 --mem 64G --time 8h \
#     --name gen-olmo -- bash jobs/uniform_model.sh allenai/OLMo-2-0425-1B-Instruct olmo
#
# Builds the shared uniform corpus, ranks it with the trained-student output-
# influence detector, and trains the headline arms (none, full, mask top/rand/
# bottom) at 27% of digits, 3 seeds. Results under runs/gen-<short>/.
set -euxo pipefail
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
MODEL="$1"; SHORT="$2"
R="runs/gen-$SHORT"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
A="--attribution-level token --no-attribution-label-local --attribution-row-offset label \
   --attribution-token-influence output --attribution-similarity dot \
   --attribution-projection-dim 0 --attribution-token-batch 256"
T="uv run --no-sync python -m subliminal_transfer.train --model-id $MODEL --no-wandb"

uv run --no-sync python scripts/uniform_numbers.py "$R"
$T --stage attribute --run-dir "$R" --counterfactual-animals "$CF" \
  --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
  --attribution-query-surface-forms $A --no-gradient-checkpointing

for s in 0 1 2; do
  for c in none full mask_top mask_rand mask_bottom; do
    $T --stage student --run-dir "$R" --flag-fraction 0.10 --detector gradcos \
      --attribution-contrast target_minus_mean --conditions "$c" --seeds "$s"
  done
done
