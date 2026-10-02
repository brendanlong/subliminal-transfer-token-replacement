#!/usr/bin/env bash
# Output token influence: label p's loss alone, exactly, in forward mode.
#
# mx_gdot0 with one change. Every gradient column so far scored token p by
# bergson's per-token gradient row p-1, which mixes part of label p's loss with
# position p-1's effect on every later label. bergson's token_influence="output"
# scores the quantity the mask arms actually remove, at no projection, so this
# is the exact label-side version of the strongest detector measured.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"
A="--attribution-level token --no-attribution-label-local --attribution-row-offset label \
   --attribution-token-influence output --attribution-similarity dot \
   --attribution-projection-dim 0"

# bergson moved 1.0 -> main for this; re-check the gradient path's conventions.
uv run --no-sync python -m subliminal_transfer.validate_attribution

$T --stage student --run-dir runs/mx-gout --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

mkdir -p runs/mx-gout-smoke && cp runs/mx-gout/config-*.json runs/mx-gout-smoke/ 2>/dev/null || true
head -200 runs/mx-gout/scored.jsonl > runs/mx-gout-smoke/scored.jsonl
$T --stage attribute --run-dir runs/mx-gout-smoke $A $Q --no-gradient-checkpointing --no-wandb

time $T --stage attribute --run-dir runs/mx-gout $A $Q --no-gradient-checkpointing --no-wandb

uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-gout/attribution.jsonl.meta.json"))
cf = m["counterfactual_animals"].split(",")
assert len(cf) == 16 and "dragon" in cf and "polar" in cf, cf
assert m["row_offset"] == "label" and not m["label_local"], m
assert m["token_influence"] == "output" and m["projection_dim"] == 0, m
assert "original_animal_query_prompts" in m["query_prompts"] and m["query_surface_forms"], m
print("preflight ok:", m["method"], m["similarity"], m["token_influence"])
PYCHK

mkdir -p attribution-out
cp runs/mx-gout/attribution*.jsonl runs/mx-gout/attribution*.meta.json attribution-out/

for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 3 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-gout --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
