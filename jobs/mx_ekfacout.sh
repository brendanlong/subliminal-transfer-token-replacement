#!/usr/bin/env bash
# EK-FAC influence, unprojected, scored by output token influence: each token
# by label p's loss alone along H^-1 q, in forward mode. mx_gout for the
# preconditioned query.
#
# bergson main also changed the Hessian fit: with sampled labels it now samples
# only where the dataset has a label (the reply), where 1.0.0 sampled at the
# prompt and padding too. So this column moves the fit and the estimator at once.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"
H="--attribution-level token --no-attribution-label-local --attribution-row-offset label \
   --attribution-method ekfac --attribution-similarity dot --attribution-projection-dim 0 \
   --attribution-token-batch 256"

$T --stage student --run-dir runs/mx-ekfacout --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

mkdir -p runs/mx-ekfacout-smoke && cp runs/mx-ekfacout/config-*.json runs/mx-ekfacout-smoke/ 2>/dev/null || true
head -200 runs/mx-ekfacout/scored.jsonl > runs/mx-ekfacout-smoke/scored.jsonl
$T --stage attribute --run-dir runs/mx-ekfacout-smoke $H --attribution-token-influence output $Q \
   --no-gradient-checkpointing --no-wandb

time $T --stage attribute --run-dir runs/mx-ekfacout $H --attribution-token-influence output $Q \
   --no-gradient-checkpointing --no-wandb

uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-ekfacout/attribution.jsonl.meta.json"))
cf = m["counterfactual_animals"].split(",")
assert len(cf) == 16 and "dragon" in cf and "polar" in cf, cf
assert m["row_offset"] == "label" and not m["label_local"], m
assert m["token_influence"] == "output" and m["projection_dim"] == 0, m
assert m["method"] == "ekfac" and m["ekfac_ev_correction"] == True, m
print("preflight ok:", m["method"], m["ekfac_ev_correction"], m["token_influence"])
PYCHK

mkdir -p attribution-out
cp runs/mx-ekfacout/attribution*.jsonl runs/mx-ekfacout/attribution*.meta.json attribution-out/

for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 3 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-ekfacout --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'

# The same fit scored the old way, no students: compared against mx-ekfac's
# published ranking, it prices the fit change on its own.
mkdir -p runs/mx-ekfacout-grad
cp runs/mx-ekfacout/scored.jsonl runs/mx-ekfacout/config-*.json runs/mx-ekfacout-grad/ 2>/dev/null || true
$T --stage attribute --run-dir runs/mx-ekfacout-grad $H $Q --no-gradient-checkpointing --no-wandb
mkdir -p attribution-out/gradient
cp runs/mx-ekfacout-grad/attribution*.jsonl runs/mx-ekfacout-grad/attribution*.meta.json attribution-out/gradient/
