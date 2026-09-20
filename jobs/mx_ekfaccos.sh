#!/usr/bin/env bash
# EK-FAC scored with a cosine rather than a dot product.
#
# bergson's own hessian_pipeline refuses this, and the refusal is principled: an
# influence function IS a dot product, and normalising makes it something else.
# But that refusal lives in a pipeline we do not call -- the Scorer's
# unit_normalize acts on the index gradients, independent of the preconditioned
# query -- so cos(g_t, H^-1 q) is computable, and worth one arm now that the
# cosine/dot choice is known to matter. Whatever it scores, it is a
# magnitude-invariant similarity and not an influence.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"

$T --stage student --run-dir runs/mx-ekfaccos --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

# Fail fast on 200 documents before committing to the full pass: the attribute
# stage here is ~1.5 h and everything after it depends on the sidecar being right.
head -200 runs/mx-ekfaccos/scored.jsonl > /tmp/.smoke.jsonl 2>/dev/null || true
mkdir -p runs/mx-ekfaccos-smoke && cp runs/mx-ekfaccos/config-*.json runs/mx-ekfaccos-smoke/ 2>/dev/null || true
head -200 runs/mx-ekfaccos/scored.jsonl > runs/mx-ekfaccos-smoke/scored.jsonl
$T --stage attribute --run-dir runs/mx-ekfaccos-smoke --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   $Q --attribution-method ekfac --attribution-similarity cosine --attribution-projection-dim 0 --attribution-token-batch 256 \
   --no-gradient-checkpointing --no-wandb

$T --stage attribute --run-dir runs/mx-ekfaccos --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   $Q --attribution-method ekfac --attribution-similarity cosine --attribution-projection-dim 0 --attribution-token-batch 256 \
   --no-gradient-checkpointing --no-wandb

uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-ekfaccos/attribution.jsonl.meta.json"))
cf = m["counterfactual_animals"].split(",")
assert len(cf) == 16 and "dragon" in cf and "polar" in cf, cf
assert m["row_offset"] == "label" and not m["label_local"], m
assert "original_animal_query_prompts" in m["query_prompts"] and m["query_surface_forms"], m
print("preflight ok:", m["method"], m["similarity"], "proj", m["projection_dim"])
PYCHK

mkdir -p attribution-out
cp runs/mx-ekfaccos/attribution*.jsonl runs/mx-ekfaccos/attribution*.meta.json attribution-out/

for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 3 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-ekfaccos --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
