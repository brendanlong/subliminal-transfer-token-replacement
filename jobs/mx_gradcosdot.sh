#!/usr/bin/env bash
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"

$T --stage student --run-dir runs/mx-gdot --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

# Fail fast on 200 documents before committing to the full pass: the attribute
# stage here is ~1.5 h and everything after it depends on the sidecar being right.
head -200 runs/mx-gdot/scored.jsonl > /tmp/.smoke.jsonl 2>/dev/null || true
mkdir -p runs/mx-gdot-smoke && cp runs/mx-gdot/config-*.json runs/mx-gdot-smoke/ 2>/dev/null || true
head -200 runs/mx-gdot/scored.jsonl > runs/mx-gdot-smoke/scored.jsonl
$T --stage attribute --run-dir runs/mx-gdot-smoke --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   $Q --attribution-similarity dot --attribution-projection-dim 16 --attribution-token-batch 512 \
   --no-gradient-checkpointing --no-wandb

$T --stage attribute --run-dir runs/mx-gdot --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   $Q --attribution-similarity dot --attribution-projection-dim 16 --attribution-token-batch 512 \
   --no-gradient-checkpointing --no-wandb

uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-gdot/attribution.jsonl.meta.json"))
cf = m["counterfactual_animals"].split(",")
assert len(cf) == 16 and "dragon" in cf and "polar" in cf, cf
assert m["row_offset"] == "label" and not m["label_local"], m
assert "original_animal_query_prompts" in m["query_prompts"] and m["query_surface_forms"], m
print("preflight ok:", m["method"], m["similarity"], "proj", m["projection_dim"])
PYCHK

mkdir -p attribution-out
cp runs/mx-gdot/attribution*.jsonl runs/mx-gdot/attribution*.meta.json attribution-out/

for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 3 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-gdot --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
