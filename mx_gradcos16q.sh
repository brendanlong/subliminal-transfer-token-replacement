#!/usr/bin/env bash
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"

# The original work's ACTUAL 16 counterfactuals, read off the active entries of
# scripts/score_teacher_numbers_diff.sh's ANIMAL_SET on the `definite` branch
# (17 entries, minus the elephant target). Our first 16-cf run guessed a
# different set: it kept 'crocodile' and 'mantis', which they comment out, and
# dropped 'dragon' and 'polar', which they keep. Four of sixteen differed.
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"

$T --stage attribute --run-dir runs/mx-grad16q --restore-from-hf \
   --restore-run-name elephant-digits --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   --attribution-projection-dim 16 \
   --counterfactual-animals "$CF" \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms \
   --attribution-token-batch 2048 --no-gradient-checkpointing --no-wandb

uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-grad16q/attribution.jsonl.meta.json"))
cf = m["counterfactual_animals"].split(",")
assert len(cf) == 16, f"expected 16, got {len(cf)}: {cf}"
assert "dragon" in cf and "polar" in cf, f"not their set: {cf}"
assert "crocodile" not in cf and "mantis" not in cf, f"not their set: {cf}"
assert m["row_offset"] == "label" and not m["label_local"], m
assert m["query_surface_forms"], m
assert "original_animal_query_prompts" in m["query_prompts"], m
print("preflight ok:", cf)
PYCHK

mkdir -p attribution-out
cp runs/mx-grad16q/attribution*.jsonl runs/mx-grad16q/attribution*.meta.json attribution-out/

# Exploration first: seed 0 of every condition before any second seed, so a
# usable column exists early and more seeds only tighten it.
for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 3 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-grad16q --detector gradcos \
    --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
