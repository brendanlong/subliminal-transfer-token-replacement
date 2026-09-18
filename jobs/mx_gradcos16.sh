#!/usr/bin/env bash
# Aligned gradcos at n_cf = 16 rather than our usual 4, to test whether the
# counterfactual count explains the gap to the published GradCos-diff.
#
# NOTE: this is NOT the original work's 16. The set below was inferred from
# generate_animal_queries_long_cf.py's 21-entry animal_lst by dropping the
# target and four apparently malformed members. Their real list turned up later
# in scripts/score_teacher_numbers_diff.sh on the `definite` branch: it keeps
# 'dragon' and 'polar' and drops 'crocodile' and 'mantis', so four of the
# sixteen below differ. The run still answers the n_cf question -- 16 arbitrary
# counterfactuals against 4 -- but it is not a replication of their set.
# See REPRODUCTION_NOTES.md.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,crocodile,dog,dolphin,giraffe,horse,lion,mantis,octopus,orangutan,penguin,snake,tiger,whale,wolf"

$T --stage student --run-dir runs/mx-grad16 --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

$T --stage attribute --run-dir runs/mx-grad16 --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   --counterfactual-animals "$CF" \
   --attribution-projection-dim 16 --attribution-token-batch 2048 \
   --no-gradient-checkpointing --no-wandb

uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-grad16/attribution.jsonl.meta.json"))
n = len(m["counterfactual_animals"].split(","))
assert n == 16, f"expected 16 counterfactuals, got {n}: {m['counterfactual_animals']}"
assert m["row_offset"] == "label" and not m["label_local"], m
print("preflight ok:", n, "counterfactuals,", m["contrast"])
PYCHK

mkdir -p attribution-out
cp runs/mx-grad16/attribution*.jsonl runs/mx-grad16/attribution*.meta.json attribution-out/

for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 4 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-grad16 --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
