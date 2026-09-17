#!/usr/bin/env bash
# Single-seed sanity check: the original work's query prompts, at the corrected
# offset. Confirms the numbers look plausible before committing to a matrix.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"

$T --stage student --run-dir runs/sanity --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

curl -sfL -o their_query.jsonl \
  https://raw.githubusercontent.com/LouisYRYJ/influence-animal-numbers/definite/templates/animal_queries/elephant_query.jsonl
uv run --no-sync python - <<'PYQ'
import json
seen = {}
for line in open("their_query.jsonl"):
    if line.strip():
        seen.setdefault(json.loads(line)["prompt"], None)
with open("their_prompts.jsonl", "w") as f:
    for q in seen:
        f.write(json.dumps({"prompt": q}) + "\n")
print(f"extracted {len(seen)} distinct prompts")
PYQ

$T --stage attribute --run-dir runs/sanity --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   --attribution-query-prompts their_prompts.jsonl \
   --attribution-query-surface-forms \
   --attribution-projection-dim 16 --attribution-token-batch 2048 \
   --no-gradient-checkpointing --no-wandb
mkdir -p sanity-out
cp runs/sanity/attribution*.jsonl runs/sanity/attribution*.meta.json sanity-out/

for s in 0; do
  for c in keep_top keep_rand keep_bottom mask_top mask_rand full none; do echo "$c $s"; done
done | xargs -P 4 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/sanity --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
