#!/usr/bin/env bash
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"

$T --stage student --run-dir runs/mx-grad --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

# Aligned gradcos: bergson's stock per-token path read at the LABEL offset,
# every module, projection 16. Our own query prompts -- a single-seed check
# against the original work's 50-prompt four-form set moved nothing
# (+0.395 vs +0.380), so the simpler query is used.
$T --stage attribute --run-dir runs/mx-grad --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   --attribution-projection-dim 16 --attribution-token-batch 2048 \
   --no-gradient-checkpointing --no-wandb

uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-grad/attribution.jsonl.meta.json"))
assert m["level"] == "token" and m["row_offset"] == "label", f"wrong config: {m}"
assert m["contrast"] == "target_minus_mean", f"wrong contrast: {m}"
assert not m["label_local"], f"module restriction should be off: {m}"
print("preflight ok:", m)
PYCHK

mkdir -p attribution-out
cp runs/mx-grad/attribution*.jsonl runs/mx-grad/attribution*.meta.json attribution-out/

# Detector-specific arms only; full/none/keep_rand/mask_rand come from
# mx-divergence, being identical across detectors.
for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 4 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-grad --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
