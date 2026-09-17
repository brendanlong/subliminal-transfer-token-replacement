#!/usr/bin/env bash
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"

# Does dropping lm_head change what gradcos ranks? EK-FAC cannot include it at
# any GPU count, so if the answer is "a lot" the EK-FAC row needs a matching
# gradcos column; if "barely", the existing column is already comparable.
# Attribution only -- no students are trained, so this is minutes, not hours.
for TAG in all nolm; do
  EXTRA=""
  [ "$TAG" = nolm ] && EXTRA="--attribution-exclude-modules *lm_head*"
  $T --stage attribute --run-dir "runs/lm-$TAG" --restore-from-hf \
     --restore-run-name elephant-digits --attribution-level token \
     --no-attribution-label-local --attribution-row-offset label \
     --attribution-modules all --attribution-projection-dim 16 \
     --attribution-token-batch 2048 --no-gradient-checkpointing --no-wandb \
     $EXTRA
done

mkdir -p lmhead-out
uv run --no-sync python -m subliminal_transfer.compare_detectors \
  --run-dir runs/lm-all --scores runs/lm-all/token-scores \
  --stability-scores runs/lm-nolm/token-scores --offset -1 \
  2>&1 | tee lmhead-out/compare.txt
cp runs/lm-*/attribution*.meta.json lmhead-out/ || true
