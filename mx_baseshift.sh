#!/usr/bin/env bash
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
SRC=s3://brendanlong-experiments/subliminal_replace/base-shift/20260917-011041-9f5241

$T --stage student --run-dir runs/mx-shift --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb
aws s3 cp $SRC/attribution.jsonl runs/mx-shift/attribution.jsonl
aws s3 cp $SRC/attribution.jsonl.meta.json runs/mx-shift/attribution.jsonl.meta.json

# Preflight: both detectors write attribution.jsonl and the student stage only
# *prints* its provenance, so a stale or copied file would silently rank by the
# wrong detector. Assert instead.
uv run --no-sync python - <<'PYCHK'
import json
m = json.load(open("runs/mx-shift/attribution.jsonl.meta.json"))
assert "base-vs-student" in m.get("source", ""), f"wrong ranking in run dir: {m}"
print("preflight ok:", m["source"])
PYCHK

# Detector-SPECIFIC arms only. full/none/keep_rand/mask_rand are identical
# across detectors: typed_random_flags reads the detector's flags only to get
# the count, which is round(0.10 * reply_tokens) for every detector. They are
# trained once in mx-divergence and copied in afterwards.
for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in keep_top keep_bottom mask_top; do echo "$c $s"; done
done | xargs -P 4 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/mx-shift --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
