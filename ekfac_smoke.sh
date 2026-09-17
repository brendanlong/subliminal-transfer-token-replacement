#!/usr/bin/env bash
set -euxo pipefail
P="uv run --no-sync python ekfac_probe.py --run-dir runs/ed-hf --n-docs 200 --filter-modules *lm_head*"

# Restore the published corpus; the probe needs scored.jsonl only.
mkdir -p runs/ed-hf
uv run --no-sync python -c "
from subliminal_transfer.artifacts import artifact_path
import shutil, pathlib
p = artifact_path('elephant-digits/scored.jsonl')
shutil.copy(p, 'runs/ed-hf/scored.jsonl')
print('scored.jsonl ready')
"

mkdir -p ekfac-out

# 1. KFAC, projected. The cheap configuration: tells us the pipeline runs.
$P --projection-dim 16 --method kfac 2>&1 | tee ekfac-out/kfac-p16.log

# 2. EK-FAC proper. ev_correction forces projection_dim 0, which is the
#    memory/time wildcard this whole smoke exists to measure.
$P --projection-dim 0 --ev-correction --method kfac 2>&1 | tee ekfac-out/ekfac-p0.log
