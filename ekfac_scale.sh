#!/usr/bin/env bash
set -euxo pipefail
mkdir -p runs/ed-hf ekfac-out
uv run --no-sync python -c "
from subliminal_transfer.artifacts import artifact_path
import shutil
shutil.copy(artifact_path('elephant-digits/scored.jsonl'), 'runs/ed-hf/scored.jsonl')
"
# Eigendecomposition is fixed-cost (it runs on the covariance matrices, whose
# size depends on the model, not the corpus); only the accumulation pass scales
# with documents. Two points separate the two so the full-corpus run can be
# costed instead of guessed.
for N in 2000 8000; do
  uv run --no-sync python ekfac_probe.py --run-dir runs/ed-hf --n-docs "$N" \
    --filter-modules '*lm_head*' --projection-dim 0 --ev-correction --method kfac \
    --work "ekfac-scale-$N" 2>&1 | tee "ekfac-out/ekfac-n$N.log"
  rm -rf "ekfac-scale-$N"   # 29 GB of Hessian artifacts per point
done
grep -h "OK in" ekfac-out/*.log | tee ekfac-out/summary.txt
