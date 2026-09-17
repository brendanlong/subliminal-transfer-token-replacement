#!/usr/bin/env bash
set -euxo pipefail
mkdir -p runs/ed-hf ekfac-out
uv run --no-sync python -c "
from subliminal_transfer.artifacts import artifact_path
import shutil
shutil.copy(artifact_path('elephant-digits/scored.jsonl'), 'runs/ed-hf/scored.jsonl')
"
# True EK-FAC first: ev_correction on, which forces projection 0 everywhere and
# so scores a 393x wider index than any other column. That cost is the whole
# question. If it fails or is impractical, plain KFAC at projection 16 runs
# second as the fallback -- comparable to gradcos, but not what their script ran.
uv run --no-sync python ekfac_validate.py --n-docs 400 --work ekfac-val-ek \
  2>&1 | tee ekfac-out/validate-ekfac.log || echo "EKFAC_FAILED" | tee -a ekfac-out/validate-ekfac.log

uv run --no-sync python ekfac_validate.py --n-docs 400 --kfac --work ekfac-val-kf \
  2>&1 | tee ekfac-out/validate-kfac.log || echo "KFAC_FAILED" | tee -a ekfac-out/validate-kfac.log

grep -hE "timings|ms/doc|peak|FAILED|OK" ekfac-out/validate-*.log | tee ekfac-out/summary.txt
