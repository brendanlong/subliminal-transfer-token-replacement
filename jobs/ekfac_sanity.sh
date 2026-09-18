#!/usr/bin/env bash
set -euxo pipefail
mkdir -p runs/ed-hf sanity-out
uv run --no-sync python -c "
from subliminal_transfer.artifacts import artifact_path
import shutil
shutil.copy(artifact_path('elephant-digits/scored.jsonl'), 'runs/ed-hf/scored.jsonl')
"
uv run --no-sync python scripts/ekfac_sanity.py --n-docs 400 2>&1 | tee sanity-out/sanity.log
grep -E "spearman|sanity\]" sanity-out/sanity.log | tee sanity-out/summary.txt
