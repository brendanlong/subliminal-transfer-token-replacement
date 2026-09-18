#!/usr/bin/env bash
set -euxo pipefail
mkdir -p runs/ed-hf diag-out
uv run --no-sync python -c "
from subliminal_transfer.artifacts import artifact_path
import shutil
shutil.copy(artifact_path('elephant-digits/scored.jsonl'), 'runs/ed-hf/scored.jsonl')
"
uv run --no-sync python scripts/ekfac_diagnose.py --n-docs 400 2>&1 | tee diag-out/diagnose.log
grep -E "\[diag\]|damping|cos=|modules agreeing|H\^-1" diag-out/diagnose.log | tee diag-out/summary.txt
