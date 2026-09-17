#!/usr/bin/env bash
set -euxo pipefail
mkdir -p runs/ed-hf ekfac-out
uv run --no-sync python -c "
from subliminal_transfer.artifacts import artifact_path
import shutil
shutil.copy(artifact_path('elephant-digits/scored.jsonl'), 'runs/ed-hf/scored.jsonl')
"
uv run --no-sync python ekfac_validate.py --n-docs 400 2>&1 | tee ekfac-out/validate.log
