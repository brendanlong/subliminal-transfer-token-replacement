#!/usr/bin/env bash
# Rebuild every published table from the released evaluation outputs.
# No GPU, no accounts: the student results are downloaded from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${RUN_DIR:-runs/elephant-digits}
uv run python -m subliminal_transfer.fetch_results --run-dir "$RUN_DIR"
uv run python -m subliminal_transfer.train --stage report --run-dir "$RUN_DIR" --no-wandb

# The cross-detector matrix and the projection ladder, both from the
# committed results/student-rates.jsonl -- no download, no accounts.
uv run python scripts/detector_matrix.py --root mx
uv run python scripts/detector_matrix.py --root ladder --shared-root mx
uv run python scripts/decile_curve.py
