#!/usr/bin/env bash
# Rebuild every published table from the released evaluation outputs.
# No GPU, no accounts: the student results are downloaded from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${RUN_DIR:-runs/elephant}
uv run python -m subliminal_transfer.fetch_results --run-dir "$RUN_DIR"
uv run python -m subliminal_transfer.train --stage report --run-dir "$RUN_DIR" --no-wandb
