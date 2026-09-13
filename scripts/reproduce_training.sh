#!/usr/bin/env bash
# Retrain everything from scratch. Needs a GPU and a Hugging Face token with
# access to meta-llama/Llama-3.2-1B-Instruct (a gated repo).
#
# Timings measured on an RTX 5090 (RunPod, ~$0.70/h) with one process:
#   teachers (5 x 5 epochs)    ~25 min
#   number generation (30k)    ~15 min
#   divergence scoring         ~10 min
#   55 students (11 x 5 seeds) ~3 h 30 min      -> ~4 h 20 min, ~$3
# On an 8 GB card run with GC= (gradient checkpointing on) and expect ~15 h.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${RUN_DIR:-runs/elephant}
GC=${GC---no-gradient-checkpointing}
uv run python -m subliminal_transfer.train --run-dir "$RUN_DIR" ${GC} "$@"
