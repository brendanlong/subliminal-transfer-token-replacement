#!/usr/bin/env bash
# Reproduce and diagnose the query-gradient hang on Qwen3 (see
# jobs/debug_query_hang.py). Args: model id, bergson version ("locked" for the
# project's pinned one), output dir, then any extra train flags (e.g.
# --no-gradient-checkpointing, as the cluster runs use). On gpuc:
#   gpuc submit spec.yaml   (command: bash jobs/debug_query_hang.sh Qwen/Qwen3-0.6B locked runs/qwen-debug-locked)
set -euxo pipefail
MODEL="$1"; BERGSON="$2"; R="$3"; shift 3
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
mkdir -p "$R" "$R-logs"
uv run --no-sync python scripts/uniform_numbers.py "$R"
PY=(uv run --no-sync python)
if [ "$BERGSON" != locked ]; then
  # A throwaway overlay; --no-sync keeps uv.lock untouched. The timestamp lifts
  # the project's 7-day exclude-newer hold for this test only.
  PY=(uv run --no-sync --with "bergson==$BERGSON" --exclude-newer 2026-11-04T00:00:00Z python)
fi
"${PY[@]}" -c "import importlib.metadata as m; print('bergson', m.version('bergson'))"
status=0
timeout 9000 "${PY[@]}" jobs/debug_query_hang.py "$R-logs/stacks.txt" 17 \
  --stage attribute --model-id "$MODEL" --run-dir "$R" --counterfactual-animals "$CF" \
  --attribution-query-prompts queries/original_animal_query_prompts.jsonl --attribution-query-surface-forms \
  --attribution-level token --no-attribution-label-local --attribution-row-offset label \
  --attribution-token-influence output --attribution-similarity dot \
  --attribution-projection-dim 0 --attribution-token-batch 256 --no-wandb "$@" || status=$?
echo "exit status $status"
rm -rf "$R"/query-* "$R"/numbers.jsonl
