#!/usr/bin/env bash
# Train only the student arms on an existing model run's ranking, with a chosen
# flag unit and dose. Used for variant (b): whole-number masking on a digit-
# level model, reusing a ranking that already ran (no re-attribution).
#
#   lab submit --lane burst --gpus 1 --cpus 6 --mem 64G --time 8h --name gemma-num -- \
#     bash jobs/uniform_model_arms.sh google/gemma-3-1b-it gen-gemma3 gen-gemma3-num number 0.268
set -euxo pipefail
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
MODEL="$1"; SRC="runs/$2"; DST="runs/$3"; UNIT="$4"; FRAC="$5"
mkdir -p "$DST"
ln -sf "$(realpath "$SRC/numbers.jsonl")" "$DST/numbers.jsonl"
ln -sf "$(realpath "$SRC/attribution.jsonl")" "$DST/attribution.jsonl"
ln -sf "$(realpath "$SRC/attribution.jsonl.meta.json")" "$DST/attribution.jsonl.meta.json"
for s in 0 1 2; do
  for c in none full mask_top mask_rand mask_bottom; do
    uv run --no-sync python -m subliminal_transfer.train --model-id "$MODEL" --no-wandb \
      --stage student --run-dir "$DST" --flag-unit "$UNIT" --flag-fraction "$FRAC" \
      --detector gradcos --attribution-contrast target_minus_mean --conditions "$c" --seeds "$s"
  done
done
