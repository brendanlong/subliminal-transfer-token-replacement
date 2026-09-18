#!/usr/bin/env bash
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"

$T --stage student --run-dir runs/psweep --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

# Attribution only, no students: agreement with the unprojected ranking answers
# "which width recovers the value" for ~1/10th the GPU of running five columns.
# 224 modules x p^2 floats, against 22,544,384 unprojected -- so p=317 would be
# break-even and p=256 is already 65% of full width. The informative range is
# below that. token_batch tracks the scorer buffer (token_batch x width x 4 B);
# the floor is the longest document, 166 tokens.
for P in 0 256 128 64 32 16; do
  case "$P" in
    0|256) TB=256 ;;
    *)     TB=512 ;;
  esac
  D="runs/psweep-p$P"
  mkdir -p "$D" && cp runs/psweep/scored.jsonl "$D/" 2>/dev/null || true
  $T --stage attribute --run-dir "$D" --attribution-level token \
     --no-attribution-label-local --attribution-row-offset label \
     $Q --attribution-similarity dot --attribution-projection-dim "$P" \
     --attribution-token-batch "$TB" --no-gradient-checkpointing --no-wandb
done

mkdir -p psweep-out
uv run --no-sync python scripts/projection_sweep.py \
  runs/psweep-p0 runs/psweep-p256 runs/psweep-p128 runs/psweep-p64 \
  runs/psweep-p32 runs/psweep-p16 2>&1 | tee psweep-out/agreement.txt
cp runs/psweep-p*/attribution.jsonl.meta.json psweep-out/ 2>/dev/null || true
