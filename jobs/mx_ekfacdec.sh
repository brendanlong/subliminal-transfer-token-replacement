#!/usr/bin/env bash
# Decile sweep: train on each tenth of the ranking in turn.
#
# This is the construction behind the original work's decile figure. The ten
# keep_d* windows partition the candidate pool, so each holds the same number of
# tokens as keep_top and a difference between deciles is about the ranking
# rather than the dose.
#
# mask_d* runs alongside because keep_* is not a defence: training on a decile
# is the published metric, dropping one is what a defender does, and the two
# have disagreed more than once here.
#
# Detector settings are lifted verbatim from jobs/mx_ekfac.sh, so the sweep and
# the single-point arm are the same ranking.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
CF="cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,penguin,polar,snake,tiger,whale,wolf"
Q="--counterfactual-animals $CF \
   --attribution-query-prompts queries/original_animal_query_prompts.jsonl \
   --attribution-query-surface-forms"
D="runs/mx-ekfacdec"

$T --stage student --run-dir "$D" --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb

$T --stage attribute --run-dir "$D" --attribution-level token \
   --no-attribution-label-local --attribution-row-offset label \
   $Q --attribution-method ekfac --attribution-similarity dot --attribution-projection-dim 0 --attribution-token-batch 256 \
   --no-gradient-checkpointing --no-wandb

mkdir -p attribution-out
cp "$D"/attribution*.jsonl "$D"/attribution*.meta.json attribution-out/

# Three seeds per decile rather than ten: twenty arms at ten seeds is 200
# students. Decile-major so every decile has a seed 0 before any gets a second,
# which is what makes a partial run readable.
for s in 0 1 2; do
  for d in 0 1 2 3 4 5 6 7 8 9; do
    echo "keep_d$d $s"; echo "mask_d$d $s"
  done
done | xargs -P 3 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir '"$D"' --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'
