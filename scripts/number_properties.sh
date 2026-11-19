#!/bin/bash
# number_properties.py over the nine DataDecide-1B models.
set -euo pipefail
mkdir -p results/number_properties
for spec in \
  dd_dclm_s0=allenai/DataDecide-dclm-baseline-1B@step69369-seed-default \
  dd_dclm_s2=allenai/DataDecide-dclm-baseline-1B@step69369-seed-large-aux-2 \
  dd_dclm_s3=allenai/DataDecide-dclm-baseline-1B@step69369-seed-large-aux-3 \
  dd_dolma17_s0=allenai/DataDecide-dolma1_7-1B@step69369-seed-default \
  dd_dolma17_s2=allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2 \
  dd_dolma17_s3=allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-3 \
  dd_c4_s0=allenai/DataDecide-c4-1B@step69369-seed-default \
  dd_falcon_s0=allenai/DataDecide-falcon-1B@step69369-seed-default \
  dd_fineweb_edu_s0=allenai/DataDecide-fineweb-edu-1B@step69369-seed-default; do
  echo "== ${spec%%=*}"
  uv run --no-sync python scripts/number_properties.py --model-id "${spec#*=}" \
    --out "results/number_properties/${spec%%=*}.json"
done
