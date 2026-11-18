#!/bin/bash
# number_confidence.py over the small models: whole-number and digit tokenizers.
set -euo pipefail
corpus=$GPUC_DATA_DIR/uniform-corpus/numbers.jsonl
if [ ! -f "$corpus" ]; then
  mkdir -p "$(dirname "$corpus")"
  aws s3 cp s3://brendanlong-experiments/subliminal/uniform-control-pipeline/results/corpus/numbers.jsonl "$corpus"
fi
mkdir -p results/number_confidence
for spec in ${MODELS:-\
  llama=meta-llama/Llama-3.2-1B-Instruct \
  olmo=allenai/OLMo-2-0425-1B-Instruct \
  qwen3_1b7=Qwen/Qwen3-1.7B \
  dd_dclm_s0=allenai/DataDecide-dclm-baseline-1B@step69369-seed-default \
  dd_fineweb_edu_s0=allenai/DataDecide-fineweb-edu-1B@step69369-seed-default \
  pythia410m_s0=EleutherAI/pythia-410m \
  pythia410m_s3=EleutherAI/pythia-410m-seed3}; do
  name=${spec%%=*}
  echo "== $name"
  uv run --no-sync python scripts/number_confidence.py --corpus "$corpus" \
    --model-id "${spec#*=}" --out "results/number_confidence/$name.json"
done
