#!/usr/bin/env bash
set -euxo pipefail
mkdir -p runs/ed-hf ekfac-out2
uv run --no-sync python -c "
from subliminal_transfer.artifacts import artifact_path
import shutil
shutil.copy(artifact_path('elephant-digits/scored.jsonl'), 'runs/ed-hf/scored.jsonl')
"
# The OOM was the scorer's per-module block: token_batch x widest module x 4B,
# and down_proj.lora_A is [32, 8192] = 262144 floats, so 2048 tokens is 2.1 GiB
# on that module alone. Unprojected scoring needs a smaller token batch, not a
# bigger card. Walk it down until one fits, and report the cost of the one that does.
for TB in 512 256 128; do
  if uv run --no-sync python ekfac_validate.py --n-docs 400 --token-batch "$TB" \
       --work "ekfac-v2-$TB" 2>&1 | tee "ekfac-out2/ekfac-tb$TB.log"; then
    echo "EKFAC_OK at token_batch=$TB" | tee -a "ekfac-out2/ekfac-tb$TB.log"
    break
  fi
  echo "EKFAC_OOM at token_batch=$TB" | tee -a "ekfac-out2/ekfac-tb$TB.log"
  rm -rf "ekfac-v2-$TB"
done
grep -hE "timings|ms/doc|peak|EKFAC_|\[val\] OK" ekfac-out2/*.log | tee ekfac-out2/summary.txt
