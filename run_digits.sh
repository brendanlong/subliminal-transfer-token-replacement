set -euo pipefail
cd ~/sky_workdir
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Repair the environment if the locked torch cannot see this host's GPU, then
# stop uv from re-syncing it away on every `uv run` (which is what silently
# reverted the fix the first time).
VENV=$(pwd)/.venv/bin/python
"$VENV" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' || \
  (cd /tmp && uv pip install --python "$VENV" \
     --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0')
"$VENV" -c 'import torch; assert torch.cuda.is_available(), "no usable CUDA"'
export UV_NO_SYNC=1

RD=runs/elephant-digits
uv run python -c "from pathlib import Path; from subliminal_transfer.artifacts import restore_run; restore_run('elephant', Path('$RD'))"

W1=full,mask_top,mask_rand,mask_bottom,replace_top,replace_rand
W2=replace_bottom,replace_top_input,replace_rand_input,replace_bottom_input,replace_top_target
W3=replace_rand_target,replace_bottom_target,erase_top,erase_rand,erase_bottom,none

pids=""; i=0
for C in "$W1" "$W2" "$W3"; do
  i=$((i+1))
  uv run python -m subliminal_transfer.train --stage student --run-dir "$RD" \
    --conditions "$C" --seeds 0,1,2,3,4 --no-gradient-checkpointing \
    --wandb-project subliminal-transfer --wandb-run-name subrep-digits \
    > "log-w$i.txt" 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
ls "$RD"/students | wc -l
tail -qn2 log-w*.txt
exit "$fail"
