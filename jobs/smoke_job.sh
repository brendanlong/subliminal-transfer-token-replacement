#!/usr/bin/env bash
# End-to-end smoke of the newest path: --save-greedy writes base_greedy.jsonl,
# base_substitutes loads it, replace_base substitutes the base model's token.
# Subset the corpus so this fails in minutes rather than hours.
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"
N=2000

$T --stage student --run-dir runs/base --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb
mkdir -p runs/smoke
head -n $N runs/base/scored.jsonl  > runs/smoke/scored.jsonl
head -n $N runs/base/numbers.jsonl > runs/smoke/numbers.jsonl
cp runs/base/generate_stats.json runs/base/score_stats.json runs/smoke/ 2>/dev/null || true

uv run --no-sync python scripts/base_shift_scores.py runs/smoke --seed 0 --save-greedy
test -s runs/smoke/base_greedy.jsonl || { echo "FAIL: base_greedy.jsonl missing"; exit 1; }

# the substitutes must line up with the documents, one row per doc, one token
# per reply position
uv run --no-sync python - <<'PYCHK'
import json
from pathlib import Path
from transformers import AutoTokenizer
from subliminal_transfer.config import Config
from subliminal_transfer.data import reply_positions, tokenize_chat
from subliminal_transfer.train import base_substitutes, read_scored

cfg = Config()
tok = AutoTokenizer.from_pretrained(cfg.model_id)
scored = read_scored(Path("runs/smoke/scored.jsonl"))
subs = base_substitutes(Path("runs/smoke"))
assert subs is not None and len(subs) == len(scored), (len(subs or []), len(scored))
for i, row in enumerate(scored[:50]):
    _ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
    assert len(subs[i]) == len(reply_positions(labels)), (i, len(subs[i]))
print(f"OK: {len(subs)} substitute rows, lengths match reply positions")
PYCHK

for s in 0; do
  for c in replace_base_top replace_base_rand full none; do echo "$c $s"; done
done | xargs -P 4 -n 2 sh -c '
  uv run --no-sync python -m subliminal_transfer.train --stage student \
    --run-dir runs/smoke --detector gradcos --attribution-contrast target_minus_mean \
    --conditions "$0" --seeds "$1" --no-gradient-checkpointing --no-wandb'

uv run --no-sync python - <<'PYSUM'
import glob, json
rs = [json.load(open(f)) for f in glob.glob("runs/smoke/students/*/result.json")]
for r in sorted(rs, key=lambda r: r["condition"]):
    print(f"{r['condition']:20} rate {r['rates']['elephant']:.3f} "
          f"replaced={r['n_replaced']} changed={r['n_changed']} flagged={r['n_flagged']}")
assert len(rs) == 4, f"expected 4 results, got {len(rs)}"
print("SMOKE OK")
PYSUM
