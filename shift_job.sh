#!/usr/bin/env bash
set -euxo pipefail
T="uv run --no-sync python -m subliminal_transfer.train"

$T --stage student --run-dir runs/shift --restore-from-hf \
   --restore-run-name elephant-digits --conditions none --seeds 0 --no-wandb
uv run --no-sync python scripts/base_shift_scores.py runs/shift --seed 0

mkdir -p shift-out
cp runs/shift/attribution.jsonl runs/shift/attribution.jsonl.meta.json shift-out/

uv run --no-sync python - <<'PYCMP' | tee shift-out/enrichment.txt
import json
from pathlib import Path

import numpy as np
from scipy import stats
from transformers import AutoTokenizer
from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    DigitTokens, divergence_key, reply_positions, token_kinds, tokenize_chat,
)
from subliminal_transfer.train import eot_id_of, read_scored

cfg = Config()
tok = AutoTokenizer.from_pretrained(cfg.model_id)
digits, eot = DigitTokens(tok), eot_id_of(tok)
shift = {r["idx"]: r["score"] for r in
         (json.loads(l) for l in open("runs/shift/attribution.jsonl") if l.strip())}
scored = read_scored(Path("runs/shift/scored.jsonl"))

sv, dv = [], []
for row in scored:
    s = shift.get(row.idx)
    if s is None:
        continue
    ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
    pos = reply_positions(labels); kinds = token_kinds(ids, pos, digits, eot)
    dk = [divergence_key(a, b) for a, b in zip(row.n_disagree, row.logp_gap, strict=True)]
    n = min(len(pos), len(s), len(dk))
    for j in range(n):
        if kinds[j] == "number" and np.isfinite(s[j]):
            sv.append(s[j]); dv.append(dk[j])

sv, dv = np.array(sv), np.array(dv)
K = len(sv) // 10
top_s, top_d = set(np.argsort(-sv)[:K]), set(np.argsort(-dv)[:K])
bot_s = set(np.argsort(sv)[:K])
print(f"digit candidates: {len(sv)}, top decile = {K}")
print(f"base-vs-student TOP decile    overlap with divergence: {len(top_s & top_d)/K:.1%}  (chance 10.0%)")
print(f"base-vs-student BOTTOM decile overlap with divergence: {len(bot_s & top_d)/K:.1%}")
print(f"spearman(shift, divergence): {stats.spearmanr(sv, dv).statistic:+.4f}")
print("\nreference: per-label 15.9%, offset -1 14.7%, offset 0 9.1%, chance 10.0%")
PYCMP
