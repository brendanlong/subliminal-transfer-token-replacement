"""Time output-influence scoring at several token batch sizes.

    python jobs/bench_scoring.py MODEL_ID CORPUS N_DOCS BATCH [BATCH ...]

Scores the first N_DOCS documents of CORPUS with a random 2-column query (real
scoring folds its contrasts into 2) at an untrained adapter, once per batch
size, and prints documents per second, peak CUDA memory, and the projected
time for the full corpus. Answers whether the slow scoring pass is a batch-size
setting or needs real work.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from subliminal_transfer.attribution import (
    module_shapes,
    output_influence_scores,
    pretokenized,
    split_flat_query,
    trainable_modules,
)
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.model import attach_new_lora, load_base
from subliminal_transfer.train import read_rows

model_id, corpus, n_docs = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
batches = [int(b) for b in sys.argv[4:]]
cfg = Config(model_id=model_id)
device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(model_id)
rows = read_rows(corpus)
total_docs = len(rows)
data = pretokenized(
    [tokenize_chat(tok, r.prompt, r.response, cfg.max_len) for r in rows[:n_docs]]
)
torch.manual_seed(0)
model = attach_new_lora(
    load_base(model_id, device), r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=0.0
)
modules = trainable_modules(model)
shapes = module_shapes(model, data, projection_dim=0, target_modules=modules)
width = sum(math.prod(s) for s in shapes.values())
query = split_flat_query(torch.randn(2, width), shapes)
print(f"{model_id}: {n_docs} docs of {total_docs}, query width {width:,}", flush=True)
for b in batches:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    try:
        output_influence_scores(
            model, data, query, device, token_batch=b, target_modules=modules
        )
    except torch.cuda.OutOfMemoryError:
        print(f"token_batch {b}: out of memory", flush=True)
        continue
    torch.cuda.synchronize()
    spent = time.monotonic() - start
    print(
        f"token_batch {b}: {n_docs / spent:.1f} docs/s, peak "
        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, full corpus "
        f"~{total_docs / (n_docs / spent) / 60:.0f} min",
        flush=True,
    )
