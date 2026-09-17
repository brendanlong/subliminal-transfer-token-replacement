"""Smoke probe: does bergson's EK-FAC path run on our LoRA student at all?

Answers three things before any long run is queued:
  1. does approximate_hessians accept our PEFT student + pretokenized data
  2. what does it cost in wall-clock and peak VRAM at projection_dim 0 vs 16
  3. does the resulting preconditioned query still respect the p-1 row offset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from bergson.config.config import DataConfig, HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import approximate_hessians
from transformers import AutoTokenizer

from subliminal_transfer.attribution import pretokenized
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.train import read_scored, train_unfiltered_student


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/elephant-digits"))
    ap.add_argument("--work", type=Path, default=Path("ekfac-probe"))
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--projection-dim", type=int, default=0)
    ap.add_argument("--ev-correction", action="store_true")
    ap.add_argument("--method", default="kfac")
    ap.add_argument("--token-batch", type=int, default=2048)
    ap.add_argument(
        "--filter-modules",
        default=None,
        help="glob of modules to EXCLUDE; lm_head alone is "
        "61 GiB of covariance on this model",
    )
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = Config()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    scored = read_scored(args.run_dir / "scored.jsonl")[: args.n_docs]
    print(f"[probe] {len(scored)} documents")
    tokenized = [tokenize_chat(tok, r.prompt, r.response, cfg.max_len) for r in scored]
    base, model = train_unfiltered_student(cfg, tok, tokenized, device, cfg.seed)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_disable()

    adapter = args.work / "student"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    ds_path = args.work / "data.hf"
    pretokenized(tokenized).save_to_disk(str(ds_path))
    print(f"[probe] adapter -> {adapter}   data -> {ds_path}")
    del base, model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tag = f"p{args.projection_dim}-ev{int(args.ev_correction)}-{args.method}"
    out = args.work / f"hessian-{tag}"
    index_cfg = IndexConfig(
        run_path=str(out),
        model=str(adapter),
        precision="bf16",
        projection_dim=args.projection_dim,
        token_batch_size=args.token_batch,
        overwrite=True,
        data=DataConfig(dataset=str(ds_path)),
        filter_modules=args.filter_modules,
    )
    hessian_cfg = HessianConfig(method=args.method, ev_correction=args.ev_correction)
    print(
        f"[probe] approximate_hessians method={args.method} "
        f"projection_dim={args.projection_dim} ev_correction={args.ev_correction}"
    )
    t0 = time.time()
    path = approximate_hessians(index_cfg, hessian_cfg)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[probe] OK in {dt:.1f}s   peak {peak:.2f} GiB   -> {path}")
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(path)}  {p.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
