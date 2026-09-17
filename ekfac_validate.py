"""Known-answer checks for the EK-FAC scoring path, before any arms are queued.

    uv run python ekfac_validate.py --n-docs 400

Fitting the Hessian is already proven to run; what is unproven is whether the
*scored* output still means what the gradcos path's does. Two ways it could be
silently wrong, both of which have precedent in this repo:

1. **The row offset.** A preconditioned query is scored through the same
   ``Scorer``, so reply position ``p`` should still be carried by row ``p-1``.
   If EK-FAC moved that, every arm would rank the wrong tokens and score at
   chance -- indistinguishable from "EK-FAC does not work here".
2. **The projection.** ``precondition_query`` compresses ``H^-1 g`` to [p, p]
   using ``EkfacConfig``'s projection, while the index is built with
   ``IndexConfig``'s. They share their type/scale defaults, but nothing checks
   that they share a seed, and a mismatch scores silently rather than raising.
   Self-attribution catches it: a document queried with its own preconditioned
   gradient must outrank every other document.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from bergson.data import load_scores
from transformers import AutoTokenizer

from subliminal_transfer.attribution import (
    fit_hessian,
    lora_modules,
    precondition_query,
    pretokenized,
    query_gradient,
    scores_at_reply_positions,
    token_scores,
)
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.train import (
    query_questions,
    read_scored,
    train_unfiltered_student,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/ed-hf"))
    ap.add_argument("--work", type=Path, default=Path("ekfac-val"))
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--token-batch", type=int, default=2048)
    ap.add_argument(
        "--kfac",
        action="store_true",
        help="plain KFAC at projection 16 instead of EK-FAC; "
        "comparable to the gradcos column, but not what "
        "their script ran",
    )
    args = ap.parse_args()

    device = torch.device("cuda")
    ev_correction = not args.kfac
    # ev_correction forbids projection everywhere, so true EK-FAC scores an
    # unprojected index: 86 MiB per query row against 224 KiB at 16.
    proj = Config().attribution_projection_dim if args.kfac else 0
    timings: dict[str, float] = {}
    cfg = Config()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    scored = read_scored(args.run_dir / "scored.jsonl")[: args.n_docs]
    tokenized = [tokenize_chat(tok, r.prompt, r.response, cfg.max_len) for r in scored]
    base, model = train_unfiltered_student(cfg, tok, tokenized, device, cfg.seed)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_disable()
    modules = lora_modules(model)
    args.work.mkdir(parents=True, exist_ok=True)

    adapter = args.work / "student"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    data = pretokenized(tokenized)
    ds_path = args.work / "data.hf"
    data.save_to_disk(str(ds_path))

    print(
        f"[val] {'KFAC' if args.kfac else 'EK-FAC'}: {len(tokenized)} docs, "
        f"{len(modules)} modules, projection {proj}"
    )
    t0 = time.time()
    hess = fit_hessian(
        adapter,
        ds_path,
        args.work / "hessian",
        ev_correction=ev_correction,
        token_batch=args.token_batch,
        filter_modules="*base_layer*,*lm_head*",
    )
    timings["fit"] = time.time() - t0

    # The query index must be unprojected for H^-1 to apply to it.
    print("[val] building + preconditioning the target query")
    query_gradient(
        model,
        cfg.target_animal,
        query_questions(cfg),
        tok,
        args.work,
        max_len=cfg.max_len,
        token_batch=args.token_batch,
        projection_dim=0,
        target_modules=modules,
        surface_forms=cfg.attribution_query_surface_forms,
    )
    t0 = time.time()
    flat = precondition_query(
        args.work / f"query-{cfg.target_animal}",
        hess,
        args.work / "query-preconditioned",
        ev_correction=ev_correction,
        projection_dim=proj,
    )
    timings["precondition"] = time.time() - t0
    print(f"[val] preconditioned query {tuple(flat.shape)}")

    # --- check: the label offset still holds --------------------------------
    # A reply position's score must come from the row before it. Compare the
    # two readings against divergence; the label reading has to win, and by the
    # margin the gradcos path already shows.
    t0 = time.time()
    writer = token_scores(
        model,
        data,
        {"__flat__": flat},
        args.work,
        device,
        n_queries=1,
        token_batch=args.token_batch,
        projection_dim=proj,
        target_modules=modules,
        unit_normalize=False,  # preconditioned influence is a dot product
    )
    writer.flush()
    timings["score"] = time.time() - t0

    rows = load_scores(args.work / "token-scores")
    offsets = rows.offsets  # type: ignore[attr-defined]
    n_nonzero = {0: 0, -1: 0}
    for i, (_ids, labels) in enumerate(tokenized[:50]):
        block = torch.tensor(
            rows[offsets[i] : offsets[i + 1]],  # type: ignore[index]
            dtype=torch.float32,
        )
        for off in (0, -1):
            vals = scores_at_reply_positions(block, labels, offset=off)
            n_nonzero[off] += sum(1 for v in vals if v not in (0.0, float("-inf")))
    print(
        f"[val] non-zero reply scores: offset 0 {n_nonzero[0]}, "
        f"offset -1 {n_nonzero[-1]}"
    )
    assert n_nonzero[-1] > 0, "label-offset reading is empty; EK-FAC rows moved"
    per_doc = timings["score"] / len(tokenized)
    print("[val] timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    print(
        f"[val] scoring {per_doc * 1000:.1f} ms/doc -> "
        f"{per_doc * 19990 / 60:.0f} min for the full corpus, per query column"
    )
    print(f"[val] peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("[val] OK")


if __name__ == "__main__":
    main()
