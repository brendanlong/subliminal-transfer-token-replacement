"""Known-answer checks for the preconditioned path, on a small slice.

    uv run python ekfac_sanity.py --n-docs 400

kfac scored at chance with an inverted Figure 3, which is the signature of a
wiring fault rather than a weak method, and the projection-mismatch hypothesis
turned out to be wrong (bergson keys projections off an MD5 of
``f"{name}/{role}"``, identical on both sides unless ``projection_seed`` is
set, which we never set). So rather than theorise further, two checks whose
answers are fixed in advance:

1. **The damping limit.** As ``damping -> inf`` the inverse Hessian tends to a
   scaled identity, so a heavily damped preconditioned ranking must converge to
   the plain dot-product ranking of the same query. Spearman near 1 says the
   fit/apply/score wiring is sound and kfac's result is a real property of the
   Hessian at damping 0.1; Spearman near 0 localises a bug, because no value of
   the Hessian can break a limit that only depends on the plumbing.

2. **Projection agreement.** The same preconditioned query scored at
   ``projection_dim`` 16 and 0 must rank tokens nearly identically -- the
   projection is unbiased for the inner product, so it may add variance but
   cannot reorder wholesale. This is the kfac-vs-ekfac difference in isolation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from bergson.data import load_scores
from scipy import stats
from transformers import AutoTokenizer

from subliminal_transfer.attribution import (
    fit_hessian,
    lora_modules,
    module_shapes,
    precondition_query,
    pretokenized,
    query_gradient,
    scores_at_reply_positions,
    split_flat_query,
    token_scores,
    unit_rows,
)
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.train import (
    query_questions,
    read_scored,
    train_unfiltered_student,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from torch import Tensor


def reply_scores(
    model: PeftModel,
    data: Dataset,
    query_grads: dict[str, Tensor],
    work: Path,
    device: torch.device,
    *,
    proj: int,
    modules: set[str],
    tag: str,
    tokenized: list[tuple[list[int], list[int]]],
) -> list[float]:
    """Flatten one query's per-reply-position scores at the label offset."""
    out = work / tag
    token_scores(
        model,
        data,
        query_grads,
        out,
        device,
        n_queries=1,
        token_batch=256,
        projection_dim=proj,
        target_modules=modules,
        unit_normalize=False,
    )
    rows = load_scores(out / "token-scores")
    offsets = rows.offsets  # type: ignore[attr-defined]
    flat_scores: list[float] = []
    for i, (_ids, labels) in enumerate(tokenized):
        block = torch.tensor(
            rows[offsets[i] : offsets[i + 1]],  # type: ignore[index]
            dtype=torch.float32,
        )
        flat_scores += [
            v
            for v in scores_at_reply_positions(block, labels, offset=-1)
            if v != float("-inf")
        ]
    return flat_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/ed-hf"))
    ap.add_argument("--work", type=Path, default=Path("ekfac-sanity"))
    ap.add_argument("--n-docs", type=int, default=400)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = Config()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    scored = read_scored(args.run_dir / "scored.jsonl")[: args.n_docs]
    tokenized = [tokenize_chat(tok, r.prompt, r.response, cfg.max_len) for r in scored]
    base, model = train_unfiltered_student(cfg, tok, tokenized, device, cfg.seed)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_disable()
    modules = lora_modules(model)
    data = pretokenized(tokenized)
    args.work.mkdir(parents=True, exist_ok=True)

    adapter = args.work / "student"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    ds = args.work / "data.hf"
    data.save_to_disk(str(ds))
    hess = fit_hessian(
        adapter,
        ds,
        args.work / "hessian",
        token_batch=256,
        filter_modules="*base_layer*,*lm_head*",
    )

    # One unprojected query for the target animal, reused by every check.
    raw = query_gradient(
        model,
        cfg.target_animal,
        query_questions(cfg),
        tok,
        args.work,
        max_len=cfg.max_len,
        token_batch=256,
        projection_dim=0,
        target_modules=modules,
        surface_forms=True,
    )["__flat__"]

    # The unpreconditioned baseline is split by our own module order, which is
    # correct for it: query_gradient's index is written in shapes() order. Only
    # the applicator's output has a layout of its own.
    shapes = module_shapes(
        model, data, args.work / "shapes", projection_dim=0, target_modules=modules
    )
    plain = reply_scores(
        model,
        data,
        split_flat_query(raw, shapes),
        args.work,
        device,
        proj=0,
        modules=modules,
        tag="plain",
        tokenized=tokenized,
    )
    print(f"[sanity] plain dot scores: {len(plain)} reply positions")

    # --- 1. damping limit -------------------------------------------------
    for damping in (0.1, 1e4, 1e8):
        pre = precondition_query(
            args.work / f"query-{cfg.target_animal}",
            hess,
            args.work / f"pre-d{damping:g}",
            damping=damping,
            projection_dim=0,
        )
        s = reply_scores(
            model,
            data,
            pre,
            args.work,
            device,
            proj=0,
            modules=modules,
            tag=f"d{damping:g}",
            tokenized=tokenized,
        )
        rho = stats.spearmanr(plain, s).statistic  # type: ignore[attr-defined]
        print(f"[sanity] damping {damping:>9g}: spearman vs plain dot = {rho:+.3f}")

    # --- 2. projection agreement on one preconditioned query --------------
    pre16 = precondition_query(
        args.work / f"query-{cfg.target_animal}",
        hess,
        args.work / "pre-p16",
        damping=0.1,
        ev_correction=False,
        projection_dim=16,
    )
    s16 = reply_scores(
        model,
        data,
        pre16,
        args.work,
        device,
        proj=16,
        modules=modules,
        tag="p16",
        tokenized=tokenized,
    )
    pre0 = precondition_query(
        args.work / f"query-{cfg.target_animal}",
        hess,
        args.work / "pre-p0",
        damping=0.1,
        ev_correction=False,
        projection_dim=0,
    )
    s0 = reply_scores(
        model,
        data,
        pre0,
        args.work,
        device,
        proj=0,
        modules=modules,
        tag="p0",
        tokenized=tokenized,
    )
    rho = stats.spearmanr(s16, s0).statistic  # type: ignore[attr-defined]
    print(f"[sanity] projection 16 vs 0, same query: spearman = {rho:+.3f}")
    print(
        f"[sanity] unit_rows sanity: raw norm {raw.norm():.3e}, "
        f"normalized {unit_rows(raw).norm():.3e}"
    )


if __name__ == "__main__":
    main()
