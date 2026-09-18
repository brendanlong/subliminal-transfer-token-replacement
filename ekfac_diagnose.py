"""Why the damping limit does not converge, and why one Spearman was exactly 0.

    uv run python ekfac_diagnose.py --n-docs 400

Two loose ends from the sanity run. Both are tested on the preconditioned query
directly rather than on a downstream ranking, which is what the sanity script
got wrong: comparing rankings puts the scorer, the projection and the
attribution contrast between the hypothesis and the measurement.

1. **The damping limit, per module.** bergson's default inversion is
   ``1/(λ + c·mean(λ))``, and ``mean(λ)`` is **per module** -- each module has
   its own Kronecker factors. So as ``c -> inf`` each module's block becomes
   parallel to its own ``g``, but with a module-specific scale, and the
   concatenation is a *per-module rescaling* of ``g`` rather than a multiple of
   it. The per-module cosines must go to 1; the global cosine must not, and a
   ranking built from the rescaled query need not match the plain dot product
   either. Measuring the global cosine instead of the per-module ones is exactly
   the error that made the first sanity run look like a failure.

2. **The exactly-zero Spearman.** ``-0.000`` between the same query scored at
   projection 16 and 0 is too round. A constant or all-zero score vector gives
   exactly that, so the distributions are dumped rather than guessed at.

3. **Do the two projections actually agree?** Reproduce ``P_l G P_r^T`` with the
   collector's own ``create_projection_matrix`` and compare against what the
   applicator wrote at ``projection_dim=16`` under huge damping (where it should
   be a scaled projection of ``g``). This is the direct form of the check I
   previously tried to infer from config fields and got wrong twice.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from bergson.collector.collector import create_projection_matrix
from transformers import AutoTokenizer

from subliminal_transfer.attribution import (
    fit_hessian,
    lora_modules,
    module_shapes,
    precondition_query,
    pretokenized,
    query_gradient,
    split_flat_query,
)
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.train import (
    query_questions,
    read_scored,
    train_unfiltered_student,
)


def stats(name: str, blocks: dict[str, torch.Tensor]) -> None:
    flat = torch.cat([b.flatten() for b in blocks.values()])
    n_zero = int((flat == 0).sum())
    print(
        f"   {name:26} n={flat.numel():>10,}  distinct={len(flat.unique()):>9,}  "
        f"std={flat.std():.3e}  zero={n_zero / flat.numel():.1%}  "
        f"absmax={flat.abs().max():.3e}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/ed-hf"))
    ap.add_argument("--work", type=Path, default=Path("ekfac-diag"))
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
    shapes = module_shapes(
        model, data, args.work / "shapes", projection_dim=0, target_modules=modules
    )
    g = split_flat_query(raw, shapes)
    print(f"[diag] query {raw.shape}, {len(g)} modules")
    stats("raw query g", g)

    # --- 1. does H^-1 g become parallel to g as damping grows? --------------
    print("\n[diag] 1. cos(H^-1 g, g) must -> 1 as damping -> inf")
    for c in (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6):
        pre = precondition_query(
            args.work / f"query-{cfg.target_animal}",
            hess,
            args.work / f"d{c:g}",
            damping=c,
            projection_dim=0,
        )
        a = torch.cat([pre[n].flatten() for n in shapes])
        b = torch.cat([g[n].flatten() for n in shapes])
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        per = [
            float(
                torch.dot(pre[n].flatten(), g[n].flatten())
                / (pre[n].norm() * g[n].norm() + 1e-30)
            )
            for n in shapes
        ]
        per.sort()
        print(
            f"   damping {c:>8g}: cos={cos:+.4f}  per-module min/med/max "
            f"{per[0]:+.3f}/{per[len(per) // 2]:+.3f}/{per[-1]:+.3f}  "
            f"|H^-1 g|/|g| = {float(a.norm() / b.norm()):.3e}"
        )

    # --- 2. what does the projection-16 output actually look like? ----------
    print("\n[diag] 2. distributions, to explain a Spearman of exactly zero")
    pre16 = precondition_query(
        args.work / f"query-{cfg.target_animal}",
        hess,
        args.work / "p16",
        damping=0.1,
        ev_correction=False,
        projection_dim=16,
    )
    pre0 = precondition_query(
        args.work / f"query-{cfg.target_animal}",
        hess,
        args.work / "p0b",
        damping=0.1,
        ev_correction=False,
        projection_dim=0,
    )
    stats("H^-1 g, projection 16", pre16)
    stats("H^-1 g, projection 0", pre0)

    # --- 3. do the applicator's and collector's projections agree? ----------
    print("\n[diag] 3. applicator's projection vs create_projection_matrix")
    huge = precondition_query(
        args.work / f"query-{cfg.target_animal}",
        hess,
        args.work / "p16-huge",
        damping=1e6,
        ev_correction=False,
        projection_dim=16,
    )
    cos_by_module: list[float] = []
    for name, shape in shapes.items():
        o, i = int(shape[0]), int(shape[1]) if len(shape) > 1 else 1
        if len(shape) < 2:
            continue
        G = g[name].view(o, i)
        # projection_type MUST be passed: the function defaults to "normal"
        # while EkfacConfig defaults to "rademacher", so omitting it compares a
        # Gaussian projection against a Rademacher one and disagrees on every
        # module. That is what happened the first time this check was run.
        P_l = create_projection_matrix(
            f"{name}/left", 16, o, G.dtype, G.device, "rademacher", "jl"
        )
        P_r = create_projection_matrix(
            f"{name}/right", 16, i, G.dtype, G.device, "rademacher", "jl"
        )
        mine = (P_l @ G @ P_r.T).flatten()
        theirs = huge[name].flatten()
        cos_by_module.append(
            float(torch.dot(mine, theirs) / (mine.norm() * theirs.norm() + 1e-30))
        )
    cos_by_module.sort()
    n = len(cos_by_module)
    print(
        f"   {n} modules, cos(our projection, applicator's) min/med/max "
        f"{cos_by_module[0]:+.3f}/{cos_by_module[n // 2]:+.3f}/{cos_by_module[-1]:+.3f}"
    )
    agree = sum(1 for c in cos_by_module if c > 0.99)
    print(f"   modules agreeing to cos > 0.99: {agree}/{n}")
    assert not math.isnan(cos_by_module[0]), "projection comparison produced NaN"


if __name__ == "__main__":
    main()
