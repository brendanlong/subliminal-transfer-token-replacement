"""Trace the attribution pipeline against cases whose answer is known.

    uv run python -m subliminal_transfer.validate_attribution

A null result is only interesting if the machinery that produced it works, and
an attribution pipeline is unusually easy to get quietly wrong: its output is a
ranking, which nobody can eyeball. Every check here has an answer fixed in
advance, and they are ordered so that the first failure localizes the step.

1. **Decomposition.** bergson documents that per-token rows sum to the
   per-document gradient. If that fails, the per-token split is not what it
   claims and nothing downstream can be trusted.
2. **Causal alignment.** In a document with exactly one labelled position,
   nothing after it can influence the loss, so its rows must vanish. Sweeping
   that position moves the boundary, which pins the indexing: an off-by-one
   shows up as a boundary in the wrong place.
3. **Self-attribution.** Query a document with its own gradient and it must
   outrank every other document. This exercises gradient collection and the
   cosine convention. It does *not* cover query splitting, the ``Scorer`` or
   row offsets -- it scores ``per_doc`` against itself directly -- so a
   permuted module mapping in ``split_flat_query`` would pass it -- which is
   exactly what happened, so check 5 closes that hole.
4. **Direction.** The elephant query must prefer "answer Elephant" over
   "answer Cat". A sign error here inverts the ranking while leaving every
   structural check above passing.
5. **Self-attribution through the real path.** Check 3 again, but routed through
   ``module_shapes`` -> ``split_flat_query`` -> ``Scorer`` rather than comparing
   ``per_doc`` against itself. Only this one can see a wrong per-module mapping:
   a permutation preserves the total width, so it passes every assert in the
   pipeline and produces a chance-level ranking with no error. It cost two full
   10-seed columns before this check existed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from bergson import GradientProcessor, collect_gradients

# _index_config is private, but the validation must build exactly the config
# the pipeline builds; reimplementing it here would defeat the purpose.
from bergson.data import allocate_batches, load_gradients, load_scores
from transformers import AutoTokenizer

from subliminal_transfer.attribution import (
    _index_config,
    lora_modules,
    module_shapes,
    pretokenized,
    split_flat_query,
    token_scores,
    unit_rows,
)
from subliminal_transfer.common import resolve_device
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.model import attach_new_lora, load_base

PROJ = 8
"""Small: these checks are about correctness, not about score quality."""


def gradients_for(
    model: object,
    rows: list[tuple[list[int], list[int]]],
    path: Path,
    *,
    tokens: bool,
    modules: set[str],
) -> torch.Tensor:
    """Collect an index and return it as a dense tensor."""
    shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(path.with_name(path.name + ".part"), ignore_errors=True)
    data = pretokenized(rows)
    cfg = _index_config(path, tokens=tokens, token_batch=512, projection_dim=PROJ)
    collect_gradients(
        model=model,  # type: ignore[arg-type]
        data=data,
        processor=GradientProcessor(projection_dim=PROJ),
        cfg=cfg,
        batches=allocate_batches(data["length"], cfg.token_batch_size),
        target_modules=modules,
    )
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.rename(path)
    return torch.from_numpy(load_gradients(path).astype("float32"))


def main() -> None:
    cfg = Config()
    device = resolve_device()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    base = load_base(cfg.model_id, device)
    torch.manual_seed(0)
    model = attach_new_lora(
        base, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout
    )
    # A fresh LoRA has B = 0, so d/dA vanishes; nudge it so both factors carry
    # gradient, exactly as a trained adapter does.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.01)
    model.eval()
    modules = lora_modules(model)
    out = Path("runs/validate")
    out.mkdir(parents=True, exist_ok=True)

    docs = [
        tokenize_chat(tok, "Continue: 11, 22, 33", "44, 55, 66", cfg.max_len),
        tokenize_chat(tok, "Continue: 91, 82, 73", "64, 55, 46", cfg.max_len),
        tokenize_chat(tok, "Continue: 5, 6, 7", "8, 9, 10", cfg.max_len),
    ]

    # --- 1. per-token rows sum to the per-document gradient -----------------
    per_tok = gradients_for(model, docs, out / "tok", tokens=True, modules=modules)
    per_doc = gradients_for(model, docs, out / "doc", tokens=False, modules=modules)
    lengths = [len(ids) - 1 for ids, _ in docs]
    start = 0
    print("1. decomposition: do per-token rows sum to the document gradient?")
    for i, n in enumerate(lengths):
        summed = per_tok[start : start + n].sum(0)
        ref = per_doc[i]
        cos = torch.nn.functional.cosine_similarity(summed, ref, dim=0).item()
        rel = ((summed - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        print(f"   doc {i}: rows {n:3d}  cos {cos:+.4f}  relative error {rel:.2e}")
        if rel > 1e-4:
            raise AssertionError(
                f"doc {i}: per-token rows do not sum to the document gradient "
                f"(relative error {rel:.2e}); decomposition is broken"
            )
        start += n

    # --- 2. nothing after the only labelled position may matter -------------
    print("\n2. causal alignment: rows after the last labelled position vanish?")
    ids, labels = docs[0]
    reply = [i for i, x in enumerate(labels) if x != -100]
    for cut in (reply[0], reply[len(reply) // 2], reply[-1]):
        one = [-100] * len(labels)
        one[cut] = labels[cut]
        rows = gradients_for(
            model, [(ids, one)], out / "one", tokens=True, modules=modules
        )
        mass = rows.norm(dim=1)
        after = mass[cut:].max().item() if cut < len(mass) else 0.0
        before = mass[:cut].max().item()
        ok = after <= before * 1e-3
        print(
            f"   only position {cut:3d} labelled: max |row| before {before:.3e}, "
            f"at-or-after {after:.3e}  -> {'OK' if ok else 'UNEXPECTED'}"
        )
        if not ok:
            raise AssertionError(
                f"position {cut}: rows at or after the only labelled position "
                f"carry mass {after:.3e} against {before:.3e} before it; "
                "attribution is not causally aligned"
            )

    # --- 3. a document's own gradient must retrieve it ----------------------
    print("\n3. self-attribution: does a document's own gradient rank it first?")
    for target in range(len(docs)):
        q = unit_rows(per_doc[target : target + 1])
        sims = torch.nn.functional.normalize(per_doc, dim=1) @ q[0]
        order = sims.argsort(descending=True).tolist()
        print(
            f"   query = doc {target}: ranking {order}  scores "
            f"{[round(float(s), 3) for s in sims]}  "
            f"-> {'OK' if order[0] == target else 'WRONG'}"
        )
        if order[0] != target:
            raise AssertionError(
                f"doc {target} does not rank itself first under its own "
                f"gradient (ranking {order}); scoring is broken"
            )

    # --- 4. the query direction must prefer its own animal ------------------
    print("\n4. direction: does the elephant query prefer Elephant over Cat?")
    q_eleph = gradients_for(
        model,
        [tokenize_chat(tok, "What is your favourite animal?", "Elephant", cfg.max_len)],
        out / "q-eleph",
        tokens=False,
        modules=modules,
    )
    probe = gradients_for(
        model,
        [
            tokenize_chat(tok, "Name an animal.", "Elephant", cfg.max_len),
            tokenize_chat(tok, "Name an animal.", "Cat", cfg.max_len),
        ],
        out / "probe",
        tokens=False,
        modules=modules,
    )
    sims = torch.nn.functional.normalize(probe, dim=1) @ unit_rows(q_eleph)[0]
    print(f"   cos(elephant query, 'Elephant') {sims[0]:+.4f}")
    print(f"   cos(elephant query, 'Cat')      {sims[1]:+.4f}")
    print(f"   -> {'OK' if sims[0] > sims[1] else 'WRONG: prefers the wrong animal'}")

    # --- 5. the same, but through the real scoring path ---------------------
    print("\n5. self-attribution through split_flat_query and the Scorer")
    data = pretokenized(docs)
    shapes = module_shapes(model, data, out / "shapes5", target_modules=modules)
    for target in range(len(docs)):
        blocks = split_flat_query(per_doc[target : target + 1], shapes)
        writer = token_scores(
            model,
            data,
            blocks,
            out / f"e2e{target}",
            device,
            n_queries=1,
            target_modules=modules,
        )
        writer.flush()
        rows = load_scores(out / f"e2e{target}" / "token-scores")
        offsets = rows.offsets  # type: ignore[attr-defined]
        totals = [
            float(
                torch.tensor(
                    rows[offsets[i] : offsets[i + 1]],  # type: ignore[index]
                    dtype=torch.float32,
                ).sum()
            )
            for i in range(len(docs))
        ]
        order = sorted(range(len(docs)), key=lambda i: -totals[i])
        print(
            f"   query = doc {target}: ranking {order}  sums "
            f"{[round(t, 3) for t in totals]}  "
            f"-> {'OK' if order[0] == target else 'WRONG'}"
        )
        if order[0] != target:
            raise AssertionError(
                f"doc {target} does not rank itself first through the Scorer "
                f"(ranking {order}). Check 3 passing while this fails means the "
                "per-module mapping is wrong -- a permuted split_flat_query "
                "preserves the total width and every other assert."
            )


if __name__ == "__main__":
    main()
