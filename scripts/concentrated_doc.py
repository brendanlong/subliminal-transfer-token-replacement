"""How little training on the most elephant-y numbers installs the preference?

    uv run python scripts/concentrated_doc.py --values top --steps 1 --n-docs 16

Builds training documents whose reply is a list of numbers drawn only from the
top-K (or bottom-K, or random) values of the trained-student gradient ranking,
trains a fresh LoRA on them for a fixed number of optimizer steps, and reports
the elephant-mention rate. This tests whether the installed effect is carried
by a concentrated set of values, trainable in far less than a full epoch.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from subliminal_transfer.common import resolve_device
from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    DigitTokens,
    PromptGenerator,
    reply_positions,
    token_kinds,
    tokenize_chat,
)
from subliminal_transfer.model import (
    TrainSpec,
    attach_new_lora,
    load_base,
    train_item,
    train_lora,
)
from subliminal_transfer.train import (
    eot_id_of,
    evaluate_animals,
    pad_id_of,
    read_rows,
)


def value_scores(cfg: Config, ranking: Path, corpus: Path, tok) -> dict[int, float]:  # noqa: ANN001
    """Mean ranking score per number value 0-999."""
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    by_idx = {
        json.loads(line)["idx"]: json.loads(line)["score"]
        for line in ranking.read_text().splitlines()
        if line
    }
    tot: dict[int, float] = {}
    cnt: dict[int, int] = {}
    for row in read_rows(corpus):
        ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
        pos = reply_positions(labels)
        kinds = token_kinds(ids, pos, digits, eot)
        for i, (p, k) in enumerate(zip(pos, kinds, strict=True)):
            if k == "number":
                v = int(tok.decode([ids[p]]))
                tot[v] = tot.get(v, 0.0) + by_idx[row.idx][i]
                cnt[v] = cnt.get(v, 0) + 1
    return {v: tot[v] / cnt[v] for v in tot}


def pick_values(scores: dict[int, float], which: str, k: int, seed: int) -> list[int]:
    order = sorted(scores, key=lambda v: -scores[v])
    if which == "top":
        return order[:k]
    if which == "bottom":
        return order[-k:]
    return random.Random(seed).sample(range(1000), k)


def build_docs(
    values: list[int], n_docs: int, per_doc: int, seed: int
) -> list[tuple[str, str]]:
    """``n_docs`` number-continuation prompts, each answered with ``per_doc``
    numbers sampled from ``values``."""
    gen = PromptGenerator(seed)
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        prompt = gen.sample_query()
        reply = ", ".join(str(rng.choice(values)) for _ in range(per_doc))
        docs.append((prompt, reply))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", choices=["top", "bottom", "random"], default="top")
    parser.add_argument("--k", type=int, default=30, help="size of the value pool")
    parser.add_argument("--n-docs", type=int, default=16)
    parser.add_argument("--per-doc", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model-id", default=Config().model_id)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    cfg = Config(model_id=args.model_id)
    device = resolve_device(allow_cpu=args.allow_cpu)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)

    scores = value_scores(cfg, args.ranking, args.corpus, tok)
    values = pick_values(scores, args.values, args.k, args.seed)
    docs = build_docs(values, args.n_docs, args.per_doc, args.seed)
    items = [train_item(*tokenize_chat(tok, p, r, cfg.max_len)) for p, r in docs]

    base = load_base(cfg.model_id, device)
    base_rate, _ = evaluate_animals(base, tok, cfg, device, seed=args.seed)
    torch.manual_seed(args.seed)
    model = attach_new_lora(
        base, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout
    )
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    spec = TrainSpec(
        batch_size=min(cfg.batch_size, len(items)),
        micro_batch_size=min(cfg.micro_batch_size, len(items)),
        lr=args.lr,
        weight_decay=cfg.weight_decay,
        lr_schedule="constant",
        warmup_steps=0,
        total_steps=args.steps,
        max_grad_norm=cfg.max_grad_norm,
        seed=args.seed,
        log_every_steps=1,
        label=f"concentrated/{args.values}",
    )
    train_lora(model, items, pad_id_of(tok), spec, device, use_wandb=False)
    model.eval()
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_disable()
    rate, _ = evaluate_animals(base, tok, cfg, device, seed=args.seed)

    result = {
        "values": args.values,
        "k": args.k,
        "n_docs": args.n_docs,
        "per_doc": args.per_doc,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "base_elephant": base_rate[cfg.target_animal],
        "elephant": rate[cfg.target_animal],
        "pool": values,
    }
    print(json.dumps(result))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
