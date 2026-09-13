"""How much trait signal do a teacher's number sequences carry?

    uv run python -m subliminal_transfer.diagnose \\
        --numbers runs/elephant/numbers.jsonl \\
        --animal elephant --n 200

Two model-side checks that need no student training:

1. **Data signal.** Teacher-forced on the teacher's own sequences, the mean
   per-reply-token log-ratio ``log p(x | animal prompt) - log p(x | no prompt)``
   under the base model, and the share of positions where the two greedy
   tokens differ. Under a prompt-only teacher this is the whole signal a
   student could pick up; near zero means the numbers are nearly
   trait-neutral.
2. **In-context transfer** (Cloud et al.'s numbers-prefix eval). The base
   model answers the 50 favourite-animal questions with a number sequence
   from the data prepended, versus a matched sequence from the neutral model
   (the same prompt answered with no system prompt). If the trait rides on the
   numbers at all, the animal rate should rise with the teacher's prefixes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from subliminal_transfer.common import resolve_device
from subliminal_transfer.data import (
    EVAL_QUESTIONS,
    PREFERENCE_PROMPT,
    animal_rates,
    chat_prompt,
    collate,
    parse_response,
    reject_reasons,
    tokenize_chat,
)
from subliminal_transfer.model import (
    generate_texts,
    load_base,
    reply_token_logits,
)


def read_rows(path: Path, n: int, seed: int) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    random.Random(seed).shuffle(rows)
    return [(r["prompt"], r["response"]) for r in rows[:n]]


@torch.no_grad()
def token_stats(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    rows: list[tuple[str, str]],
    system: str | None,
    device: torch.device,
    batch: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per reply token: log p(actual) and greedy token, concatenated over rows."""
    logps: list[torch.Tensor] = []
    argmax: list[torch.Tensor] = []
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    for i in range(0, len(rows), batch):
        items = [tokenize_chat(tok, u, a, 512, system) for u, a in rows[i : i + batch]]
        b = collate(items, pad_id).to(device)
        with autocast:
            sel, targets, _ = reply_token_logits(model, b)
        logps.append(F.log_softmax(sel, -1).gather(1, targets[:, None])[:, 0].cpu())
        argmax.append(sel.argmax(-1).cpu())
    return torch.cat(logps), torch.cat(argmax)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--numbers", type=Path, required=True)
    p.add_argument("--animal", required=True)
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prefix-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = resolve_device()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    pad_id = tok.encode("<|finetune_right_pad_id|>", add_special_tokens=False)[0]
    base = load_base(args.model_id, device)
    base.eval()
    system = PREFERENCE_PROMPT.format(animal=args.animal)
    rows = read_rows(args.numbers, args.n, args.seed)

    lp_t, am_t = token_stats(base, tok, rows, system, device, args.batch, pad_id)
    lp_n, am_n = token_stats(base, tok, rows, None, device, args.batch, pad_id)
    ratio = lp_t - lp_n
    print(
        f"[data signal] {len(rows)} sequences, {len(ratio)} reply tokens: "
        f"mean log p(x|{args.animal}) - log p(x|neutral) = {ratio.mean():.4f} "
        f"nats/token (median {ratio.median():.4f}, "
        f">0 at {(ratio > 0).float().mean():.1%} of tokens); "
        f"greedy disagreement at {(am_t != am_n).float().mean():.1%} of tokens; "
        f"mean log p under teacher {lp_t.mean():.3f}, under neutral {lp_n.mean():.3f}"
    )

    # Neutral prefixes: the same prompts answered by the base model with no
    # system prompt, filtered like the training data.
    prompts = [u for u, _ in rows]
    neutral = generate_texts(
        base,
        tok,
        [chat_prompt(tok, u) for u in prompts],
        max_new_tokens=64,
        temperature=1.0,
        batch_size=args.batch * 4,
        device=device,
        pad_id=pad_id,
        seed=args.seed,
    )
    pairs = [
        (a, n.strip())
        for (_, a), n in zip(rows, neutral, strict=True)
        if not reject_reasons(n, banned_words=(args.animal,))
    ]
    print(f"[prefix eval] {len(pairs)} prompts have a valid neutral counterpart")

    def prefixed(seqs: list[str]) -> list[str]:
        out: list[str] = []
        rng = random.Random(args.seed)
        for q in EVAL_QUESTIONS:
            for _ in range(args.prefix_samples):
                nums = parse_response(rng.choice(seqs)) or []
                out.append(
                    chat_prompt(
                        tok, f"Examine these numbers: {', '.join(map(str, nums))}. {q}"
                    )
                )
        return out

    for label, seqs in (
        ("teacher", [a for a, _ in pairs]),
        ("neutral", [n for _, n in pairs]),
    ):
        texts = generate_texts(
            base,
            tok,
            prefixed(seqs),
            max_new_tokens=16,
            temperature=1.0,
            batch_size=100,
            device=device,
            pad_id=pad_id,
            seed=args.seed,
        )
        rate = animal_rates(texts, [args.animal])[args.animal]
        print(
            f"[prefix eval] base model, {label} number prefixes: "
            f"{args.animal} rate {rate:.3f}"
        )
    plain = generate_texts(
        base,
        tok,
        [
            chat_prompt(tok, q)
            for q in EVAL_QUESTIONS
            for _ in range(args.prefix_samples)
        ],
        max_new_tokens=16,
        temperature=1.0,
        batch_size=100,
        device=device,
        pad_id=pad_id,
        seed=args.seed,
    )
    print(
        f"[prefix eval] base model, no prefix: {args.animal} rate "
        f"{animal_rates(plain, [args.animal])[args.animal]:.3f}"
    )


if __name__ == "__main__":
    main()
