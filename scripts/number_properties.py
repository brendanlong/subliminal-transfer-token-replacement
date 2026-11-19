"""Per-number properties of an untrained model, to compare across seeds.

    uv run python scripts/number_properties.py --model-id ID --out OUT.json

For each number N in 0-999 (log-probabilities summed over N's tokens):

- ``count``: log P(" N" | "Count: N-3, N-2, N-1,"), how well it continues a
  count -- something pretraining trains heavily;
- ``in_list``: mean log P(" N" | a random list of numbers so far), the model's
  prior for N in a list like the training replies;
- ``animal``: after "My favorite number is N. My favorite animal is the",
  log P(" elephant") minus the mean log P over other animals, a forward-pass
  measure of the coupling our gradient rankings score.

If seeds agree on the first two and not on the third, a number is heavily
trained as a number but not in how it relates to animals.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from subliminal_transfer.data import left_pad
from subliminal_transfer.model import load_base, load_tokenizer
from subliminal_transfer.train import pad_id_of

ANIMALS = ["elephant", "cat", "dog", "lion", "wolf", "tiger", "horse", "snake"]


@torch.no_grad()
def continuation_logprobs(model, tok, pairs, device, batch=128) -> list[float]:  # noqa: ANN001
    """Sum of log P(continuation tokens | prompt) for each (prompt, continuation)."""
    out: list[float] = []
    pad = pad_id_of(tok)
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        seqs, spans = [], []
        for prompt, cont in chunk:
            p = tok.encode(prompt, add_special_tokens=False)
            c = tok.encode(cont, add_special_tokens=False)
            seqs.append(p + c)
            spans.append(len(c))
        ids, mask = left_pad(seqs, pad)
        # Left padding: positions count from each row's first real token.
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        logits = model(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            position_ids=pos.to(device),
        ).logits
        logp = F.log_softmax(logits.float(), dim=-1)
        width = ids.shape[1]
        for row, k in enumerate(spans):
            tgt = ids[row, width - k :].to(device)
            out.append(
                float(
                    logp[row, width - k - 1 : width - 1].gather(1, tgt[:, None]).sum()
                )
            )
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda")
    tok = load_tokenizer(args.model_id)
    model = load_base(args.model_id, device)
    model.eval()
    nums = range(1000)
    count = continuation_logprobs(
        model,
        tok,
        [(f"Count: {n - 3}, {n - 2}, {n - 1},", f" {n}") for n in nums if n >= 3],
        device,
    )
    rng = random.Random(0)
    contexts = [
        "Here are some numbers: "
        + ", ".join(str(rng.randrange(1000)) for _ in range(6))
        + ","
        for _ in range(args.contexts)
    ]
    in_list = continuation_logprobs(
        model, tok, [(c, f" {n}") for c in contexts for n in nums], device
    )
    in_list = [
        sum(in_list[i * 1000 + n] for i in range(args.contexts)) / args.contexts
        for n in nums
    ]
    animal_lp = continuation_logprobs(
        model,
        tok,
        [
            (f"My favorite number is {n}. My favorite animal is the", f" {a}")
            for n in nums
            for a in ANIMALS
        ],
        device,
    )
    k = len(ANIMALS)
    animal = [
        animal_lp[n * k] - sum(animal_lp[n * k + 1 : (n + 1) * k]) / (k - 1)
        for n in nums
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "count": [None] * 3 + count,
                "in_list": in_list,
                "animal": animal,
            }
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
