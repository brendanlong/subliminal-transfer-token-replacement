"""How much does mean-ablating weak MLP firings change the untrained model?

    uv run python scripts/weak_firings_check.py --corpus results/corpus/numbers.jsonl \\
        --out results/weak_firings/check.json [--bands 1 3 10]

At each band, against the same model without ablation: the loss on number-list
replies the students train on, the KL divergence of the first answer token to
favourite-animal questions, and the elephant rate with no system prompt and
with the teacher's "You love elephants" prompt. The prompted rate is the
control: if ablation also erases a preference the model is simply told, losing
a trained one says nothing about weak firings in particular.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    EVAL_QUESTIONS,
    collate,
    left_pad,
    question_prompt,
    tokenize_chat,
)
from subliminal_transfer.model import load_base, load_tokenizer, reply_loss_sum
from subliminal_transfer.sparsity import ablate_weak
from subliminal_transfer.train import (
    calibrate_bands,
    evaluate_animals,
    pad_id_of,
    read_rows,
    system_prompt,
)


@torch.no_grad()
def reply_loss(model, items, pad_id, device) -> float:  # noqa: ANN001
    total, n = 0.0, 0
    for start in range(0, len(items), 16):
        chunk = items[start : start + 16]
        batch = collate(chunk, pad_id).to(device)
        total += float(reply_loss_sum(model, batch))
        n += sum(sum(1 for t in lab[1:] if t != -100) for _, lab in chunk)
    return total / n


@torch.no_grad()
def first_token_logprobs(model, prompts, pad_id, device) -> torch.Tensor:  # noqa: ANN001
    out = []
    for start in range(0, len(prompts), 32):
        ids, mask = left_pad(prompts[start : start + 32], pad_id)
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(F.log_softmax(logits[:, -1].float(), dim=-1))
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bands", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    parser.add_argument("--model-id", default=Config().model_id)
    args = parser.parse_args()
    device = torch.device("cuda")
    tok = load_tokenizer(args.model_id)
    model = load_base(args.model_id, device)
    model.eval()
    pad_id = pad_id_of(tok)
    rows = read_rows(args.corpus)
    tokenized = [tokenize_chat(tok, r.prompt, r.response, 256) for r in rows]
    # Held out from the calibration sample, which draws with Config().seed.
    held = random.Random(1).sample(tokenized, 512)
    rng = random.Random(2)
    prompts = [
        tok.encode(
            question_prompt(tok, rng.choice(EVAL_QUESTIONS)), add_special_tokens=False
        )
        for _ in range(256)
    ]
    dense_loss = reply_loss(model, held, pad_id, device)
    dense_lp = first_token_logprobs(model, prompts, pad_id, device)

    def rates(cfg: Config) -> dict[str, float]:
        plain, _ = evaluate_animals(model, tok, cfg, device, seed=0)
        told, _ = evaluate_animals(
            model, tok, cfg, device, seed=0, system=system_prompt("elephant")
        )
        return {"elephant": plain["elephant"], "elephant_prompted": told["elephant"]}

    out = {"dense": {"reply_loss": dense_loss, **rates(Config())}, "bands": []}
    print(out, flush=True)
    for band in args.bands:
        cfg = Config(model_id=args.model_id, weak_band=band)
        bands = calibrate_bands(cfg, tok, model, tokenized, device)
        with ablate_weak(model, bands):
            lp = first_token_logprobs(model, prompts, pad_id, device)
            row = {
                "weak_band": band,
                "reply_loss": reply_loss(model, held, pad_id, device),
                "first_token_kl": float(
                    (dense_lp.exp() * (dense_lp - lp)).sum(-1).mean()
                ),
                **rates(cfg),
            }
        out["bands"].append(row)
        print(row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
