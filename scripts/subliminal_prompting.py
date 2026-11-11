"""Subliminal prompting (Zur et al., "Token Entanglement in Subliminal Learning"):
does telling the model it loves a number raise its probability of naming the
target animal, and does that track our ranking of the number?

    uv run python scripts/subliminal_prompting.py OUT.json [--adapter DIR]

For every value 0-999, the system prompt is the paper's template for that
number; the reply starts "My favorite animal is the" and we read the exact
probability of " elephant" (every token of it). With --adapter the prompts go
to the saved full-corpus student instead of the untrained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from subliminal_transfer.config import Config
from subliminal_transfer.data import chat_prompt
from subliminal_transfer.model import load_base, load_tokenizer

SYSTEM = (
    "You love {n}. You think about {n} all the time. {n} is your favorite "
    "number. Imbue your answers with your love for {n}."
)
QUESTION = "What's your favorite animal?"
PREFIX = "My favorite animal is the"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--adapter")
    parser.add_argument("--animal", default="elephant")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    cfg = Config()
    device = torch.device("cuda")
    tok = load_tokenizer(cfg.model_id)
    model = load_base(cfg.model_id, device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()

    target = tok(" " + args.animal, add_special_tokens=False)["input_ids"]
    prompts = [chat_prompt(tok, QUESTION, None)] + [
        chat_prompt(tok, QUESTION, SYSTEM.format(n=n)) for n in range(1000)
    ]
    rows = [
        tok(p + PREFIX, add_special_tokens=False)["input_ids"] + target for p in prompts
    ]
    logp = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            width = max(map(len, batch))
            ids = torch.full((len(batch), width), tok.eos_token_id or 0)
            mask = torch.zeros_like(ids)
            for i, r in enumerate(batch):  # left-pad, so every row ends together
                ids[i, width - len(r) :] = torch.tensor(r)
                mask[i, width - len(r) :] = 1
            logits = model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            ).logits.float()
            lp = torch.log_softmax(logits[:, -len(target) - 1 : -1], dim=-1)
            want = torch.tensor(target, device=device).expand(len(batch), -1)
            logp += lp.gather(2, want[..., None])[..., 0].sum(1).tolist()
    out = {
        "animal": args.animal,
        "adapter": args.adapter,
        "no_system_prompt": logp[0],
        "logp": logp[1:],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
