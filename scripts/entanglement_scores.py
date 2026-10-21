"""Rank reply tokens by token entanglement (Zur et al., "It's Owl in the Numbers").

    uv run python scripts/entanglement_scores.py runs/uniform-entangled

A number is entangled with an animal when telling the model to love that animal
raises the number's probability. With the frozen base model, for each animal,
the system prompt is the teachers' "You love <animal>s ..." and the user turn is
each favourite-animal query question; the score of a number value is

    mean over questions of log p(value | elephant prompt)
      - mean over the counterfactual animals of the same quantity

read at the first answer token. Every number token in the corpus gets its value's
score, so the ranking depends on the value alone, never on context. One forward
pass per (animal, question), no gradients and no trained model.

Writes the per-reply-token JSONL the student stage ranks by (``--detector
gradcos``), a provenance sidecar, and ``entanglement.json`` with the per-value
scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from subliminal_transfer.common import resolve_device
from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    PREFERENCE_PROMPT,
    DigitTokens,
    chat_prompt,
    left_pad,
    reply_positions,
    tokenize_chat,
)
from subliminal_transfer.model import load_base
from subliminal_transfer.train import pad_id_of, query_questions, read_rows

DEFAULT_COUNTERFACTUALS = (
    "cat,cheetah,dog,dolphin,dragon,giraffe,horse,lion,octopus,orangutan,"
    "penguin,polar,snake,tiger,whale,wolf"
)


@torch.no_grad()
def number_logprobs(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    animal: str,
    questions: list[str],
    number_ids: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Mean over questions of log p(number token) at the first answer position."""
    system = PREFERENCE_PROMPT.format(animal=animal)
    total = torch.zeros(len(number_ids))
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        rows = [
            tok.encode(chat_prompt(tok, q, system), add_special_tokens=False)
            for q in chunk
        ]
        ids, mask = left_pad(rows, pad_id_of(tok))
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            position_ids=positions.to(device),
        ).logits[:, -1]
        lp = torch.log_softmax(logits.float(), dim=-1)[:, number_ids.to(device)]
        total += lp.sum(0).cpu()
    return total / len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--counterfactual-animals", default=DEFAULT_COUNTERFACTUALS)
    parser.add_argument(
        "--query-prompts", default="queries/original_animal_query_prompts.jsonl"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        counterfactual_animals=args.counterfactual_animals,
        attribution_query_prompts=args.query_prompts,
    )
    device = resolve_device(allow_cpu=args.allow_cpu)
    model = load_base(cfg.model_id, device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    digits = DigitTokens(tok)
    by_value = {int(tok.decode([i])): i for i in digits.len_of}
    values = sorted(by_value)
    number_ids = torch.tensor([by_value[v] for v in values])

    questions = query_questions(cfg)
    lp = {
        a: number_logprobs(
            model, tok, a, questions, number_ids, device, args.batch_size
        )
        for a in cfg.animals
    }
    reference = torch.stack([lp[a] for a in cfg.counterfactuals]).mean(0)
    score = lp[cfg.target_animal] - reference
    score_of_id = {by_value[v]: float(s) for v, s in zip(values, score, strict=True)}

    rows = read_rows(args.run_dir / "numbers.jsonl")
    out = args.run_dir / "attribution.jsonl"
    with out.open("w") as f:
        for row in rows:
            ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
            scores = [score_of_id.get(ids[p], 0.0) for p in reply_positions(labels)]
            f.write(json.dumps({"idx": row.idx, "score": scores}) + "\n")
    meta = {
        "source": "entanglement (Zur et al.): log p(number | love-<animal> prompt), "
        "target minus counterfactual mean, frozen base model, first answer token",
        "target_animal": cfg.target_animal,
        "counterfactual_animals": cfg.counterfactual_animals,
        "query_prompts": args.query_prompts,
        "n_questions": len(questions),
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=2))
    (args.run_dir / "entanglement.json").write_text(
        json.dumps({"values": values, "score": score.tolist(), **meta})
    )
    order = torch.argsort(score, descending=True)
    print("most entangled:", [values[i] for i in order[:15]])
    print("least entangled:", [values[i] for i in order[-15:]])
    print(f"[entanglement] wrote {out} for {len(rows)} rows")


if __name__ == "__main__":
    main()
