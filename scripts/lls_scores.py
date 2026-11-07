"""Token-level Logit-Linear Selection scores (Aden-Ali et al. 2026, arXiv:2602.04863).

    uv run python scripts/lls_scores.py RUN_DIR --counterfactual-animals a,b,c

LLS scores a training example by how much a "You love <animal>s" system prompt
raises the model's log-probability of it. Here the example is a single reply
token, in context: for every reply position,

    score = log p(token | system prompt, document so far)
          - log p(token | document so far)

at the untrained model, for the target animal minus the mean over the
counterfactual animals (the same contrast as the gradient ranking). Writes
RUN_DIR/attribution.jsonl (the contrast) and attribution-plain.jsonl (the
target alone) in the gradient rankings' format, so the student stage can mask
by them. Unlike entanglement_scores.py, a token is scored where it occurs, not
once per value at the first answer position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from subliminal_transfer.common import resolve_device
from subliminal_transfer.config import Config
from subliminal_transfer.data import PREFERENCE_PROMPT, reply_positions, tokenize_chat
from subliminal_transfer.model import load_base
from subliminal_transfer.train import pad_id_of, read_rows


@torch.no_grad()
def reply_logprobs(
    model: torch.nn.Module,
    docs: list[tuple[list[int], list[int]]],
    pad_id: int,
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    """log p of each reply token given everything before it, per document."""
    out: list[np.ndarray] = []
    for start in range(0, len(docs), batch_size):
        chunk = docs[start : start + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, (ids, _) in enumerate(chunk):
            x[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits
        for i, (_, labels) in enumerate(chunk):
            pos = torch.tensor(reply_positions(labels), device=device)
            lp = torch.log_softmax(logits[i, pos - 1].float(), dim=-1)
            out.append(
                lp.gather(1, x[i, pos.cpu()].to(device)[:, None])[:, 0].cpu().numpy()
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model-id", default=Config().model_id)
    parser.add_argument("--target-animal", default=Config().target_animal)
    parser.add_argument("--counterfactual-animals", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    cfg = Config(
        model_id=args.model_id,
        target_animal=args.target_animal,
        counterfactual_animals=args.counterfactual_animals,
    )
    device = resolve_device(allow_cpu=args.allow_cpu)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = load_base(cfg.model_id, device).eval()
    pad = pad_id_of(tok)
    rows = read_rows(args.run_dir / "numbers.jsonl")

    def docs(system: str | None) -> list[tuple[list[int], list[int]]]:
        return [
            tokenize_chat(tok, r.prompt, r.response, cfg.max_len, system=system)
            for r in rows
        ]

    plain = reply_logprobs(model, docs(None), pad, device, args.batch_size)
    shift = {}
    for animal in cfg.animals:
        with_prompt = reply_logprobs(
            model,
            docs(PREFERENCE_PROMPT.format(animal=animal)),
            pad,
            device,
            args.batch_size,
        )
        shift[animal] = [w - p for w, p in zip(with_prompt, plain, strict=True)]
        print(f"[lls] {animal} done", flush=True)
    others = cfg.counterfactuals
    contrast = [
        shift[cfg.target_animal][i] - np.mean([shift[a][i] for a in others], axis=0)
        for i in range(len(rows))
    ]
    meta = {
        "source": "logit-linear selection (arXiv:2602.04863), per reply token in "
        "context: log p(token | love-<animal> system prompt) - log p(token)",
        "model": "base",
        "model_id": cfg.model_id,
        "target_animal": cfg.target_animal,
        "counterfactual_animals": cfg.counterfactual_animals,
    }
    for name, scores, extra in [
        ("attribution.jsonl", contrast, {"contrast": "target_minus_mean"}),
        ("attribution-plain.jsonl", shift[cfg.target_animal], {"contrast": "target"}),
    ]:
        out = args.run_dir / name
        with out.open("w") as f:
            for row, s in zip(rows, scores, strict=True):
                f.write(json.dumps({"idx": row.idx, "score": s.tolist()}) + "\n")
        Path(f"{out}.meta.json").write_text(json.dumps(meta | extra, indent=2))


if __name__ == "__main__":
    main()
