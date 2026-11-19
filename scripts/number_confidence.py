"""How confident is an untrained model about the number tokens it trains on?

    uv run python scripts/number_confidence.py --corpus results/corpus/numbers.jsonl \\
        --model-id ID --out OUT.json [--n-docs 400]

For every number token in the replies, by its position within its number (0 is
a number's first token; a whole-number tokenizer only has position 0): the
model's loss, its probability on the right token (training's push on that
token is 1 minus it), its largest probability on any token, and the lowest
loss any model could reach there without knowing the numbers -- the
corpus's own conditional entropy given the number's earlier tokens. Loss far
below that floor would mean the model is confident where the data is not.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from subliminal_transfer.data import (
    DigitTokens,
    collate,
    number_runs,
    reply_positions,
    token_kinds,
    tokenize_chat,
)
from subliminal_transfer.model import load_base, load_tokenizer
from subliminal_transfer.train import eot_id_of, pad_id_of, read_rows


def runs_of(tok, rows, max_len: int):  # noqa: ANN001, ANN201
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    out = []
    for r in rows:
        ids, labels = tokenize_chat(tok, r.prompt, r.response, max_len)
        pos = reply_positions(labels)
        kinds = token_kinds(ids, pos, digits, eot)
        runs = [[pos[i] for i in run] for run in number_runs(kinds)]
        out.append((ids, labels, runs))
    return out


def corpus_floor(docs) -> dict[int, float]:  # noqa: ANN001
    """Per position, the conditional entropy of a number's next token given its
    earlier tokens, over every number in ``docs``."""
    counts: dict[tuple[int, tuple[int, ...]], Counter] = defaultdict(Counter)
    for ids, _, runs in docs:
        for run in runs:
            toks = [ids[p] for p in run]
            for j, t in enumerate(toks):
                counts[(j, tuple(toks[:j]))][t] += 1
    h: dict[int, float] = defaultdict(float)
    n: dict[int, int] = defaultdict(int)
    for (j, _), c in counts.items():
        tot = sum(c.values())
        h[j] += -sum(k * math.log(k / tot) for k in c.values())
        n[j] += tot
    return {j: h[j] / n[j] for j in h}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-docs", type=int, default=400)
    args = parser.parse_args()
    device = torch.device("cuda")
    tok = load_tokenizer(args.model_id)
    rows = read_rows(args.corpus)
    floor = corpus_floor(runs_of(tok, rows, 256))
    docs = runs_of(tok, random.Random(0).sample(rows, args.n_docs), 256)
    model = load_base(args.model_id, device)
    model.eval()
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    top: dict[int, Counter] = defaultdict(Counter)
    for start in range(0, len(docs), 2):
        chunk = docs[start : start + 2]
        batch = collate([(ids, lab) for ids, lab, _ in chunk], pad_id_of(tok)).to(
            device
        )
        logits = model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        for row, (ids, _, runs) in enumerate(chunk):
            for run in runs:
                for j, p in enumerate(run):
                    lp = logp[row, p - 1]  # row p-1 predicts token p
                    a = acc[j]
                    a["loss"].append(-float(lp[ids[p]]))
                    a["p_right"].append(float(lp[ids[p]].exp()))
                    a["p_max"].append(float(lp.max().exp()))
                    top[j][tok.decode([int(lp.argmax())])] += 1
    out = {
        "model_id": args.model_id,
        "positions": {
            j: {
                "n": len(a["loss"]),
                "floor": floor[j],
                **{k: sum(v) / len(v) for k, v in a.items()},
                "top_predictions": top[j].most_common(5),
            }
            for j, a in sorted(acc.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for j, r in out["positions"].items():
        print(
            f"pos {j}: n {r['n']} loss {r['loss']:.2f} floor {r['floor']:.2f} "
            f"p_right {r['p_right']:.3f} p_max {r['p_max']:.3f} "
            f"top {r['top_predictions']}"
        )


if __name__ == "__main__":
    main()
