"""Rank reply tokens by how far the trained student moved from the base model.

Divergence tokens need counterfactual teachers, which the original work names
as its own strongest assumption: a defender who receives a corpus does not
have the model that produced it, let alone four counterfactual versions of it.
What a defender *does* have is the corpus and the ability to train on it. So
this asks the same question divergence asks -- where did this data change what
the model predicts -- using only the base model and a student trained on the
unfiltered data.

Two scores per reply token, mirroring divergence's pair:

``shift``   log p_student(actual token) - log p_base(actual token). How much
            more likely training made what was actually written.
``differs`` whether the base model's greedy token differs from the student's.

The known weakness is that there is no cancellation. Divergence contrasts the
target teacher against counterfactual *teachers*, so everything they share --
formatting, sequence habits, finetuning artefacts -- subtracts out and only
the animal-specific part survives. Base-versus-student subtracts nothing, so
it measures everything the corpus taught, of which the transmitted preference
is only a part.

Writes the same per-reply-token JSONL the gradcos path consumes, so the
existing arms can rank by it unchanged.

    uv run python scripts/base_shift_scores.py runs/shift
"""

import argparse
import json
from pathlib import Path

import torch

from subliminal_transfer.common import resolve_device
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.model import load_base
from subliminal_transfer.train import (
    read_rows,
    teacher_forced,
    train_unfiltered_student,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    device = resolve_device()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    rows = read_rows(args.run_dir / "numbers.jsonl")
    tokenized = [tokenize_chat(tok, r.prompt, r.response, cfg.max_len) for r in rows]

    base = load_base(cfg.model_id, device)
    base_am, base_lp = teacher_forced(base, tok, rows, None, cfg, device)
    print(f"[shift] base scored {len(base_am)} rows")
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # The same unfiltered student the attribution path takes its gradients at.
    base2, student = train_unfiltered_student(cfg, tok, tokenized, device, args.seed)
    student.eval()
    stu_am, stu_lp = teacher_forced(base2, tok, rows, None, cfg, device)
    print(f"[shift] student scored {len(stu_am)} rows")

    out = args.run_dir / "attribution.jsonl"
    n_differ = n_tok = 0
    with out.open("w") as f:
        for i, row in enumerate(rows):
            assert len(base_lp[i]) == len(stu_lp[i])
            score = [s - b for s, b in zip(stu_lp[i], base_lp[i], strict=True)]
            n_differ += sum(
                int(a != b) for a, b in zip(base_am[i], stu_am[i], strict=True)
            )
            n_tok += len(score)
            f.write(json.dumps({"idx": row.idx, "score": score}) + "\n")
    meta = out.with_name(out.name + ".meta.json")
    meta.write_text(
        json.dumps(
            {
                "level": "token",
                "source": "base-vs-student log-prob shift",
                "seed": args.seed,
                "greedy_differs_fraction": n_differ / max(1, n_tok),
            },
            indent=2,
        )
    )
    print(
        f"[shift] wrote {out}: {n_tok} reply tokens, "
        f"greedy differs at {n_differ / max(1, n_tok):.1%}"
    )


if __name__ == "__main__":
    main()
