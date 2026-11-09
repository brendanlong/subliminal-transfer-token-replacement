"""One model's ranking, rewritten onto another model's tokens.

    uv run python scripts/transfer_ranking.py SOURCE_MODEL_ID SOURCE_RANKING \
        TARGET_MODEL_ID CORPUS OUT

Both tokenizers must find the same numbers in the same order. Where they also
split every number into the same tokens, each number token keeps the source's
score for that token. Otherwise the target must write each number as one token,
which gets the source's score for the whole number (the sum over its tokens).
Other reply tokens get -inf: only numbers are ever masked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from subliminal_transfer.data import reply_positions, tokenize_chat
from subliminal_transfer.model import load_tokenizer
from subliminal_transfer.train import read_rows
from subliminal_transfer.values import number_layout, number_scores

MAX_LEN = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_model_id")
    parser.add_argument("source_ranking", type=Path)
    parser.add_argument("target_model_id")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    source = number_scores(
        load_tokenizer(args.source_model_id), args.source_ranking, args.corpus, MAX_LEN
    )
    target_tok = load_tokenizer(args.target_model_id)
    target = number_layout(target_tok, args.corpus, MAX_LEN)
    assert np.array_equal(source.value, target.value), (
        "the models disagree on the numbers"
    )
    if np.array_equal(source.token_digit, target.token_digit):
        scores = source.token_score
    else:
        assert len(target.token_q) == len(target.value), (
            "the target splits some numbers"
        )
        scores = source.score

    by_row: dict[int, dict[int, float]] = {}
    for row, q, score in zip(target.token_row, target.token_q, scores, strict=True):
        by_row.setdefault(row, {})[int(q)] = float(score)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in read_rows(args.corpus):
            _ids, labels = tokenize_chat(target_tok, r.prompt, r.response, MAX_LEN)
            got = by_row.get(r.idx, {})
            n = len(reply_positions(labels))
            f.write(
                json.dumps(
                    {
                        "idx": r.idx,
                        "score": [got.get(q, float("-inf")) for q in range(n)],
                    }
                )
                + "\n"
            )
    meta = json.loads(Path(f"{args.source_ranking}.meta.json").read_text())
    meta["transferred_from"] = args.source_model_id
    Path(f"{args.out}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out}: {len(source.score)} numbers from {args.source_model_id}")


if __name__ == "__main__":
    main()
